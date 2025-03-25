import MetaTrader5 as mt5
import logging
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from functools import wraps
from config import (
    SYMBOL, RISK_PER_TRADE, MIN_STOPLOSS_PIPS, MAX_SPREAD, 
    MAX_POSITIONS, MAX_DAILY_RISK, TIMEFRAMES, STRATEGY_SETTINGS,
    TRADING_START_HOUR, TRADING_END_HOUR
)
from mt5_connector import (
    connect_mt5, disconnect_mt5, get_account_info, 
    get_open_positions, get_symbol_info, open_order, 
    is_connected, reconnect_if_needed
)
from strategy import find_trade_signal

# Настройка логгера
logger = logging.getLogger(__name__)

# Блокировка для потокобезопасности
_trade_lock = threading.RLock()

try:
    from trade_journal import TradeJournal
    TRADE_JOURNAL_AVAILABLE = True
except ImportError:
    TRADE_JOURNAL_AVAILABLE = False
    logger.warning("Модуль trade_journal не найден. Расширенное логирование сделок отключено")

# Глобальный экземпляр журнала сделок
_trade_journal = None

# Кэш для хранения информации о символах
_symbol_info_cache = {}

# Ограничитель для защиты от слишком частых запросов
_rate_limiter = {
    "last_trade_time": None,
    "min_interval_seconds": 5,  # Минимальный интервал между сделками
    "today_risk": 0.0,         # Отслеживание риска за день
    "today_date": None,        # Текущая дата для отслеживания риска
    "executed_trades": []      # История выполненных сделок за сессию
}

def get_trade_journal(symbol=SYMBOL, account_id=None, init_if_none=True):
    """
    Получение глобального экземпляра журнала сделок
    
    Параметры:
    symbol (str): Торговый символ
    account_id (int, optional): ID торгового счета
    init_if_none (bool): Инициализировать журнал, если он не существует
    
    Возвращает:
    TradeJournal: Экземпляр журнала сделок или None, если журнал недоступен
    """
    global _trade_journal
    
    if not TRADE_JOURNAL_AVAILABLE:
        return None
    
    # Проверяем настройку в конфиге
    try:
        from config import TRADE_JOURNAL_ENABLED
        if not TRADE_JOURNAL_ENABLED:
            return None
    except ImportError:
        # Если настройка не определена, считаем что журнал включен
        pass
    
    if _trade_journal is None and init_if_none:
        try:
            # Если account_id не указан, попробуем получить из MT5
            if account_id is None:
                account_info = get_account_info()
                if account_info:
                    account_id = account_info.get('login')
            
            _trade_journal = TradeJournal(symbol, account_id)
            logger.info(f"Инициализирован журнал сделок для {symbol}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации журнала сделок: {str(e)}")
            return None
    
    return _trade_journal

def retry_on_error(max_attempts=3, retry_delay=2):
    """
    Декоратор для повторных попыток выполнения функции при ошибке
    
    Параметры:
    max_attempts (int): Максимальное количество попыток
    retry_delay (int): Задержка между попытками в секундах
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Ошибка при выполнении {func.__name__} (попытка {attempt}/{max_attempts}): {str(e)}")
                    if attempt < max_attempts:
                        logger.info(f"Повторная попытка через {retry_delay} секунд...")
                        time.sleep(retry_delay)
                        # Проверка подключения перед следующей попыткой
                        if not is_connected():
                            reconnect_if_needed()
            
            # Если все попытки не удались
            logger.error(f"Не удалось выполнить {func.__name__} после {max_attempts} попыток. Последняя ошибка: {str(last_exception)}")
            return None
        return wrapper
    return decorator

def get_cached_symbol_info(symbol=SYMBOL, max_age_seconds=3600):
    """
    Получение информации о символе с кэшированием для оптимизации
    
    Параметры:
    symbol (str): Торговый символ
    max_age_seconds (int): Максимальный возраст кэша в секундах
    
    Возвращает:
    dict: Информация о символе или None в случае ошибки
    """
    global _symbol_info_cache
    
    current_time = time.time()
    
    # Проверяем кэш
    if symbol in _symbol_info_cache:
        cache_time, symbol_info = _symbol_info_cache[symbol]
        # Если кэш не устарел, возвращаем данные из кэша
        if current_time - cache_time < max_age_seconds:
            return symbol_info
    
    # Если кэш устарел или отсутствует, получаем свежую информацию
    symbol_info = get_symbol_info(symbol)
    if symbol_info:
        # Обновляем кэш
        _symbol_info_cache[symbol] = (current_time, symbol_info)
    
    return symbol_info

def reset_daily_risk():
    """Сброс ежедневного риска при смене дня"""
    current_date = datetime.now().date()
    
    if _rate_limiter["today_date"] != current_date:
        _rate_limiter["today_date"] = current_date
        _rate_limiter["today_risk"] = 0.0
        logger.info(f"Сброс ежедневного риска. Новая дата: {current_date}")

@retry_on_error()
def calculate_lot_size(balance, risk_percentage, stop_loss_pips, symbol=SYMBOL, max_risk_per_trade=0.02):
    """
    Улучшенный расчет размера лота в зависимости от риска и стоп-лосса
    с использованием системы управления рисками
    
    Параметры:
    balance (float): Текущий баланс счета
    risk_percentage (float): Процент риска (например, 0.005 для 0.5%)
    stop_loss_pips (int): Размер стоп-лосса в пипсах
    symbol (str): Торговый символ
    max_risk_per_trade (float): Максимальный риск на сделку (например, 0.02 для 2%)
    
    Возвращает:
    tuple: (рассчитанный размер лота, риск в валюте, скорректированный процент риска)
    """
    with _trade_lock:
        if stop_loss_pips <= 0:
            logger.error(f"Некорректное значение стоп-лосса: {stop_loss_pips} пипсов")
            return 0.01, 0.0, 0.0  # Минимальный лот при ошибке
        
        # Получаем информацию о символе
        symbol_info = get_cached_symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Не удалось получить информацию о символе {symbol}")
            return 0.01, 0.0, 0.0  # Минимальный лот при ошибке
        
        # Используем риск-менеджер для расчета размера позиции, если доступен
        try:
            from risk_manager import get_risk_manager
            
            # Инициализируем риск-менеджер с текущим балансом
            risk_manager = get_risk_manager(account_balance=balance)
            
            # Если риск-менеджер доступен, используем его для расчета
            if risk_manager is not None:
                # Ограничиваем риск
                adjusted_risk = min(risk_percentage, max_risk_per_trade)
                
                # Рассчитываем размер позиции с помощью риск-менеджера
                lot_size, risk_amount, actual_risk = risk_manager.calculate_position_size(
                    stop_loss_pips, 0, symbol_info, max_risk_override=adjusted_risk
                )
                
                # Логируем результат
                logger.info(f"Расчет лота через риск-менеджер: баланс=${balance}, "
                           f"риск={adjusted_risk*100:.2f}%, SL={stop_loss_pips} пипсов, "
                           f"лот={lot_size}, риск=${risk_amount:.2f}")
                
                return lot_size, risk_amount, actual_risk
                
        except ImportError:
            logger.debug("Модуль риск-менеджера недоступен, используем стандартный метод расчета")
        except Exception as e:
            logger.warning(f"Ошибка при использовании риск-менеджера: {str(e)}. Используем стандартный метод расчета")
        
        # Стандартный метод расчета, если риск-менеджер недоступен или произошла ошибка
        
        # Ограничиваем риск
        adjusted_risk = min(risk_percentage, max_risk_per_trade)
        
        # Проверяем ежедневный риск
        reset_daily_risk()
        remaining_daily_risk = MAX_DAILY_RISK - _rate_limiter["today_risk"]
        
        if remaining_daily_risk <= 0:
            logger.warning(f"Достигнут максимальный дневной риск ({MAX_DAILY_RISK*100}%). Торговля остановлена на сегодня.")
            return 0.0, 0.0, 0.0
        
        # Ограничиваем риск оставшимся дневным лимитом
        adjusted_risk = min(adjusted_risk, remaining_daily_risk)
        
        # Получаем значение пункта (point) и его стоимость (tick_value)
        point = symbol_info["point"]
        contract_size = symbol_info["trade_contract_size"]
        
        # Для валютных пар типа EUR/USD точка обычно 0.00001, а пипс 0.0001 (10 пунктов)
        pip_value = point * 10
        
        # Расчет стоимости 1 пипса для лота 1.0
        price = symbol_info.get("bid", 0)  # Используем bid как текущую цену
        if price <= 0:
            logger.error(f"Некорректная цена для {symbol}: {price}")
            return 0.01, 0.0, adjusted_risk  # Минимальный лот при ошибке
        
        tick_value = symbol_info["trade_tick_value"]
        tick_size = symbol_info["trade_tick_size"]
        
        # Стоимость 1 пипса для лота 1.0
        pip_cost = (pip_value / tick_size) * tick_value
        
        # Определяем валюту счета и символа
        account_info = get_account_info()
        if account_info is None:
            logger.error("Не удалось получить информацию о счете")
            return 0.01, 0.0, adjusted_risk  # Минимальный лот при ошибке
            
        account_currency = account_info["currency"]
        profit_currency = symbol_info["currency_profit"]
        
        # Если валюты разные, возможно потребуется конвертация
        conversion_rate = 1.0  # По умолчанию
        if account_currency != profit_currency:
            try:
                # Пытаемся найти курс конвертации через кросс-курс с USD
                if account_currency != "USD" and profit_currency != "USD":
                    # Ищем символы для конвертации
                    usd_account = account_currency + "USD"
                    usd_profit = "USD" + profit_currency
                    
                    # Пробуем получить курсы
                    usd_account_info = get_cached_symbol_info(usd_account)
                    usd_profit_info = get_cached_symbol_info(usd_profit)
                    
                    if usd_account_info and usd_profit_info:
                        account_rate = usd_account_info.get("bid", 1.0)
                        profit_rate = usd_profit_info.get("bid", 1.0)
                        conversion_rate = account_rate * profit_rate
                    else:
                        # Если не нашли символы, оставляем коэффициент 1.0
                        logger.warning(f"Не удалось найти символы для конвертации {account_currency} -> {profit_currency}")
                        conversion_rate = 1.0
                elif account_currency == "USD":
                    # USD -> другая валюта
                    conversion_symbol = "USD" + profit_currency
                    conversion_info = get_cached_symbol_info(conversion_symbol)
                    if conversion_info:
                        conversion_rate = conversion_info.get("bid", 1.0)
                elif profit_currency == "USD":
                    # Другая валюта -> USD
                    conversion_symbol = account_currency + "USD"
                    conversion_info = get_cached_symbol_info(conversion_symbol)
                    if conversion_info:
                        conversion_rate = conversion_info.get("bid", 1.0)
            except Exception as e:
                logger.warning(f"Ошибка при расчете коэффициента конвертации: {str(e)}")
                conversion_rate = 1.0
        
        # Сумма, которой мы готовы рисковать с учетом конвертации
        risk_amount = balance * adjusted_risk * conversion_rate
        
        # Расчет размера лота
        # risk_amount = lot_size * stop_loss_pips * pip_cost
        lot_size = risk_amount / (stop_loss_pips * pip_cost)
        
        # Ограничения на минимальный и максимальный лот
        min_lot = symbol_info["volume_min"]
        max_lot = symbol_info["volume_max"]
        step = symbol_info["volume_step"]
        
        # Округляем до ближайшего шага объема
        lot_size = round(lot_size / step) * step
        
        # Лог расчета для отладки
        logger.info(f"Расчет лота: balance=${balance}, risk={adjusted_risk*100:.2f}%, SL={stop_loss_pips} пипсов")
        logger.info(f"pip_cost=${pip_cost}, conversion_rate={conversion_rate}, risk_amount=${risk_amount:.2f}, lot={lot_size}")
        
        # Ограничиваем лот в пределах допустимых значений
        lot_size = max(min(lot_size, max_lot), min_lot)
        
        # Возвращаем размер лота, риск в валюте и скорректированный процент риска
        return lot_size, risk_amount, adjusted_risk

def check_trade_conditions(symbol=SYMBOL, max_spread_multiplier=1.5):
    """
    Проверка условий для торговли (спред, время сессии, и т.д.)
    
    Параметры:
    symbol (str): Торговый символ
    max_spread_multiplier (float): Множитель для максимального спреда
    
    Возвращает:
    tuple: (торговля разрешена (bool), причина запрета (str или None))
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        return False, "Нет подключения к MT5"
    
    # Проверяем ограничение частоты торговли
    if _rate_limiter["last_trade_time"] is not None:
        elapsed = time.time() - _rate_limiter["last_trade_time"]
        if elapsed < _rate_limiter["min_interval_seconds"]:
            return False, f"Слишком частые сделки (прошло {elapsed:.1f} сек из минимальных {_rate_limiter['min_interval_seconds']})"
    
    # Проверяем дневной риск
    reset_daily_risk()
    if _rate_limiter["today_risk"] >= MAX_DAILY_RISK:
        return False, f"Достигнут максимальный дневной риск ({MAX_DAILY_RISK*100}%)"
    
    # Проверяем количество открытых позиций
    positions = get_open_positions(symbol)
    if len(positions) >= MAX_POSITIONS:
        return False, f"Достигнуто максимальное количество открытых позиций ({MAX_POSITIONS})"
    
    # Проверяем текущий спред
    symbol_info = get_cached_symbol_info(symbol)
    if symbol_info:
        current_spread = symbol_info.get("spread_current", 0)
        if current_spread == 0 and "ask" in symbol_info and "bid" in symbol_info:
            current_spread = (symbol_info["ask"] - symbol_info["bid"]) / 0.0001  # В пипсах
        
        # Адаптивный максимальный спред, зависящий от волатильности
        max_spread = MAX_SPREAD
        
        # Если есть информация о волатильности, адаптируем максимальный спред
        if "atr20" in symbol_info:
            atr = symbol_info["atr20"]
            max_spread = MAX_SPREAD * (1 + atr * 10)  # Адаптивный спред на основе ATR
        
        # Применяем множитель
        max_spread *= max_spread_multiplier
        
        if current_spread > max_spread:
            return False, f"Спред {current_spread:.1f} пипсов превышает максимум {max_spread:.1f}"
    
    # Проверяем время торговой сессии
    now = datetime.now()
    
    # Проверка рабочих дней недели (0-6, где 0 - понедельник, 6 - воскресенье)
    if now.weekday() >= 5:  # Суббота и воскресенье
        return False, f"Нерабочий день недели ({now.strftime('%A')})"
    
    # Проверка времени торговой сессии используя настройки из конфига
    if now.hour < TRADING_START_HOUR or now.hour >= TRADING_END_HOUR:
        return False, f"Нерабочее время ({now.strftime('%H:%M')}), торговля разрешена с {TRADING_START_HOUR}:00 до {TRADING_END_HOUR}:00"
    
    # Все проверки пройдены
    return True, None

@retry_on_error()
def execute_trade(from_signal=None, symbol=SYMBOL, risk_per_trade=None, check_conditions=True):
    """
    Выполняет торговую операцию на основе сигнала с расширенными проверками и логированием
    
    Параметры:
    from_signal (dict, optional): Предопределенный сигнал. Если None, будет искать новый.
    symbol (str): Торговый символ. По умолчанию из конфига.
    risk_per_trade (float): Риск на сделку. Если None, используется RISK_PER_TRADE из конфига.
    check_conditions (bool): Проверять ли условия для торговли
    
    Возвращает:
    dict: Информация о выполненной сделке или None в случае ошибки
    """
    with _trade_lock:
        # Инициализируем Telegram-нотификатор, если включен
        try:
            from telegram_notifier import initialize_telegram_notifier
            from config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            
            if TELEGRAM_ENABLED:
                initialize_telegram_notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        except Exception as e:
            logger.warning(f"Не удалось инициализировать Telegram-нотификатор: {str(e)}")

        # Инициализируем риск-менеджер, если он доступен
        risk_manager = None
        try:
            from risk_manager import get_risk_manager
            # Получаем информацию о счете для инициализации риск-менеджера
            account_info = get_account_info()
            if account_info:
                balance = account_info["balance"]
                risk_manager = get_risk_manager(account_balance=balance)
                logger.info("Риск-менеджер инициализирован")
        except ImportError:
            logger.debug("Модуль риск-менеджера недоступен")
        except Exception as e:
            logger.warning(f"Ошибка при инициализации риск-менеджера: {str(e)}")

        # Используем значение риска из параметров или из конфига
        if risk_per_trade is None:
            risk_per_trade = RISK_PER_TRADE
        
        # Проверяем условия для торговли
        if check_conditions:
            can_trade, reason = check_trade_conditions(symbol)
            if not can_trade:
                logger.info(f"Торговля не разрешена: {reason}")
                return None
            
            # Дополнительная проверка через риск-менеджер, если он доступен
            if risk_manager:
                can_trade_risk, reason_risk = risk_manager.can_trade()
                if not can_trade_risk:
                    logger.warning(f"Торговля остановлена риск-менеджером: {reason_risk}")
                    return None
        
        from data_fetcher import get_historical_data, get_multi_timeframe_data
        
        # Если нет предопределенного сигнала, ищем новый
        if from_signal is None:
            # Получаем данные и ищем сигнал
            df = get_historical_data(symbol, timeframe="M5", num_candles=100, use_cache=False)
            signal = find_trade_signal(df) if df is not None else None
            
            # Сортируем таймфреймы от старшего к младшему
            sorted_timeframes = sorted(TIMEFRAMES, key=lambda x: TIMEFRAMES.index(x), reverse=True)
            
            # Загружаем данные для всех таймфреймов
            multi_tf_data = get_multi_timeframe_data(symbol, sorted_timeframes, use_cache=True)
            
            if multi_tf_data is None:
                logger.error("Не удалось загрузить данные для анализа")
                return None
            
            # Проверяем сигналы, начиная со старших таймфреймов
            signal = None
            for tf in sorted_timeframes:
                if tf in multi_tf_data:
                    df = multi_tf_data[tf]
                    if df is not None and len(df) >= 30:  # Проверяем, что достаточно данных для анализа
                        temp_signal = find_trade_signal(df)
                        if temp_signal:
                            # Добавляем информацию о таймфрейме к сигналу
                            temp_signal['tf'] = tf
                            signal = temp_signal
                            logger.info(f"Найден сигнал на таймфрейме {tf}")
                            break
        else:
            signal = from_signal
        
        if not signal:
            logger.info("Сигналов нет, сделка не открыта")
            return None
        
        order_type = signal["type"]
        entry_price = signal["level"]
        
        # Используем стоп-лосс из сигнала, если он есть, иначе используем минимальный
        if "stop_loss" in signal and signal["stop_loss"] > 0:
            stop_loss = signal["stop_loss"]
            stop_loss_pips = int(stop_loss / 0.0001)  # Перевод в пипсы
        else:
            stop_loss_pips = MIN_STOPLOSS_PIPS if MIN_STOPLOSS_PIPS else 30  # По умолчанию 30 пипсов
            stop_loss = stop_loss_pips * 0.0001
        
        # Получаем информацию о счете
        account_info = get_account_info()
        if account_info is None:
            logger.error("Не удалось получить информацию о счете")
            return None
        
        balance = account_info["balance"]
        
        # Определяем тейк-профит с использованием риск-менеджера, если доступен
        if risk_manager:
            # Получаем рекомендованное соотношение риск/доходность
            recommendations = risk_manager.get_trade_recommendations()
            recommended_rr = recommendations.get("recommended_r_r_ratio")
            
            # Используем его, если доступно
            if recommended_rr:
                take_profit_pips = risk_manager.calculate_take_profit(stop_loss_pips, recommended_rr)
                take_profit = take_profit_pips * 0.0001
                logger.info(f"Тейк-профит рассчитан риск-менеджером: {take_profit_pips} пипсов (R/R = {recommended_rr})")
            else:
                # Аналогично с тейк-профитом, если нет в сигнале
                if "take_profit" in signal and signal["take_profit"] > 0:
                    take_profit = signal["take_profit"]
                    take_profit_pips = int(take_profit / 0.0001)  # Перевод в пипсы
                else:
                    take_profit_pips = stop_loss_pips * 3  # TP = 3x SL
                    take_profit = take_profit_pips * 0.0001
        else:
            # Стандартный расчет, если риск-менеджер недоступен
            # Аналогично с тейк-профитом, если нет в сигнале
            if "take_profit" in signal and signal["take_profit"] > 0:
                take_profit = signal["take_profit"]
                take_profit_pips = int(take_profit / 0.0001)  # Перевод в пипсы
            else:
                take_profit_pips = stop_loss_pips * 3  # TP = 3x SL
                take_profit = take_profit_pips * 0.0001
        
        # Рассчитываем размер лота, риск в валюте и скорректированный процент риска
        lot_size, risk_amount, adjusted_risk = calculate_lot_size(
            balance, risk_per_trade, stop_loss_pips, symbol
        )
        
        # Если лот равен 0, значит торговля невозможна
        if lot_size <= 0:
            logger.warning("Торговля невозможна: рассчитанный размер лота <= 0")
            return None
        
        # Проверяем, достаточно ли средств для открытия сделки
        symbol_info = get_cached_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Не удалось получить информацию о символе {symbol}")
            return None
        
        # Обновляем ежедневный риск
        _rate_limiter["today_risk"] += adjusted_risk
        
        # Устанавливаем цену входа в зависимости от текущих цен
        if order_type == "buy":
            price = symbol_info.get("ask", entry_price)
            sl = price - stop_loss
            tp = price + take_profit
        else:  # sell
            price = symbol_info.get("bid", entry_price)
            sl = price + stop_loss
            tp = price - take_profit
        
        # Формируем комментарий для ордера
        setup_type = signal.get("setup", "Standard")
        comment = f"Algo_{setup_type}"[:31]  # Ограничиваем длиной 31 символ
        
        # Открываем ордер
        order_info = open_order(
            symbol=symbol,
            order_type=order_type,
            volume=lot_size,
            price=None,  # Используем текущую рыночную цену
            sl=sl,
            tp=tp,
            deviation=10,
            magic=123456,
            comment=comment
        )
        
        if order_info is None:
            logger.error("Не удалось открыть ордер")
            # Отменяем обновление ежедневного риска, так как сделка не открылась
            _rate_limiter["today_risk"] -= adjusted_risk
            return None
        
        # Обновляем время последней сделки
        _rate_limiter["last_trade_time"] = time.time()
        
        # Добавляем сделку в историю
        _rate_limiter["executed_trades"].append({
            "time": datetime.now(),
            "symbol": symbol,
            "type": order_type,
            "volume": lot_size,
            "price": price,
            "sl": sl,
            "tp": tp,
            "risk_amount": risk_amount,
            "risk_percent": adjusted_risk * 100,
            "setup": setup_type,
            "ticket": order_info["ticket"]
        })
        
        # Регистрируем сделку в риск-менеджере, если он доступен
        if risk_manager:
            # Риск-менеджер ожидает отрицательное значение для риска, так как это потенциальный убыток
            risk_manager.register_trade_result(0, -risk_amount, balance)
            logger.info("Сделка зарегистрирована в риск-менеджере")
        
        # Логируем успешное открытие сделки
        logger.info(
            f"Сделка {order_type} открыта: Символ {symbol}, Лот {lot_size}, "
            f"Цена {price}, SL {sl} ({stop_loss_pips} пипс), TP {tp} ({take_profit_pips} пипс)"
        )
        logger.info(
            f"Сетап: {setup_type}, Риск: ${risk_amount:.2f} ({adjusted_risk*100:.2f}%), "
            f"Дневной риск: {_rate_limiter['today_risk']*100:.2f}% из {MAX_DAILY_RISK*100:.2f}%"
        )
        
        # Добавляем запись в журнал сделок, если он доступен
        journal = get_trade_journal(symbol)
        if journal is not None:
            try:
                # Создаем запись для журнала с дополнительной информацией
                journal_entry = {
                    'ticket': order_info["ticket"],
                    'symbol': symbol,
                    'order': order_type,
                    'lot_size': lot_size,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': sl,
                    'take_profit': tp,
                    'risk_amount': risk_amount,
                    'risk_percent': adjusted_risk * 100,
                    'setup': setup_type,
                    'tf': signal.get('tf') if isinstance(signal, dict) and 'tf' in signal else None,
                    'comment': comment
                }
                
                # Добавляем запись в журнал
                journal.add_trade(journal_entry)
                logger.info(f"Сделка добавлена в расширенный журнал (тикет: {order_info['ticket']})")
            except Exception as e:
                logger.warning(f"Ошибка при добавлении сделки в журнал: {str(e)}")

        # Возвращаем информацию о сделке
        trade_info = {
            "ticket": order_info["ticket"],
            "symbol": symbol,
            "type": order_type,
            "volume": lot_size,
            "price": price,
            "sl": sl,
            "tp": tp,
            "risk_amount": risk_amount,
            "risk_percent": adjusted_risk * 100,
            "setup": setup_type,
            "time": datetime.now()
        }
        
        return trade_info

def get_trading_statistics():
    """
    Получение статистики по торговле за текущую сессию
    
    Возвращает:
    dict: Статистика по торговле
    """
    # Обновляем дневной риск
    reset_daily_risk()
    
    # Получаем открытые позиции
    positions = get_open_positions()
    
    # Рассчитываем статистику
    total_trades = len(_rate_limiter["executed_trades"])
    total_volume = sum(trade["volume"] for trade in _rate_limiter["executed_trades"])
    
    buy_trades = sum(1 for trade in _rate_limiter["executed_trades"] if trade["type"] == "buy")
    sell_trades = sum(1 for trade in _rate_limiter["executed_trades"] if trade["type"] == "sell")
    
    # Группируем по сетапам
    setup_counts = {}
    for trade in _rate_limiter["executed_trades"]:
        setup = trade.get("setup", "Standard")
        setup_counts[setup] = setup_counts.get(setup, 0) + 1
    
    # Собираем статистику по открытым позициям
    total_profit = 0
    total_swap = 0
    total_commission = 0
    
    for pos in positions:
        total_profit += pos.get("profit", 0)
        total_swap += pos.get("swap", 0)
        total_commission += pos.get("commission", 0)
    
    return {
        "date": _rate_limiter["today_date"],
        "daily_risk_used": _rate_limiter["today_risk"] * 100,  # В процентах
        "daily_risk_limit": MAX_DAILY_RISK * 100,  # В процентах
        "total_trades": total_trades,
        "total_volume": total_volume,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "open_positions": len(positions),
        "setups": setup_counts,
        "current_profit": total_profit,
        "current_swap": total_swap,
        "current_commission": total_commission,
        "total_pnl": total_profit + total_swap + total_commission
    }

if __name__ == "__main__":
    # Настраиваем логирование для консоли при запуске скрипта напрямую
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    if is_connected() or connect_mt5():
        # Тестируем расчет размера лота
        account_info = get_account_info()
        if account_info:
            balance = account_info["balance"]
            print(f"Баланс счета: {balance} {account_info['currency']}")
            
            # Рассчитываем размер лота для разных стоп-лоссов
            test_sl_pips = [30, 50, 100]
            for sl_pips in test_sl_pips:
                lot, risk_amount, adj_risk = calculate_lot_size(
                    balance, RISK_PER_TRADE, sl_pips, SYMBOL
                )
                print(f"Стоп-лосс: {sl_pips} пипсов -> Лот: {lot}, Риск: ${risk_amount:.2f} ({adj_risk*100:.2f}%)")
        
        # Проверяем условия для торговли
        can_trade, reason = check_trade_conditions()
        print(f"Торговля разрешена: {can_trade}")
        if not can_trade:
            print(f"Причина запрета: {reason}")
        
        # Тестируем выполнение торговли, но без реального исполнения
        if can_trade:
            from data_fetcher import get_historical_data
            
            df = get_historical_data(SYMBOL, timeframe="M5", num_candles=100)
            if df is not None:
                signal = find_trade_signal(df)
                if signal:
                    print(f"Найден сигнал: {signal}")
                    print("Симуляция выполнения сделки (без реального исполнения):")
                    
                    # Поменяйте на True, если хотите выполнить реальный ордер
                    execute_real_order = False
                    
                    if execute_real_order:
                        trade_info = execute_trade(signal)
                        if trade_info:
                            print(f"Сделка выполнена: {trade_info}")
                    else:
                        # Симуляция - рассчитываем размер лота без выполнения
                        sl_pips = int(signal.get("stop_loss", 0.0003) / 0.0001)
                        lot, risk_amount, adj_risk = calculate_lot_size(
                            balance, RISK_PER_TRADE, sl_pips, SYMBOL
                        )
                        print(f"Симуляция: Лот {lot}, Риск ${risk_amount:.2f} ({adj_risk*100:.2f}%)")
                        print(f"Тип: {signal['type']}, Уровень: {signal['level']}")
                        print(f"Стоп-лосс: {signal['stop_loss']} ({sl_pips} пипсов)")
                        print(f"Тейк-профит: {signal['take_profit']} ({int(signal['take_profit']/0.0001)} пипсов)")
                        print(f"Сетап: {signal.get('setup', 'Standard')}")
                else:
                    print("Сигналов нет")
        
        disconnect_mt5()
    else:
        print("Не удалось подключиться к MT5")