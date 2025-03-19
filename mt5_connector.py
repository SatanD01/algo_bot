import MetaTrader5 as mt5
import os
import logging
import time
import threading
from datetime import datetime
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, SYMBOL

# Настройка логгера
logger = logging.getLogger(__name__)

# Блокировка для потокобезопасности
_mt5_lock = threading.RLock()

# Состояние подключения
_connection_state = {
    "connected": False,
    "last_check": None,
    "retries": 0,
    "max_retries": 5
}

def is_connected():
    """
    Проверка активного подключения к MT5
    
    Возвращает:
    bool: True, если подключение активно, иначе False
    """
    with _mt5_lock:
        try:
            # Обновляем время последней проверки
            _connection_state["last_check"] = datetime.now()
            
            if mt5.terminal_info() is not None:
                if not _connection_state["connected"]:
                    _connection_state["connected"] = True
                    logger.debug("Соединение с MT5 обнаружено")
                return True
            else:
                if _connection_state["connected"]:
                    _connection_state["connected"] = False
                    logger.warning("Соединение с MT5 потеряно")
                return False
        except Exception as e:
            _connection_state["connected"] = False
            logger.error(f"Ошибка при проверке подключения к MT5: {str(e)}")
            return False

def connect_mt5(max_attempts=3, retry_delay=5, force_reconnect=False):
    """
    Подключение к MetaTrader 5 с повторными попытками
    
    Параметры:
    max_attempts (int): Максимальное количество попыток подключения
    retry_delay (int): Задержка между попытками в секундах
    force_reconnect (bool): Принудительное переподключение даже если соединение активно
    
    Возвращает:
    bool: True в случае успешного подключения, иначе False
    """
    with _mt5_lock:
        # Проверка, не подключены ли мы уже
        if not force_reconnect and is_connected():
            logger.info("Уже подключен к MT5")
            return True
        
        # Если предыдущее подключение активно, завершаем его
        if mt5.terminal_info() is not None:
            logger.info("Завершаем предыдущее подключение к MT5")
            mt5.shutdown()
            time.sleep(1)  # Даем немного времени для корректного завершения
        
        # Проверка наличия файла MT5
        if not os.path.exists(MT5_PATH):
            logger.error(f"Файл MT5 не найден: {MT5_PATH}")
            return False
        
        # Попытки подключения
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Попытка подключения к MT5 ({attempt}/{max_attempts})...")
            
            # Инициализация MT5
            try:
                if not mt5.initialize(path=MT5_PATH):
                    error = mt5.last_error()
                    logger.error(f"Ошибка инициализации MT5 (код: {error[0]}): {error[1]}")
                    
                    if attempt < max_attempts:
                        logger.info(f"Повторная попытка через {retry_delay} секунд...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        _connection_state["connected"] = False
                        return False
            except Exception as e:
                logger.error(f"Исключение при инициализации MT5: {str(e)}")
                if attempt < max_attempts:
                    time.sleep(retry_delay)
                    continue
                else:
                    _connection_state["connected"] = False
                    return False
            
            # Авторизация
            try:
                logger.info(f"Авторизация на сервере {MT5_SERVER} с логином {MT5_LOGIN}")
                authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
                
                if not authorized:
                    error = mt5.last_error()
                    logger.error(f"Ошибка авторизации в MT5 (код: {error[0]}): {error[1]}")
                    
                    # Завершаем текущую инициализацию перед повторной попыткой
                    mt5.shutdown()
                    
                    if attempt < max_attempts:
                        logger.info(f"Повторная попытка через {retry_delay} секунд...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        _connection_state["connected"] = False
                        return False
            except Exception as e:
                logger.error(f"Исключение при авторизации в MT5: {str(e)}")
                # Завершаем текущую инициализацию
                try:
                    mt5.shutdown()
                except:
                    pass
                
                if attempt < max_attempts:
                    time.sleep(retry_delay)
                    continue
                else:
                    _connection_state["connected"] = False
                    return False
            
            # Проверяем, что подключились к нужному счету
            try:
                account_info = mt5.account_info()
                if account_info is None:
                    logger.error("Не удалось получить информацию о счете после авторизации")
                    mt5.shutdown()
                    
                    if attempt < max_attempts:
                        time.sleep(retry_delay)
                        continue
                    else:
                        _connection_state["connected"] = False
                        return False
            except Exception as e:
                logger.error(f"Исключение при получении информации о счете: {str(e)}")
                try:
                    mt5.shutdown()
                except:
                    pass
                
                if attempt < max_attempts:
                    time.sleep(retry_delay)
                    continue
                else:
                    _connection_state["connected"] = False
                    return False
            
            # Успешное подключение
            _connection_state["connected"] = True
            _connection_state["retries"] = 0
            
            logger.info(f"Успешное подключение к MetaTrader 5 (Счет: {account_info.login}, Сервер: {MT5_SERVER})")
            
            # Выводим информацию о терминале
            try:
                terminal_info = mt5.terminal_info()
                if terminal_info is not None:
                    # Используем только доступные атрибуты
                    logger.info(f"Терминал MT5: {terminal_info.name}")
                    
                    # Пробуем получить дополнительную информацию безопасно
                    try:
                        terminal_build = terminal_info.build if hasattr(terminal_info, 'build') else "Неизвестно"
                        logger.info(f"Сборка терминала: {terminal_build}")
                        
                        # Проверяем другие полезные атрибуты
                        if hasattr(terminal_info, 'connected'):
                            logger.info(f"Состояние подключения к серверу брокера: {'Подключен' if terminal_info.connected else 'Не подключен'}")
                        
                        if hasattr(terminal_info, 'community_account'):
                            logger.info(f"MQL5 аккаунт: {'Да' if terminal_info.community_account else 'Нет'}")
                        
                        if hasattr(terminal_info, 'community_connection'):
                            logger.info(f"Подключение к MQL5 сообществу: {'Да' if terminal_info.community_connection else 'Нет'}")
                        
                        if hasattr(terminal_info, 'dlls_allowed'):
                            logger.info(f"DLL импорт разрешен: {'Да' if terminal_info.dlls_allowed else 'Нет'}")
                        
                        if hasattr(terminal_info, 'trade_allowed'):
                            logger.info(f"Торговля разрешена: {'Да' if terminal_info.trade_allowed else 'Нет'}")
                    except Exception as e:
                        logger.debug(f"Не удалось получить дополнительную информацию о терминале: {str(e)}")
            except Exception as e:
                logger.warning(f"Не удалось получить информацию о терминале: {str(e)}")
            
            return True
        
        # Если все попытки не удались
        _connection_state["connected"] = False
        return False

def reconnect_if_needed(check_interval_seconds=300):
    """
    Автоматическое переподключение, если соединение потеряно
    
    Параметры:
    check_interval_seconds (int): Интервал проверки подключения в секундах
    
    Возвращает:
    bool: True, если подключение активно или удалось переподключиться, иначе False
    """
    # Проверяем, прошло ли достаточно времени с последней проверки
    if (_connection_state["last_check"] is not None and 
        (datetime.now() - _connection_state["last_check"]).total_seconds() < check_interval_seconds):
        # Если проверка была недавно и соединение активно, то все в порядке
        if _connection_state["connected"]:
            return True
    
    # Проверяем текущее состояние подключения
    if is_connected():
        return True
    
    # Если число попыток превышает лимит, ждем дольше
    if _connection_state["retries"] >= _connection_state["max_retries"]:
        logger.warning(f"Превышено максимальное число попыток подключения ({_connection_state['max_retries']}). Увеличиваем интервал.")
        time.sleep(check_interval_seconds)
        _connection_state["retries"] = 0
    
    # Пытаемся переподключиться
    logger.info("Соединение с MT5 потеряно. Пытаемся переподключиться...")
    _connection_state["retries"] += 1
    
    if connect_mt5(force_reconnect=True):
        logger.info("Успешное переподключение к MT5")
        _connection_state["retries"] = 0
        return True
    else:
        logger.error(f"Не удалось переподключиться к MT5 (попытка {_connection_state['retries']}/{_connection_state['max_retries']})")
        return False

def get_account_info(force_refresh=False):
    """
    Получение детальной информации о счете с обработкой ошибок
    и автоматическим переподключением
    
    Параметры:
    force_refresh (bool): Принудительное обновление информации
    
    Возвращает:
    dict: Словарь с информацией о счете или None в случае ошибки
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error("Нет подключения к MT5, не удалось получить информацию о счете")
        return None
    
    try:
        # Получаем информацию о счете
        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            logger.error(f"Не удалось получить информацию о счете (код: {error[0]}): {error[1]}")
            return None
        
        # Собираем основную информацию
        info = {
            "login": account_info.login,
            "server": account_info.server,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "margin_free": account_info.margin_free,
            "margin_level": account_info.margin_level,
            "leverage": account_info.leverage,
            "currency": account_info.currency
        }
        
        # Добавляем дополнительную информацию
        for attr in ['profit', 'credit', 'name', 'company', 'limit_orders', 'margin_so_mode', 
                    'trade_allowed', 'trade_expert', 'margin_so_call', 'margin_so_so', 
                    'margin_initial', 'margin_maintenance', 'assets', 'liabilities']:
            if hasattr(account_info, attr):
                info[attr] = getattr(account_info, attr)
        
        # Логируем основную информацию
        logger.info(f"Счет {info['login']}: Баланс: {info['balance']} {info['currency']}, "
                    f"Средства: {info['equity']} {info['currency']}, "
                    f"Свободная маржа: {info['margin_free']} {info['currency']}, "
                    f"Уровень маржи: {info['margin_level']}%, "
                    f"Кредитное плечо: 1:{info['leverage']}")
        
        return info
    
    except Exception as e:
        logger.error(f"Исключение при получении информации о счете: {str(e)}")
        return None

def get_open_positions(symbol=None, magic=None, retry_on_error=True):
    """
    Получение открытых позиций с фильтрацией по символу и/или magic
    
    Параметры:
    symbol (str, optional): Торговый символ для фильтрации позиций
    magic (int, optional): Magic номер для фильтрации позиций
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    list: Список открытых позиций или пустой список в случае ошибки
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error("Нет подключения к MT5, не удалось получить позиции")
        return []
    
    try:
        # Формируем фильтр для positions_get
        request = {}
        if symbol is not None:
            request['symbol'] = symbol
        if magic is not None:
            request['magic'] = magic
        
        # Получаем открытые позиции
        positions = mt5.positions_get(**request) if request else mt5.positions_get()
        
        if positions is None:
            error = mt5.last_error()
            # Код 0 означает, что просто нет открытых позиций
            if error[0] == 0:
                filters = []
                if symbol:
                    filters.append(f"символа {symbol}")
                if magic:
                    filters.append(f"magic {magic}")
                filter_str = f" для {' и '.join(filters)}" if filters else ""
                
                logger.info(f"Нет открытых позиций{filter_str}")
                return []
            else:
                logger.error(f"Ошибка при получении позиций (код: {error[0]}): {error[1]}")
                
                # При ошибке пробуем переподключиться и повторить запрос
                if retry_on_error:
                    logger.info("Пробуем переподключиться и повторить запрос")
                    if connect_mt5(force_reconnect=True):
                        return get_open_positions(symbol, magic, False)  # Рекурсивный вызов без повторной попытки
                
                return []
        
        # Преобразуем позиции в список словарей для удобства
        positions_list = []
        for position in positions:
            pos_info = {
                "ticket": position.ticket,
                "symbol": position.symbol,
                "type": "buy" if position.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": position.volume,
                "open_price": position.price_open,
                "current_price": position.price_current,
                "sl": position.sl,
                "tp": position.tp,
                "profit": position.profit,
                "swap": position.swap,
                "commission": position.commission,
                "open_time": datetime.fromtimestamp(position.time),
                "magic": position.magic,
                "comment": position.comment,
                "identifier": position.identifier,
                "reason": position.reason
            }
            
            # Добавляем расчет прибыли в пунктах
            if pos_info["type"] == "buy":
                pos_info["profit_points"] = int((position.price_current - position.price_open) / position.point)
            else:
                pos_info["profit_points"] = int((position.price_open - position.price_current) / position.point)
            
            positions_list.append(pos_info)
        
        # Логируем количество открытых позиций
        filters = []
        if symbol:
            filters.append(f"символа {symbol}")
        if magic:
            filters.append(f"magic {magic}")
        filter_str = f" для {' и '.join(filters)}" if filters else ""
        
        logger.info(f"Открыто {len(positions_list)} позиций{filter_str}")
        
        return positions_list
    
    except Exception as e:
        logger.error(f"Исключение при получении позиций: {str(e)}")
        
        # При ошибке пробуем переподключиться и повторить запрос
        if retry_on_error:
            logger.info("Пробуем переподключиться и повторить запрос")
            if connect_mt5(force_reconnect=True):
                return get_open_positions(symbol, magic, False)  # Рекурсивный вызов без повторной попытки
        
        return []

def close_position(ticket, deviation=10, retry_on_error=True):
    """
    Закрытие позиции по тикету с улучшенной обработкой ошибок
    
    Параметры:
    ticket (int): Тикет позиции
    deviation (int): Допустимое отклонение цены в пунктах
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    bool: True в случае успешного закрытия, иначе False
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error(f"Нет подключения к MT5, не удалось закрыть позицию {ticket}")
        return False
    
    try:
        # Получаем позицию по тикету
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Позиция с тикетом {ticket} не найдена")
            return False
        
        position = position[0]
        
        # Формируем запрос на закрытие
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": deviation,
            "magic": position.magic,
            "comment": f"Close position #{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Отправляем запрос
        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            logger.error(f"Не удалось отправить запрос на закрытие позиции {ticket} (код: {error[0]}): {error[1]}")
            
            # При ошибке пробуем переподключиться и повторить запрос
            if retry_on_error:
                logger.info("Пробуем переподключиться и повторить запрос")
                if connect_mt5(force_reconnect=True):
                    return close_position(ticket, deviation, False)  # Рекурсивный вызов без повторной попытки
            
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка закрытия позиции {ticket}: {result.retcode} - {result.comment}")
            
            # При определенных ошибках пробуем еще раз с другими параметрами
            if retry_on_error and result.retcode in [10004, 10018, 10019, 10025, 10026]:
                logger.info(f"Повторная попытка с другими параметрами (код ошибки: {result.retcode})")
                
                # Изменяем параметры заполнения ордера
                if 'type_filling' in request:
                    if request['type_filling'] == mt5.ORDER_FILLING_IOC:
                        request['type_filling'] = mt5.ORDER_FILLING_FOK
                    elif request['type_filling'] == mt5.ORDER_FILLING_FOK:
                        request['type_filling'] = mt5.ORDER_FILLING_RETURN
                    else:
                        request['type_filling'] = mt5.ORDER_FILLING_IOC
                
                # Пробуем отправить запрос с новыми параметрами
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Позиция {ticket} успешно закрыта со второй попытки")
                    return True
                else:
                    # Если вторая попытка не удалась, пробуем закрыть с рыночной ценой
                    logger.info("Попытка закрытия по рыночной цене")
                    return close_position_market(ticket, False)
            
            return False
        
        logger.info(f"Позиция {ticket} успешно закрыта")
        return True
        
    except Exception as e:
        logger.error(f"Исключение при закрытии позиции {ticket}: {str(e)}")
        
        # При ошибке пробуем переподключиться и повторить запрос
        if retry_on_error:
            logger.info("Пробуем переподключиться и повторить запрос")
            if connect_mt5(force_reconnect=True):
                return close_position(ticket, deviation, False)  # Рекурсивный вызов без повторной попытки
        
        return False

def close_position_market(ticket, retry_on_error=True):
    """
    Закрытие позиции по рыночной цене (альтернативный метод)
    
    Параметры:
    ticket (int): Тикет позиции
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    bool: True в случае успешного закрытия, иначе False
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error(f"Нет подключения к MT5, не удалось закрыть позицию {ticket}")
        return False
    
    try:
        # Получаем позицию по тикету
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Позиция с тикетом {ticket} не найдена")
            return False
        
        position = position[0]
        
        # Получаем информацию о символе
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            logger.error(f"Не удалось получить информацию о символе {position.symbol}")
            return False
        
        # Формируем запрос на закрытие по рыночной цене
        request = {
            "action": mt5.TRADE_ACTION_CLOSE_BY,
            "position": ticket,
            "symbol": position.symbol,
            "magic": position.magic,
            "comment": f"Close market #{ticket}",
        }
        
        # Отправляем запрос
        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            logger.error(f"Не удалось отправить запрос на закрытие позиции {ticket} (код: {error[0]}): {error[1]}")
            
            if retry_on_error:
                logger.info("Пробуем использовать стандартный метод закрытия")
                return close_position(ticket, 20, False)  # Пробуем использовать стандартный метод
            
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка закрытия позиции {ticket} по рынку: {result.retcode} - {result.comment}")
            
            if retry_on_error:
                logger.info("Пробуем использовать стандартный метод закрытия")
                return close_position(ticket, 20, False)  # Пробуем использовать стандартный метод
            
            return False
        
        logger.info(f"Позиция {ticket} успешно закрыта по рыночной цене")
        return True
        
    except Exception as e:
        logger.error(f"Исключение при закрытии позиции {ticket} по рынку: {str(e)}")
        
        if retry_on_error:
            logger.info("Пробуем использовать стандартный метод закрытия")
            return close_position(ticket, 20, False)  # Пробуем использовать стандартный метод
        
        return False

def modify_position(ticket, sl=None, tp=None, retry_on_error=True):
    """
    Изменение стоп-лосса и тейк-профита для открытой позиции
    
    Параметры:
    ticket (int): Тикет позиции
    sl (float, optional): Новый уровень стоп-лосса
    tp (float, optional): Новый уровень тейк-профита
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    bool: True в случае успешного изменения, иначе False
    """
    # Если не указаны ни SL, ни TP, нечего менять
    if sl is None and tp is None:
        logger.warning(f"Не указаны новые значения SL или TP для позиции {ticket}")
        return False
    
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error(f"Нет подключения к MT5, не удалось изменить позицию {ticket}")
        return False
    
    try:
        # Получаем позицию по тикету
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Позиция с тикетом {ticket} не найдена")
            return False
        
        position = position[0]
        
        # Используем текущие значения, если новые не указаны
        if sl is None:
            sl = position.sl
        if tp is None:
            tp = position.tp
        
        # Формируем запрос на изменение
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": sl,
            "tp": tp,
        }
        
        # Отправляем запрос
        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            logger.error(f"Не удалось отправить запрос на изменение позиции {ticket} (код: {error[0]}): {error[1]}")
            
            # При ошибке пробуем переподключиться и повторить запрос
            if retry_on_error:
                logger.info("Пробуем переподключиться и повторить запрос")
                if connect_mt5(force_reconnect=True):
                    return modify_position(ticket, sl, tp, False)  # Рекурсивный вызов без повторной попытки
            
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка изменения позиции {ticket}: {result.retcode} - {result.comment}")
            
            # При определенных ошибках пробуем еще раз
            if retry_on_error and result.retcode in [10004, 10018, 10019, 10025, 10026]:
                logger.info(f"Повторная попытка (код ошибки: {result.retcode})")
                
                # Ждем немного и пробуем еще раз
                time.sleep(1)
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Позиция {ticket} успешно изменена со второй попытки: SL = {sl}, TP = {tp}")
                    return True
            
            return False
        
        logger.info(f"Позиция {ticket} успешно изменена: SL = {sl}, TP = {tp}")
        return True
        
    except Exception as e:
        logger.error(f"Исключение при изменении позиции {ticket}: {str(e)}")
        
        # При ошибке пробуем переподключиться и повторить запрос
        if retry_on_error:
            logger.info("Пробуем переподключиться и повторить запрос")
            if connect_mt5(force_reconnect=True):
                return modify_position(ticket, sl, tp, False)  # Рекурсивный вызов без повторной попытки
        
        return False

def open_order(symbol, order_type, volume, price=None, sl=None, tp=None, deviation=10, magic=123456, comment=None, retry_on_error=True):
    """
    Открытие нового торгового ордера
    
    Параметры:
    symbol (str): Торговый символ
    order_type (str): Тип ордера ('buy', 'sell', 'buy_limit', 'sell_limit', 'buy_stop', 'sell_stop')
    volume (float): Объем в лотах
    price (float, optional): Цена исполнения (для лимитных и стоп-ордеров)
    sl (float, optional): Уровень стоп-лосса
    tp (float, optional): Уровень тейк-профита
    deviation (int): Допустимое отклонение цены в пунктах
    magic (int): Идентификатор эксперта
    comment (str, optional): Комментарий к ордеру
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    dict: Информация о созданном ордере или None в случае ошибки
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error(f"Нет подключения к MT5, не удалось открыть ордер {order_type} {symbol}")
        return None
    
    try:
        # Получаем информацию о символе
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Не удалось получить информацию о символе {symbol}")
            return None
        
        # Проверяем, что символ выбран в обзоре рынка
        if not symbol_info.visible:
            logger.warning(f"Символ {symbol} не виден в обзоре рынка, пробуем добавить его")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Не удалось добавить символ {symbol}")
                return None
        
        # Определяем тип ордера и текущие цены
        mt5_order_type = None
        if order_type == "buy":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            current_price = mt5.symbol_info_tick(symbol).ask
        elif order_type == "sell":
            mt5_order_type = mt5.ORDER_TYPE_SELL
            current_price = mt5.symbol_info_tick(symbol).bid
        elif order_type == "buy_limit":
            mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            current_price = mt5.symbol_info_tick(symbol).ask
        elif order_type == "sell_limit":
            mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            current_price = mt5.symbol_info_tick(symbol).bid
        elif order_type == "buy_stop":
            mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            current_price = mt5.symbol_info_tick(symbol).ask
        elif order_type == "sell_stop":
            mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            current_price = mt5.symbol_info_tick(symbol).bid
        else:
            logger.error(f"Неизвестный тип ордера: {order_type}")
            return None
        
        # Используем текущую цену, если цена не указана
        if price is None:
            price = current_price
        
        # Проверяем минимальный объем
        min_volume = symbol_info.volume_min
        if volume < min_volume:
            logger.warning(f"Объем {volume} меньше минимального {min_volume}, устанавливаем минимальный")
            volume = min_volume
        
        # Округляем объем согласно шагу
        volume_step = symbol_info.volume_step
        volume = round(volume / volume_step) * volume_step
        
        # Формируем запрос
        request = {
            "action": mt5.TRADE_ACTION_DEAL,  # Рыночный ордер
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": comment if comment else f"Order {order_type} {volume} lots",
            "type_time": mt5.ORDER_TIME_GTC,  # Ордер действителен до отмены
            "type_filling": mt5.ORDER_FILLING_IOC,  # Исполнить сразу или отменить
        }
        
        # Для отложенных ордеров используем TRADE_ACTION_PENDING
        if order_type in ["buy_limit", "sell_limit", "buy_stop", "sell_stop"]:
            request["action"] = mt5.TRADE_ACTION_PENDING
        
        # Добавляем SL и TP, если они указаны
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Отправляем запрос
        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            logger.error(f"Не удалось отправить запрос на открытие ордера (код: {error[0]}): {error[1]}")
            
            # При ошибке пробуем переподключиться и повторить запрос
            if retry_on_error:
                logger.info("Пробуем переподключиться и повторить запрос")
                if connect_mt5(force_reconnect=True):
                    return open_order(symbol, order_type, volume, price, sl, tp, deviation, magic, comment, False)
            
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка открытия ордера: {result.retcode} - {result.comment}")
            
            # При определенных ошибках пробуем с другими параметрами заполнения
            if retry_on_error and result.retcode in [10004, 10018, 10019, 10025, 10026]:
                logger.info(f"Повторная попытка с другими параметрами (код ошибки: {result.retcode})")
                
                # Изменяем параметры заполнения ордера
                if request['type_filling'] == mt5.ORDER_FILLING_IOC:
                    request['type_filling'] = mt5.ORDER_FILLING_FOK
                elif request['type_filling'] == mt5.ORDER_FILLING_FOK:
                    request['type_filling'] = mt5.ORDER_FILLING_RETURN
                else:
                    request['type_filling'] = mt5.ORDER_FILLING_IOC
                
                # Пробуем отправить запрос с новыми параметрами
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Ордер успешно открыт со второй попытки: {order_type} {volume} лот(ов) {symbol}")
                else:
                    logger.error(f"Не удалось открыть ордер даже со второй попытки: {result.retcode} - {result.comment}")
                    return None
            else:
                return None
        
        # Формируем информацию о созданном ордере
        order_info = {
            "ticket": result.order,
            "volume": volume,
            "price": price,
            "symbol": symbol,
            "type": order_type,
            "magic": magic,
            "comment": comment,
            "sl": sl,
            "tp": tp,
            "time": datetime.now()
        }
        
        logger.info(f"Ордер успешно открыт: {order_type} {volume} лот(ов) {symbol} по цене {price}")
        return order_info
        
    except Exception as e:
        logger.error(f"Исключение при открытии ордера: {str(e)}")
        
        # При ошибке пробуем переподключиться и повторить запрос
        if retry_on_error:
            logger.info("Пробуем переподключиться и повторить запрос")
            if connect_mt5(force_reconnect=True):
                return open_order(symbol, order_type, volume, price, sl, tp, deviation, magic, comment, False)
        
        return None

def get_pending_orders(symbol=None, magic=None, retry_on_error=True):
    """
    Получение отложенных ордеров с фильтрацией по символу и/или magic
    
    Параметры:
    symbol (str, optional): Торговый символ для фильтрации ордеров
    magic (int, optional): Magic номер для фильтрации ордеров
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    list: Список отложенных ордеров или пустой список в случае ошибки
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error("Нет подключения к MT5, не удалось получить отложенные ордера")
        return []
    
    try:
        # Формируем фильтр для orders_get
        request = {}
        if symbol is not None:
            request['symbol'] = symbol
        if magic is not None:
            request['magic'] = magic
        
        # Получаем отложенные ордера
        orders = mt5.orders_get(**request) if request else mt5.orders_get()
        
        if orders is None:
            error = mt5.last_error()
            # Код 0 означает, что просто нет отложенных ордеров
            if error[0] == 0:
                filters = []
                if symbol:
                    filters.append(f"символа {symbol}")
                if magic:
                    filters.append(f"magic {magic}")
                filter_str = f" для {' и '.join(filters)}" if filters else ""
                
                logger.info(f"Нет отложенных ордеров{filter_str}")
                return []
            else:
                logger.error(f"Ошибка при получении отложенных ордеров (код: {error[0]}): {error[1]}")
                
                # При ошибке пробуем переподключиться и повторить запрос
                if retry_on_error:
                    logger.info("Пробуем переподключиться и повторить запрос")
                    if connect_mt5(force_reconnect=True):
                        return get_pending_orders(symbol, magic, False)
                
                return []
        
        # Преобразуем ордера в список словарей для удобства
        orders_list = []
        for order in orders:
            order_type = "unknown"
            if order.type == mt5.ORDER_TYPE_BUY_LIMIT:
                order_type = "buy_limit"
            elif order.type == mt5.ORDER_TYPE_SELL_LIMIT:
                order_type = "sell_limit"
            elif order.type == mt5.ORDER_TYPE_BUY_STOP:
                order_type = "buy_stop"
            elif order.type == mt5.ORDER_TYPE_SELL_STOP:
                order_type = "sell_stop"
            
            order_info = {
                "ticket": order.ticket,
                "symbol": order.symbol,
                "type": order_type,
                "volume": order.volume_initial,
                "price": order.price_open,
                "sl": order.sl,
                "tp": order.tp,
                "time_setup": datetime.fromtimestamp(order.time_setup),
                "time_expiration": datetime.fromtimestamp(order.time_expiration) if order.time_expiration > 0 else None,
                "magic": order.magic,
                "comment": order.comment,
                "state": order.state
            }
            
            orders_list.append(order_info)
        
        # Логируем количество отложенных ордеров
        filters = []
        if symbol:
            filters.append(f"символа {symbol}")
        if magic:
            filters.append(f"magic {magic}")
        filter_str = f" для {' и '.join(filters)}" if filters else ""
        
        logger.info(f"Найдено {len(orders_list)} отложенных ордеров{filter_str}")
        
        return orders_list
    
    except Exception as e:
        logger.error(f"Исключение при получении отложенных ордеров: {str(e)}")
        
        # При ошибке пробуем переподключиться и повторить запрос
        if retry_on_error:
            logger.info("Пробуем переподключиться и повторить запрос")
            if connect_mt5(force_reconnect=True):
                return get_pending_orders(symbol, magic, False)
        
        return []

def cancel_order(ticket, retry_on_error=True):
    """
    Отмена отложенного ордера по тикету
    
    Параметры:
    ticket (int): Тикет ордера
    retry_on_error (bool): Повторная попытка при ошибке
    
    Возвращает:
    bool: True в случае успешной отмены, иначе False
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error(f"Нет подключения к MT5, не удалось отменить ордер {ticket}")
        return False
    
    try:
        # Получаем ордер по тикету
        order = mt5.orders_get(ticket=ticket)
        if order is None or len(order) == 0:
            logger.error(f"Ордер с тикетом {ticket} не найден")
            return False
        
        # Формируем запрос на отмену
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
            "comment": f"Cancel order #{ticket}"
        }
        
        # Отправляем запрос
        result = mt5.order_send(request)
        if result is None:
            error = mt5.last_error()
            logger.error(f"Не удалось отправить запрос на отмену ордера {ticket} (код: {error[0]}): {error[1]}")
            
            # При ошибке пробуем переподключиться и повторить запрос
            if retry_on_error:
                logger.info("Пробуем переподключиться и повторить запрос")
                if connect_mt5(force_reconnect=True):
                    return cancel_order(ticket, False)
            
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ошибка отмены ордера {ticket}: {result.retcode} - {result.comment}")
            
            # При определенных ошибках пробуем еще раз
            if retry_on_error and result.retcode in [10004, 10018, 10019]:
                logger.info(f"Повторная попытка (код ошибки: {result.retcode})")
                
                # Ждем немного и пробуем еще раз
                time.sleep(1)
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Ордер {ticket} успешно отменен со второй попытки")
                    return True
            
            return False
        
        logger.info(f"Ордер {ticket} успешно отменен")
        return True
        
    except Exception as e:
        logger.error(f"Исключение при отмене ордера {ticket}: {str(e)}")
        
        # При ошибке пробуем переподключиться и повторить запрос
        if retry_on_error:
            logger.info("Пробуем переподключиться и повторить запрос")
            if connect_mt5(force_reconnect=True):
                return cancel_order(ticket, False)
        
        return False

def get_symbol_info(symbol):
    """
    Получение детальной информации о торговом символе
    
    Параметры:
    symbol (str): Торговый символ
    
    Возвращает:
    dict: Словарь с информацией о символе или None в случае ошибки
    """
    # Проверяем подключение
    if not reconnect_if_needed():
        logger.error(f"Нет подключения к MT5, не удалось получить информацию о символе {symbol}")
        return None
    
    try:
        # Получаем информацию о символе
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Не удалось получить информацию о символе {symbol}")
            return None
        
        # Проверяем, что символ выбран в обзоре рынка
        if not symbol_info.visible:
            logger.warning(f"Символ {symbol} не виден в обзоре рынка, пробуем добавить его")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Не удалось добавить символ {symbol}")
                return None
        
        # Собираем основную информацию
        info = {
            "name": symbol_info.name,
            "currency_base": symbol_info.currency_base,
            "currency_profit": symbol_info.currency_profit,
            "description": symbol_info.description,
            "digits": symbol_info.digits,
            "point": symbol_info.point,
            "volume_min": symbol_info.volume_min,
            "volume_max": symbol_info.volume_max,
            "volume_step": symbol_info.volume_step,
            "trade_contract_size": symbol_info.trade_contract_size,
            "trade_tick_value": symbol_info.trade_tick_value,
            "trade_tick_size": symbol_info.trade_tick_size,
            "session_open": None,
            "session_close": None,
            "spread": symbol_info.spread,
            "swap_long": symbol_info.swap_long,
            "swap_short": symbol_info.swap_short
        }
        
        # Получаем текущие цены
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            info["bid"] = tick.bid
            info["ask"] = tick.ask
            info["spread_current"] = tick.ask - tick.bid
            info["time"] = datetime.fromtimestamp(tick.time)
        
        # Получаем расписание торговых сессий
        try:
            sessions = mt5.symbol_info_session(symbol, mt5.SYMBOL_SESSION_TRADE)
            if sessions is not None and len(sessions) > 0:
                info["sessions"] = []
                for session in sessions:
                    start_time = datetime.fromtimestamp(session[0])
                    end_time = datetime.fromtimestamp(session[1])
                    info["sessions"].append({
                        "start": start_time.strftime("%H:%M"),
                        "end": end_time.strftime("%H:%M"),
                        "day": start_time.strftime("%A")
                    })
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о сессиях для {symbol}: {str(e)}")
        
        logger.info(f"Получена информация о символе {symbol}")
        return info
        
    except Exception as e:
        logger.error(f"Исключение при получении информации о символе {symbol}: {str(e)}")
        return None

def disconnect_mt5():
    """Отключение от MetaTrader 5"""
    with _mt5_lock:
        if mt5.terminal_info() is not None:
            mt5.shutdown()
            _connection_state["connected"] = False
            logger.info("Отключение от MT5")
        else:
            logger.info("MT5 уже отключен")

# Тест подключения
if __name__ == "__main__":
    # Настраиваем логирование для консоли при запуске скрипта напрямую
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    if connect_mt5():
        print("--- Информация о счете ---")
        account_info = get_account_info()
        if account_info:
            print(f"Логин: {account_info['login']}")
            print(f"Сервер: {account_info['server']}")
            print(f"Баланс: {account_info['balance']} {account_info['currency']}")
            print(f"Средства: {account_info['equity']} {account_info['currency']}")
            print(f"Свободная маржа: {account_info['margin_free']} {account_info['currency']}")
            print(f"Уровень маржи: {account_info['margin_level']}%")
            print(f"Кредитное плечо: 1:{account_info['leverage']}")
        
        print("\n--- Открытые позиции ---")
        positions = get_open_positions()
        if positions:
            print(f"Всего открытых позиций: {len(positions)}")
            for pos in positions:
                print(f"Позиция {pos['ticket']}: {pos['symbol']} {pos['type']} {pos['volume']} лот(ов), прибыль: {pos['profit']}")
        else:
            print("Нет открытых позиций")
        
        print("\n--- Отложенные ордера ---")
        orders = get_pending_orders()
        if orders:
            print(f"Всего отложенных ордеров: {len(orders)}")
            for order in orders:
                print(f"Ордер {order['ticket']}: {order['symbol']} {order['type']} {order['volume']} лот(ов) по цене {order['price']}")
        else:
            print("Нет отложенных ордеров")
        
        print("\n--- Информация о символе ---")
        symbol_info = get_symbol_info(SYMBOL)
        if symbol_info:
            print(f"Символ: {symbol_info['name']}")
            print(f"Описание: {symbol_info['description']}")
            print(f"Базовая валюта: {symbol_info['currency_base']}")
            print(f"Валюта прибыли: {symbol_info['currency_profit']}")
            print(f"Минимальный объем: {symbol_info['volume_min']} лот(ов)")
            print(f"Шаг объема: {symbol_info['volume_step']} лот(ов)")
            print(f"Текущие цены: Bid = {symbol_info.get('bid')}, Ask = {symbol_info.get('ask')}")
            print(f"Спред: {symbol_info['spread']} пунктов")
            print(f"Своп (длинная/короткая позиция): {symbol_info['swap_long']}/{symbol_info['swap_short']}")
        
        disconnect_mt5()
    else:
        print("Не удалось подключиться к MT5")