import pytz
from datetime import datetime
import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

#Ошибка связана с кодировкой файла .env. Давайте исправим проблему в файле config.py:
import logging

# Получаем абсолютный путь к текущей директории
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, '.env')

# Загрузка переменных из .env файла с явным приоритетом над системными
if os.path.exists(env_path):
    # Загружаем .env файл с явным указанием override=True
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Явно установим значение MODE
    try:
        # Используем явное указание кодировки utf-8 или latin-1
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() == 'MODE':
                            # Явно установим значение MODE из .env
                            os.environ['MODE'] = value.strip()
                            print(f"MODE установлен из .env: {value.strip()}")
    except UnicodeDecodeError:
        # Если UTF-8 не работает, пробуем другую кодировку
        try:
            with open(env_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if key.strip() == 'MODE':
                                os.environ['MODE'] = value.strip()
                                print(f"MODE установлен из .env (latin-1): {value.strip()}")
        except Exception as e:
            print(f"Ошибка при чтении .env файла: {e}")
    
    print(f"Загружены настройки из {env_path}")
else:
    print(f"ВНИМАНИЕ: Файл .env не найден в {base_dir}. Используются значения по умолчанию.")



# === Настройки подключения к MT5 ===
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))  # Заменено на нейтральное значение
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")  # Пустая строка по умолчанию
MT5_SERVER = os.getenv("MT5_SERVER", "")  # Пустая строка по умолчанию
MT5_PATH = os.getenv("MT5_PATH", "")  # Пустая строка по умолчанию

# === Торговые настройки ===
SYMBOL = os.getenv("SYMBOL", "EURUSD")  # Базовое значение
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # Стандартное значение
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "1.5"))
MIN_STOPLOSS_PIPS = int(os.getenv("MIN_STOPLOSS_PIPS", "20"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "1"))
MAX_DAILY_RISK = float(os.getenv("MAX_DAILY_RISK", "0.01"))

# === Telegram-уведомления ===
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False").lower() == "true"  # Отключено по умолчанию
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")  # Пустая строка по умолчанию
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # Пустая строка по умолчанию

# === Временные рамки ===
TRADING_START_HOUR = int(os.getenv("TRADING_START_HOUR", "9"))  # Начало торговли (UTC)
TRADING_END_HOUR = int(os.getenv("TRADING_END_HOUR", "18"))  # Конец торговли (UTC)
TIMEZONE = pytz.timezone(os.getenv("TIMEZONE", "Etc/UTC"))

# === Настройки стратегии (Smart Money Concepts) ===
TIMEFRAMES = os.getenv("TIMEFRAMES", "M5,M15,H1,H4,D1").split(",")  # Таймфреймы для анализа
FVG_CONFIRMATION = os.getenv("FVG_CONFIRMATION", "True").lower() == "true"  # Использовать FVG (Fair Value Gap) для подтверждения входа
SKIP_NIGHT_TRADES = os.getenv("SKIP_NIGHT_TRADES", "True").lower() == "true"  # Пропускать ночные сделки

# Настройки стратегии для различных сетапов
STRATEGY_SETTINGS = {
    "OrderBlock": {
        "min_stop_loss_pips": int(os.getenv("OB_MIN_SL_PIPS", "25")),
        "risk_reward_ratio": float(os.getenv("OB_RR_RATIO", "2.5")),  # Соотношение риск/доходность
        "confirmation_needed": os.getenv("OB_CONFIRMATION", "True").lower() == "true"  # Требуется ли подтверждение на младшем ТФ
    },
    "BreakerBlock": {
        "min_stop_loss_pips": int(os.getenv("BB_MIN_SL_PIPS", "40")),
        "risk_reward_ratio": float(os.getenv("BB_RR_RATIO", "3.5")),
        "confirmation_needed": os.getenv("BB_CONFIRMATION", "False").lower() == "true"
    },
    "FVG": {
        "min_stop_loss_pips": int(os.getenv("FVG_MIN_SL_PIPS", "30")),
        "risk_reward_ratio": float(os.getenv("FVG_RR_RATIO", "3.0")),
        "confirmation_needed": os.getenv("FVG_CONFIRMATION", "True").lower() == "true"
    },
    "LiquidityGrab": {
        "min_stop_loss_pips": int(os.getenv("LG_MIN_SL_PIPS", "35")),
        "risk_reward_ratio": float(os.getenv("LG_RR_RATIO", "2.8")),
        "confirmation_needed": os.getenv("LG_CONFIRMATION", "True").lower() == "true"
    },
    "EqualHighLow": {
        "min_stop_loss_pips": int(os.getenv("EQ_MIN_SL_PIPS", "25")),
        "risk_reward_ratio": float(os.getenv("EQ_RR_RATIO", "2.0")),
        "confirmation_needed": os.getenv("EQ_CONFIRMATION", "False").lower() == "true"
    }
}

# === Настройки бэктеста ===
# Преобразуем строки в объект datetime
def parse_date(date_string, default):
    if not date_string:
        return default
    try:
        parts = date_string.split('-')
        if len(parts) == 3:
            year, month, day = map(int, parts)
            return datetime(year, month, day)
    except:
        return default
    return default

BACKTEST_START = parse_date(os.getenv("BACKTEST_START"), datetime(2024, 10, 1))
BACKTEST_END = parse_date(os.getenv("BACKTEST_END"), datetime(2025, 3, 1))

# Количество свечей для каждого таймфрейма
CANDLES_FOR_EACH_TF = {
    "M1": int(os.getenv("CANDLES_M1", "1000")),
    "M5": int(os.getenv("CANDLES_M5", "1000")),
    "M15": int(os.getenv("CANDLES_M15", "500")),
    "M30": int(os.getenv("CANDLES_M30", "300")),
    "H1": int(os.getenv("CANDLES_H1", "200")),
    "H4": int(os.getenv("CANDLES_H4", "100")),
    "D1": int(os.getenv("CANDLES_D1", "50")),
}

# === Настройки визуализации ===
VISUALIZATION_THEME = os.getenv("VISUALIZATION_THEME", "white")  # white или dark
VISUALIZATION_AUTO_OPEN = os.getenv("VISUALIZATION_AUTO_OPEN", "True").lower() == "true"  # Автоматически открывать браузер
VISUALIZATION_TIMEFRAMES = os.getenv("VIS_TIMEFRAMES", "M5,M15,H1").split(",")  # Таймфреймы для графиков цены
VISUALIZATION_WITH_PRICES = os.getenv("VIS_WITH_PRICES", "True").lower() == "true"  # Включать ли графики цен 
VISUALIZATION_DPI = int(os.getenv("VIS_DPI", "150"))  # Разрешение графиков
VISUALIZATION_OUTPUT_DIR = os.getenv("VIS_OUTPUT_DIR", "backtest_results")  # Директория для результатов бэктеста

# === Логирование ===
LOG_FILE = os.getenv("LOG_FILE", "bot_logs.txt")  # Файл для логов
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
SAVE_CHARTS = os.getenv("SAVE_CHARTS", "True").lower() == "true"  # Сохранять ли графики с сигналами

# === Журнал сделок ===
TRADE_JOURNAL_ENABLED = os.getenv("TRADE_JOURNAL_ENABLED", "True").lower() == "true"  # Включить расширенный журнал сделок
TRADE_JOURNAL_AUTO_REPORT = os.getenv("TRADE_JOURNAL_AUTO_REPORT", "True").lower() == "true"  # Автоматически создавать отчеты после сессии
TRADE_JOURNAL_DAYS_SUMMARY = int(os.getenv("TRADE_JOURNAL_DAYS_SUMMARY", "7"))  # Количество дней для краткой статистики

# === Настройки трейлинг-стопа и управления позициями ===
TRAILING_ACTIVATION = float(os.getenv("TRAILING_ACTIVATION", "0.5"))  # Активация трейлинга при достижении 50% от стоп-лосса
BREAKEVEN_ACTIVATION = float(os.getenv("BREAKEVEN_ACTIVATION", "0.3"))  # Активация безубытка при достижении 30% от стоп-лосса
TRAILING_STEP = float(os.getenv("TRAILING_STEP", "0.1"))  # Шаг трейлинга (10% от расстояния прибыли)
PARTIAL_CLOSE_PCT = float(os.getenv("PARTIAL_CLOSE_PCT", "0.5"))  # Процент частичного закрытия (50% от позиции)
USE_AUTO_CLOSE = os.getenv("USE_AUTO_CLOSE", "True").lower() == "true"  # Использовать автоматическое закрытие позиций по определенным правилам
MAX_TRADE_DURATION = int(os.getenv("MAX_TRADE_DURATION", "12"))  # Максимальная длительность сделки в часах до автозакрытия

# Настройки адаптивного трейлинга для разных типов трендов
TRAILING_CONFIG = {
    "strong_trend": {
        "activation": float(os.getenv("STRONG_TREND_ACTIVATION", "0.6")),  # Более поздняя активация для сильного тренда
        "step": float(os.getenv("STRONG_TREND_STEP", "0.08")),       # Более осторожный шаг для сильного тренда
    },
    "weak_trend": {
        "activation": float(os.getenv("WEAK_TREND_ACTIVATION", "0.4")),  # Более ранняя активация для слабого тренда
        "step": float(os.getenv("WEAK_TREND_STEP", "0.15")),       # Более агрессивный шаг для слабого тренда
    },
    "sideways": {
        "activation": float(os.getenv("SIDEWAYS_ACTIVATION", "0.3")),  # Ранняя активация для бокового движения
        "step": float(os.getenv("SIDEWAYS_STEP", "0.2")),        # Агрессивный шаг для бокового движения
    }
}

# === Интервал проверки сделок ===
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))  # Время между проверками сигналов (в секундах, увеличено до 5 минут)

# === Режим работы (backtest / live) ===
MODE = os.getenv("MODE", "")  # Установи "backtest" для бэктеста, "live" для реальной торговли

# === Начальный баланс ===
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10000"))  # Начальный депозит для расчета прибыли

# === Риск-менеджмент ===
RISK_MANAGER_ENABLED = os.getenv("RISK_MANAGER_ENABLED", "True").lower() == "true"  # Включить систему риск-менеджмента
RISK_POSITION_SIZING_METHOD = os.getenv("RISK_POSITION_SIZING_METHOD", "fixed_percent")  # Метод расчета размера позиции (fixed_percent, kelly, optimal_f)

# === Функция для проверки рыночных условий ===
def is_market_open():
    """Проверяет, открыт ли рынок сейчас"""
    now = datetime.now(TIMEZONE)
    current_hour = now.hour
    
    # Проверка на выходные
    if now.weekday() >= 5:  # 5 = суббота, 6 = воскресенье
        return False
    
    # Проверка торгового времени
    if current_hour >= TRADING_START_HOUR and current_hour < TRADING_END_HOUR:
        return True
    
    return False

# === Экспортируем функцию проверки времени работы рынка ===
def is_trading_allowed():
    """
    Проверяет, разрешена ли торговля в текущее время
    Учитывает настройки времени и режим работы
    """
    if MODE == "backtest":
        return True  # В режиме бэктеста всегда разрешено
    
    if not is_market_open():
        return False
    
    # Проверка на ночное время, если нужно пропускать ночные сделки
    if SKIP_NIGHT_TRADES:
        now = datetime.now(TIMEZONE)
        if now.hour < 8 or now.hour > 20:  # Пример ночного времени
            return False
    
    return True