import pytz
from datetime import datetime

# === Настройки подключения к MT5 ===
MT5_LOGIN = 10005844482  # замените на свой логин
MT5_PASSWORD = "OsWcA@7m"  # замените на свой пароль
MT5_SERVER = "MetaQuotes-Demo"  # замените на сервер своего брокера
MT5_PATH = "D:/MetaTrader 5/terminal64.exe"  # Путь к терминалу MT5

# === Торговые настройки ===
SYMBOL = "EURUSD"  # Валютная пара
RISK_PER_TRADE = 0.005  # Риск на сделку (0.5% от депозита)
MAX_SPREAD = 1.5  # Максимальный спред для входа в сделку (в пипсах)
MIN_STOPLOSS_PIPS = 30  # Минимальный стоп-лосс в пипсах (увеличен для большей надежности)
MAX_POSITIONS = 3  # Максимальное количество одновременно открытых позиций
MAX_DAILY_RISK = 0.02  # Максимальный дневной риск (2% от депозита)

# === Временные рамки ===
TRADING_START_HOUR = 9  # Начало торговли (UTC)
TRADING_END_HOUR = 18  # Конец торговли (UTC)
TIMEZONE = pytz.timezone("Etc/UTC")

# === Настройки стратегии (Smart Money Concepts) ===
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]  # Таймфреймы для анализа
FVG_CONFIRMATION = True  # Использовать FVG (Fair Value Gap) для подтверждения входа
SKIP_NIGHT_TRADES = True  # Пропускать ночные сделки

# Настройки стратегии для различных сетапов
STRATEGY_SETTINGS = {
    "OrderBlock": {
        "min_stop_loss_pips": 25,
        "risk_reward_ratio": 2.5,  # Соотношение риск/доходность
        "confirmation_needed": True  # Требуется ли подтверждение на младшем ТФ
    },
    "BreakerBlock": {
        "min_stop_loss_pips": 40,
        "risk_reward_ratio": 3.5,
        "confirmation_needed": False
    },
    "FVG": {
        "min_stop_loss_pips": 30,
        "risk_reward_ratio": 3.0,
        "confirmation_needed": True
    },
    "LiquidityGrab": {
        "min_stop_loss_pips": 35,
        "risk_reward_ratio": 2.8,
        "confirmation_needed": True
    },
    "EqualHighLow": {
        "min_stop_loss_pips": 25,
        "risk_reward_ratio": 2.0,
        "confirmation_needed": False
    }
}

# === Настройки бэктеста ===
BACKTEST_START = datetime(2024, 10, 1)  # Дата начала бэктеста
BACKTEST_END = datetime(2025, 3, 1)  # Дата окончания бэктеста
CANDLES_FOR_EACH_TF = {  # Количество свечей для каждого таймфрейма
    "M1": 1000,
    "M5": 1000,
    "M15": 500,  # Увеличено для более качественного анализа
    "M30": 300,  # Увеличено для более качественного анализа
    "H1": 200,   # Увеличено для более качественного анализа
    "H4": 100,   # Увеличено для более качественного анализа
    "D1": 50,    # Увеличено для более качественного анализа
}

# === Настройки визуализации ===
VISUALIZATION_THEME = "white"  # white или dark
VISUALIZATION_AUTO_OPEN = True  # Автоматически открывать браузер
VISUALIZATION_TIMEFRAMES = ["M5", "M15", "H1"]  # Таймфреймы для графиков цены
VISUALIZATION_WITH_PRICES = True  # Включать ли графики цен 
VISUALIZATION_DPI = 150  # Разрешение графиков
VISUALIZATION_OUTPUT_DIR = "backtest_results"  # Директория для результатов бэктеста

# === Логирование ===
LOG_FILE = "bot_logs.txt"  # Файл для логов
LOG_LEVEL = "INFO"  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
SAVE_CHARTS = True  # Сохранять ли графики с сигналами

# === Журнал сделок ===
TRADE_JOURNAL_ENABLED = True  # Включить расширенный журнал сделок
TRADE_JOURNAL_AUTO_REPORT = True  # Автоматически создавать отчеты после сессии
TRADE_JOURNAL_DAYS_SUMMARY = 7  # Количество дней для краткой статистики

# === Telegram-уведомления ===
TELEGRAM_ENABLED = True  # Включить уведомления в Telegram
TELEGRAM_BOT_TOKEN = "8010458678:AAHGU6oqW8DbAQlhvqz-bRGr4hk023RUSi8"  # Токен вашего Telegram-бота
TELEGRAM_CHAT_ID = "739207956"  # ID чата для отправки уведомлений

# === Интервал проверки сделок ===
CHECK_INTERVAL = 300  # Время между проверками сигналов (в секундах, увеличено до 5 минут)

# === Режим работы (backtest / live) ===
MODE = "backtest"  # Установи "backtest" для бэктеста, "live" для реальной торговли

# === Начальный баланс ===
INITIAL_BALANCE = 10000  # Начальный депозит для расчета прибыли

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