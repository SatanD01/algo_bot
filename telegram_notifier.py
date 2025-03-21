import requests
import logging
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import os

# Настройка логирования
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID):
        """
        Инициализация нотификатора Telegram
        
        Параметры:
        bot_token (str): Токен бота Telegram
        chat_id (str): ID чата для отправки сообщений
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = False
        
        # Проверяем наличие токена и chat_id
        if bot_token and chat_id:
            self.enabled = True
        else:
            logger.warning("Telegram-уведомления отключены. Не указан токен бота или ID чата.")
    
    def send_message(self, message, disable_notification=False):
        """
        Отправляет сообщение в Telegram
        
        Параметры:
        message (str): Текст сообщения
        disable_notification (bool): Отключить звуковое уведомление
        
        Возвращает:
        bool: True в случае успеха, False в случае ошибки
        """
        if not self.enabled:
            logger.debug("Telegram-уведомления отключены.")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            params = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_notification": disable_notification
            }
            
            response = requests.post(url, data=params, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Telegram-уведомление отправлено успешно")
                return True
            else:
                logger.error(f"Ошибка при отправке Telegram-уведомления: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Исключение при отправке Telegram-уведомления: {str(e)}")
            return False
    
    def send_trade_open_notification(self, trade_info):
        """
        Отправляет уведомление об открытии сделки
        
        Параметры:
        trade_info (dict): Информация о сделке
        
        Возвращает:
        bool: True в случае успеха, False в случае ошибки
        """
        if not self.enabled:
            return False
        
        try:
            # Форматируем данные
            symbol = trade_info.get('symbol', 'Unknown')
            order_type = trade_info.get('order', 'Unknown').upper()
            entry_price = trade_info.get('entry_price', 0)
            stop_loss = trade_info.get('sl', 0)
            take_profit = trade_info.get('tp', 0)
            lot_size = trade_info.get('lot_size', 0)
            setup = trade_info.get('setup', 'Standard')
            
            # Рассчитываем риск и потенциальную прибыль в пипсах
            sl_pips = abs(entry_price - stop_loss) / 0.0001
            tp_pips = abs(entry_price - take_profit) / 0.0001
            
            # Формируем текст сообщения
            message = (
                f"🔔 <b>ОТКРЫТА НОВАЯ СДЕЛКА</b> 🔔\n\n"
                f"<b>Символ:</b> {symbol}\n"
                f"<b>Тип:</b> {order_type}\n"
                f"<b>Цена входа:</b> {entry_price:.5f}\n"
                f"<b>Стоп-лосс:</b> {stop_loss:.5f} ({sl_pips:.0f} пипсов)\n"
                f"<b>Тейк-профит:</b> {take_profit:.5f} ({tp_pips:.0f} пипсов)\n"
                f"<b>Объем:</b> {lot_size:.2f} лот\n"
                f"<b>Сетап:</b> {setup}\n"
                f"<b>Время:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Ошибка при формировании уведомления об открытии сделки: {str(e)}")
            return False
    
    def send_trade_close_notification(self, trade_info):
        """
        Отправляет уведомление о закрытии сделки
        
        Параметры:
        trade_info (dict): Информация о сделке
        
        Возвращает:
        bool: True в случае успеха, False в случае ошибки
        """
        if not self.enabled:
            return False
        
        try:
            # Форматируем данные
            symbol = trade_info.get('symbol', 'Unknown')
            order_type = trade_info.get('order', 'Unknown').upper()
            entry_price = trade_info.get('entry_price', 0)
            exit_price = trade_info.get('exit_price', 0)
            profit = trade_info.get('profit', 0)
            result = trade_info.get('result', 'Unknown')
            exit_type = trade_info.get('exit_type', 'Unknown')
            
            # Определяем эмодзи на основе результата
            emoji = "✅" if result == 'win' else "❌"
            result_text = "ВЫИГРЫШ" if result == 'win' else "ПРОИГРЫШ"
            
            # Формируем текст сообщения
            message = (
                f"{emoji} <b>СДЕЛКА ЗАКРЫТА - {result_text}</b> {emoji}\n\n"
                f"<b>Символ:</b> {symbol}\n"
                f"<b>Тип:</b> {order_type}\n"
                f"<b>Цена входа:</b> {entry_price:.5f}\n"
                f"<b>Цена выхода:</b> {exit_price:.5f}\n"
                f"<b>Прибыль:</b> {profit:.2f}\n"
                f"<b>Тип выхода:</b> {exit_type}\n"
                f"<b>Время:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Ошибка при формировании уведомления о закрытии сделки: {str(e)}")
            return False

# Единственный экземпляр нотификатора для всего приложения
notifier = None

def initialize_telegram_notifier(bot_token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID):
    """
    Инициализирует глобальный экземпляр Telegram-нотификатора
    
    Параметры:
    bot_token (str): Токен бота Telegram
    chat_id (str): ID чата для отправки сообщений
    
    Возвращает:
    TelegramNotifier: Инициализированный нотификатор
    """
    global notifier
    
    # Если токен и чат не указаны, пробуем загрузить из переменных окружения
    if not bot_token:
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    
    if not chat_id:
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    notifier = TelegramNotifier(bot_token, chat_id)
    return notifier

def get_notifier():
    """
    Возвращает глобальный экземпляр нотификатора, инициализируя его при необходимости
    
    Возвращает:
    TelegramNotifier: Экземпляр нотификатора
    """
    global notifier
    if notifier is None:
        notifier = initialize_telegram_notifier()
    return notifier

# Тестирование при запуске файла напрямую
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Тестируем отправку сообщений, если токен и чат указаны в переменных окружения
    test_notifier = get_notifier()
    
    if test_notifier.enabled:
        print("Тест отправки простого сообщения...")
        test_notifier.send_message("Тестовое сообщение от AlgoTrade бота")
        
        print("Тест отправки уведомления об открытии сделки...")
        test_trade_open = {
            'symbol': 'EURUSD',
            'order': 'buy',
            'entry_price': 1.12345,
            'sl': 1.12245,
            'tp': 1.12545,
            'lot_size': 0.1,
            'setup': 'OrderBlock'
        }
        test_notifier.send_trade_open_notification(test_trade_open)
        
        print("Тест отправки уведомления о закрытии сделки...")
        test_trade_close = {
            'symbol': 'EURUSD',
            'order': 'buy',
            'entry_price': 1.12345,
            'exit_price': 1.12545,
            'profit': 20.0,
            'result': 'win',
            'exit_type': 'take_profit'
        }
        test_notifier.send_trade_close_notification(test_trade_close)
        
        print("Тесты завершены")
    else:
        print("Telegram-нотификатор не включен. Укажите TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в переменных окружения.")