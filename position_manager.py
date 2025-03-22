import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from mt5_connector import (
    get_open_positions, modify_position, close_position,
    get_symbol_info, reconnect_if_needed
)
from config import SYMBOL

# Настройка логгера
logger = logging.getLogger(__name__)

class PositionManager:
    """
    Класс для управления открытыми позициями, включая трейлинг-стопы,
    частичное закрытие и другие стратегии управления позициями.
    """
    
    def __init__(self, symbol=SYMBOL, trailing_activation=0.5, breakeven_activation=0.3,
                 trailing_step=0.1, partial_close_pct=0.5, use_auto_close=True):
        """
        Инициализация менеджера позиций
        
        Параметры:
        symbol (str): Торговый символ по умолчанию
        trailing_activation (float): Активация трейлинга когда прибыль достигнет доли стоп-лосса (0.5 = 50%)
        breakeven_activation (float): Активация безубытка когда прибыль достигнет доли стоп-лосса (0.3 = 30%)
        trailing_step (float): Шаг трейлинга как доля от диапазона стоп-лосс -> тейк-профит (0.1 = 10%)
        partial_close_pct (float): Процент для частичного закрытия (0.5 = 50%)
        use_auto_close (bool): Использовать ли автоматическое закрытие при определенных условиях
        """
        self.symbol = symbol
        self.trailing_activation = trailing_activation
        self.breakeven_activation = breakeven_activation
        self.trailing_step = trailing_step
        self.partial_close_pct = partial_close_pct
        self.use_auto_close = use_auto_close
        
        # Кэш позиций для отслеживания изменений
        self.position_cache = {}
        # Состояние трейлинга для каждой позиции
        self.trailing_state = {}
        # Информация о частичном закрытии позиций
        self.partial_close_info = {}
        
        logger.info(f"Инициализирован менеджер позиций (трейлинг: {trailing_activation*100:.0f}%, "
                  f"безубыток: {breakeven_activation*100:.0f}%, шаг: {trailing_step*100:.0f}%)")
    
    def process_open_positions(self):
        """
        Обработка всех открытых позиций, включая применение трейлинга и других стратегий
        
        Возвращает:
        dict: Информация о выполненных действиях
        """
        if not reconnect_if_needed():
            logger.error("Нет подключения к MT5, невозможно обработать позиции")
            return {"error": "Нет подключения к MT5"}
        
        # Получаем текущие открытые позиции
        positions = get_open_positions(symbol=self.symbol)
        
        if not positions:
            return {"message": "Нет открытых позиций"}
        
        actions_performed = {
            "trailing_stop_modified": [],
            "moved_to_breakeven": [],
            "partially_closed": [],
            "auto_closed": []
        }
        
        # Обрабатываем каждую открытую позицию
        for position in positions:
            ticket = position["ticket"]
            symbol = position["symbol"]
            position_type = position["type"]  # "buy" или "sell"
            open_price = position["open_price"]
            current_price = position["current_price"]
            sl = position["sl"]
            tp = position["tp"]
            position_profit = position["profit"]
            
            # Получаем информацию о символе
            symbol_info = get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"Невозможно получить информацию о символе {symbol}, пропускаем")
                continue
            
            # Рассчитываем текущее расстояние до стопа в пипсах
            if position_type == "buy":
                sl_distance = (open_price - sl) / 0.0001 if sl > 0 else 0
                tp_distance = (tp - open_price) / 0.0001 if tp > 0 else 0
                current_profit_pips = (current_price - open_price) / 0.0001
            else:  # sell
                sl_distance = (sl - open_price) / 0.0001 if sl > 0 else 0
                tp_distance = (open_price - tp) / 0.0001 if tp > 0 else 0
                current_profit_pips = (open_price - current_price) / 0.0001
            
            # Если стоп-лосс не установлен, невозможно применить трейлинг
            if sl_distance <= 0:
                logger.info(f"Позиция {ticket}: отсутствует стоп-лосс, трейлинг невозможен")
                continue
            
            # Проверяем, достигла ли позиция минимальной прибыли для трейлинг-стопа
            if current_profit_pips >= sl_distance * self.trailing_activation:
                # Рассчитываем смещение трейлинг-стопа
                # Процент перемещения от текущей прибыли к точке входа
                trailing_offset = current_profit_pips * self.trailing_step
                
                # Определяем новый уровень стоп-лосса
                if position_type == "buy":
                    new_sl = max(sl, current_price - trailing_offset * 0.0001)
                    # Дополнительная проверка: новый стоп должен быть ниже текущей цены
                    if new_sl >= current_price:
                        new_sl = current_price - 0.0001  # Отступ в 1 пипс от текущей цены
                else:  # sell
                    new_sl = min(sl, current_price + trailing_offset * 0.0001)
                    # Дополнительная проверка: новый стоп должен быть выше текущей цены
                    if new_sl <= current_price:
                        new_sl = current_price + 0.0001  # Отступ в 1 пипс от текущей цены
                
                # Проверяем, изменился ли стоп-лосс значительно
                if abs(new_sl - sl) / 0.0001 >= 5:  # Минимум 5 пипсов для модификации
                    # Модифицируем позицию
                    if modify_position(ticket, sl=new_sl, tp=tp):
                        logger.info(f"Позиция {ticket}: трейлинг-стоп применен, новый SL: {new_sl:.5f}, "
                                  f"прибыль: {position_profit:.2f}, в пипсах: {current_profit_pips:.1f}")
                        actions_performed["trailing_stop_modified"].append({
                            "ticket": ticket,
                            "symbol": symbol,
                            "old_sl": sl,
                            "new_sl": new_sl,
                            "profit_pips": current_profit_pips
                        })
                        
                        # Обновляем кэш состояния трейлинга
                        self.trailing_state[ticket] = {
                            "activated": True,
                            "last_modified": datetime.now(),
                            "current_sl": new_sl
                        }
                
            # Проверяем, нужно ли перенести в безубыток
            elif current_profit_pips >= sl_distance * self.breakeven_activation:
                # Если стоп еще не в безубытке
                if (position_type == "buy" and sl < open_price) or (position_type == "sell" and sl > open_price):
                    # Определяем новый стоп на безубытке с небольшим запасом в 1 пипс
                    new_sl = open_price + 0.0001 if position_type == "buy" else open_price - 0.0001
                    
                    # Модифицируем позицию
                    if modify_position(ticket, sl=new_sl, tp=tp):
                        logger.info(f"Позиция {ticket}: перенос в безубыток, новый SL: {new_sl:.5f}, "
                                  f"прибыль: {position_profit:.2f}, в пипсах: {current_profit_pips:.1f}")
                        actions_performed["moved_to_breakeven"].append({
                            "ticket": ticket,
                            "symbol": symbol,
                            "old_sl": sl,
                            "new_sl": new_sl,
                            "profit_pips": current_profit_pips
                        })
            
            # Проверяем на возможность частичного закрытия
            # Выполняем частичное закрытие, если прибыль достигла 75% от уровня тейк-профита
            # и позиция еще не была частично закрыта
            if (tp_distance > 0 and current_profit_pips >= 0.75 * tp_distance and 
                ticket not in self.partial_close_info):
                
                # Расчет объема для частичного закрытия
                partial_volume = position["volume"] * self.partial_close_pct
                
                # Частичное закрытие позиции (предполагаем, что это реализовано в MT5-коннекторе)
                try:
                    # Здесь должна быть функция для частичного закрытия
                    # Пример: close_partial_position(ticket, partial_volume)
                    # Пока просто логируем информацию
                    logger.info(f"Позиция {ticket}: необходимо частичное закрытие {partial_volume} лот (75% TP)")
                    
                    # Отмечаем, что позиция была частично закрыта
                    self.partial_close_info[ticket] = {
                        "time": datetime.now(),
                        "volume_closed": partial_volume,
                        "price": current_price,
                        "profit": position_profit * self.partial_close_pct
                    }
                    
                    actions_performed["partially_closed"].append({
                        "ticket": ticket,
                        "symbol": symbol,
                        "volume_closed": partial_volume,
                        "price": current_price,
                        "profit": position_profit * self.partial_close_pct
                    })
                except Exception as e:
                    logger.error(f"Ошибка при частичном закрытии позиции {ticket}: {e}")
            
            # Автоматическое закрытие в некоторых случаях
            if self.use_auto_close:
                # Время удержания позиции (в часах)
                holding_time = None
                if "open_time" in position:
                    holding_time = (datetime.now() - position["open_time"]).total_seconds() / 3600
                
                # Закрытие при достижении определенной прибыли и длительном удержании
                if (holding_time and holding_time > 48 and current_profit_pips > 0):
                    # Закрываем позицию, если она в прибыли и мы держим её более 48 часов
                    if close_position(ticket):
                        logger.info(f"Позиция {ticket} автоматически закрыта: время удержания {holding_time:.1f} ч, "
                                  f"прибыль: {position_profit:.2f}")
                        actions_performed["auto_closed"].append({
                            "ticket": ticket,
                            "symbol": symbol,
                            "reason": "long_hold_in_profit",
                            "holding_time": holding_time,
                            "profit": position_profit
                        })
                
                # Закрытие при развороте рынка (анализ на основе технических индикаторов)
                # Здесь может быть реализована логика для определения разворота рынка
                # Пример: определение дивергенции, пробоя ключевых уровней и т.д.
        
        # Обновляем кэш позиций для следующего вызова
        self.update_position_cache(positions)
        
        # Считаем количество выполненных действий
        total_actions = sum(len(actions) for actions in actions_performed.values())
        
        if total_actions > 0:
            return {
                "message": f"Выполнено {total_actions} действий с позициями",
                "actions": actions_performed
            }
        else:
            return {"message": "Нет изменений в позициях"}
    
    def update_position_cache(self, positions):
        """
        Обновление кэша позиций для отслеживания изменений
        
        Параметры:
        positions (list): Список текущих открытых позиций
        """
        # Создаем новый кэш
        new_cache = {}
        
        # Заполняем новый кэш
        for position in positions:
            ticket = position["ticket"]
            new_cache[ticket] = position
        
        # Обновляем кэш класса
        self.position_cache = new_cache
    
    def clear_state(self):
        """Очистка внутреннего состояния и кэша"""
        self.position_cache = {}
        self.trailing_state = {}
        self.partial_close_info = {}
        logger.info("Состояние менеджера позиций сброшено")

# Глобальный экземпляр менеджера позиций
_position_manager = None

def get_position_manager(symbol=SYMBOL, init_if_none=True, **kwargs):
    """
    Получение глобального экземпляра менеджера позиций
    
    Параметры:
    symbol (str): Торговый символ
    init_if_none (bool): Инициализировать экземпляр, если он не существует
    **kwargs: Дополнительные параметры для PositionManager
    
    Возвращает:
    PositionManager: Экземпляр менеджера позиций
    """
    global _position_manager
    
    if _position_manager is None and init_if_none:
        _position_manager = PositionManager(symbol=symbol, **kwargs)
    
    return _position_manager

def reset_position_manager():
    """Сброс глобального экземпляра менеджера позиций"""
    global _position_manager
    if _position_manager is not None:
        _position_manager.clear_state()
    _position_manager = None
    logger.info("Глобальный экземпляр менеджера позиций сброшен")

def process_positions(symbol=SYMBOL, update_interval=None):
    """
    Удобная функция для обработки открытых позиций с возможностью настройки
    
    Параметры:
    symbol (str): Торговый символ
    update_interval (int, optional): Интервал обновления трейлинг-стопов
                                   (если None, используется интервал по умолчанию)
    
    Возвращает:
    dict: Результат обработки позиций
    """
    # Статический счетчик для отслеживания последнего времени обновления
    if not hasattr(process_positions, "last_update_time"):
        process_positions.last_update_time = 0
    
    # Интервал обновления по умолчанию (5 минут)
    default_interval = 300
    interval = update_interval if update_interval is not None else default_interval
    
    # Проверяем, прошло ли достаточно времени с последнего обновления
    current_time = time.time()
    if current_time - process_positions.last_update_time < interval:
        return {"message": f"Слишком рано для обновления. Следующее обновление через {interval - (current_time - process_positions.last_update_time):.0f} сек."}
    
    # Получаем менеджер позиций
    manager = get_position_manager(symbol)
    
    # Обрабатываем позиции
    result = manager.process_open_positions()
    
    # Обновляем время последнего обновления
    process_positions.last_update_time = current_time
    
    return result