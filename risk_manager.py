import pandas as pd
import numpy as np
import logging
import time
import os
import json
from datetime import datetime, timedelta
from functools import lru_cache

# Настройка логгера
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Класс для управления рисками и капиталом, реализующий различные методы 
    управления позициями, защиты счёта и оптимизации соотношения риск/доходность.
    """
    
    def __init__(self, account_balance=10000, risk_per_trade=0.01, max_daily_risk=0.05, 
                 max_drawdown_limit=0.15, min_win_rate=0.40, position_sizing_method="fixed_percent"):
        """
        Инициализация менеджера рисков с параметрами по умолчанию
        
        Параметры:
        account_balance (float): Текущий баланс аккаунта
        risk_per_trade (float): Риск на одну сделку (доля от баланса)
        max_daily_risk (float): Максимальный дневной риск (доля от баланса)
        max_drawdown_limit (float): Лимит на максимальную просадку для остановки торговли
        min_win_rate (float): Минимальный винрейт для продолжения торговли
        position_sizing_method (str): Метод расчета размера позиции:
                                      "fixed_percent", "kelly", "optimal_f", "martingale", "anti_martingale"
        """
        # Основные параметры риск-менеджмента
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown_limit = max_drawdown_limit
        self.min_win_rate = min_win_rate
        self.position_sizing_method = position_sizing_method
        
        # Инициализация отслеживания торговых результатов
        self.daily_risk_used = 0.0
        self.daily_loss = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = account_balance
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Журнал сделок для анализа и оптимизации
        self.trade_history = []
        self.today_date = datetime.now().date()
        
        # Загрузка предыдущих данных, если доступны
        self._load_trading_stats()
        
        # Директория для сохранения данных
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Инициализирован риск-менеджер: риск на сделку {risk_per_trade*100}%, "
                   f"макс. дневной риск {max_daily_risk*100}%, метод расчета позиции: {position_sizing_method}")
    
    def _load_trading_stats(self):
        """Загрузка предыдущих торговых статистик"""
        stats_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "risk_data", "trading_stats.json")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                # Загружаем только те данные, которые относятся к текущему дню
                current_date = datetime.now().date().isoformat()
                if stats.get("date") == current_date:
                    self.daily_risk_used = stats.get("daily_risk_used", 0.0)
                    self.daily_loss = stats.get("daily_loss", 0.0)
                
                # Общая статистика обновляется независимо от даты
                self.consecutive_losses = stats.get("consecutive_losses", 0)
                self.consecutive_wins = stats.get("consecutive_wins", 0)
                self.total_trades = stats.get("total_trades", 0)
                self.winning_trades = stats.get("winning_trades", 0)
                self.losing_trades = stats.get("losing_trades", 0)
                
                # Загружаем данные о просадке
                self.peak_balance = stats.get("peak_balance", self.account_balance)
                self.current_drawdown = stats.get("current_drawdown", 0.0)
                
                # Загружаем историю сделок, если есть
                if "trade_history" in stats:
                    self.trade_history = stats["trade_history"]
                
                logger.info(f"Загружена торговая статистика: всего сделок {self.total_trades}, "
                           f"винрейт {self.get_win_rate()*100:.2f}%")
            except Exception as e:
                logger.error(f"Ошибка при загрузке торговой статистики: {str(e)}")
    
    def _save_trading_stats(self):
        """Сохранение текущих торговых статистик"""
        stats_file = os.path.join(self.data_dir, "trading_stats.json")
        
        stats = {
            "date": datetime.now().date().isoformat(),
            "daily_risk_used": self.daily_risk_used,
            "daily_loss": self.daily_loss,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown,
            "account_balance": self.account_balance,
            "trade_history": self.trade_history[-100:]  # Сохраняем только последние 100 сделок
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            logger.error(f"Ошибка при сохранении торговой статистики: {str(e)}")
    
    def reset_daily_stats(self):
        """Сброс дневной статистики при начале нового дня"""
        current_date = datetime.now().date()
        
        if self.today_date != current_date:
            self.today_date = current_date
            self.daily_risk_used = 0.0
            self.daily_loss = 0.0
            self._save_trading_stats()
            logger.info(f"Сброс дневной статистики на новую дату: {current_date}")
            return True
        
        return False
    
    def update_balance(self, new_balance):
        """
        Обновление баланса и расчет просадки
        
        Параметры:
        new_balance (float): Новый баланс аккаунта
        
        Возвращает:
        float: Текущая просадка в процентах
        """
        self.account_balance = new_balance
        
        # Обновляем пиковый баланс, если текущий баланс выше
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.current_drawdown = 0.0
        else:
            # Рассчитываем текущую просадку
            self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
        
        self._save_trading_stats()
        return self.current_drawdown * 100  # Возвращаем просадку в процентах
    
    def register_trade_result(self, profit, risk_amount, balance_after=None):
        """
        Регистрация результата сделки для обновления статистики
        
        Параметры:
        profit (float): Прибыль/убыток по сделке
        risk_amount (float): Сумма риска в сделке
        balance_after (float, optional): Баланс после сделки, если None, рассчитывается
        
        Возвращает:
        bool: True, если торговля может продолжаться, False если достигнуты лимиты
        """
        # Обновляем баланс, если он передан
        if balance_after is not None:
            self.update_balance(balance_after)
        else:
            self.update_balance(self.account_balance + profit)
        
        # Обновляем общую статистику сделок
        self.total_trades += 1
        
        # Обновляем статистику по результату
        if profit > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Обновляем дневные потери
            self.daily_loss += abs(profit)
        
        # Создаем запись о сделке
        trade_record = {
            "time": datetime.now().isoformat(),
            "profit": profit,
            "risk_amount": risk_amount,
            "balance_after": self.account_balance,
            "drawdown": self.current_drawdown
        }
        
        # Добавляем запись в историю
        self.trade_history.append(trade_record)
        
        # Сохраняем обновленную статистику
        self._save_trading_stats()
        
        # Проверяем, можно ли продолжать торговлю
        return self.can_trade()
    
    def get_win_rate(self):
        """
        Расчет текущего винрейта
        
        Возвращает:
        float: Текущий винрейт (доля)
        """
        if self.total_trades == 0:
            return 0.0
        
        return self.winning_trades / self.total_trades
    
    def get_average_win_loss_ratio(self):
        """
        Расчет соотношения среднего выигрыша к среднему проигрышу
        
        Возвращает:
        float: Соотношение среднего выигрыша к среднему проигрышу
        """
        # Выделяем выигрышные и проигрышные сделки из истории
        if len(self.trade_history) < 5:  # Нужно минимальное количество сделок для статистики
            return 1.0  # По умолчанию
        
        # Извлекаем прибыли из истории
        profits = [trade["profit"] for trade in self.trade_history]
        
        # Разделяем на выигрыши и проигрыши
        wins = [p for p in profits if p > 0]
        losses = [abs(p) for p in profits if p < 0]
        
        # Рассчитываем средние значения
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 1.0  # Избегаем деления на ноль
        
        # Возвращаем соотношение
        return avg_win / avg_loss if avg_loss != 0 else 1.0
    
    def can_trade(self):
        """
        Проверка возможности торговли на основе риск-параметров
        
        Возвращает:
        tuple: (можно_торговать, причина_запрета)
        """
        # Сбрасываем дневную статистику, если нужно
        self.reset_daily_stats()
        
        # Проверка достижения максимальной просадки
        if self.current_drawdown >= self.max_drawdown_limit:
            reason = f"Достигнут лимит просадки ({self.current_drawdown*100:.2f}% >= {self.max_drawdown_limit*100:.2f}%)"
            logger.warning(reason)
            return False, reason
        
        # Проверка на максимальный дневной риск
        if self.daily_risk_used >= self.max_daily_risk:
            reason = f"Достигнут максимальный дневной риск ({self.daily_risk_used*100:.2f}% >= {self.max_daily_risk*100:.2f}%)"
            logger.warning(reason)
            return False, reason
        
        # Проверка на винрейт (только если достаточно сделок)
        if self.total_trades >= 20 and self.get_win_rate() < self.min_win_rate:
            reason = f"Винрейт ниже минимального ({self.get_win_rate()*100:.2f}% < {self.min_win_rate*100:.2f}%)"
            logger.warning(reason)
            return False, reason
        
        # Проверка на серию последовательных убытков (защита от затяжной просадки)
        max_consecutive_losses = 6  # Настраиваемый параметр
        if self.consecutive_losses >= max_consecutive_losses:
            reason = f"Достигнут лимит последовательных убытков ({self.consecutive_losses} >= {max_consecutive_losses})"
            logger.warning(reason)
            return False, reason
        
        # Все проверки пройдены
        return True, None
    
    def calculate_position_size(self, stop_loss_pips, entry_price, symbol_info, max_risk_override=None):
        """
        Расчет оптимального размера позиции на основе выбранного метода
        
        Параметры:
        stop_loss_pips (int): Размер стоп-лосса в пипсах
        entry_price (float): Цена входа
        symbol_info (dict): Информация о символе
        max_risk_override (float, optional): Переопределение максимального риска
        
        Возвращает:
        tuple: (размер_лота, риск_в_валюте, риск_в_процентах)
        """
        # Сбрасываем дневную статистику, если нужно
        self.reset_daily_stats()
        
        # Определяем максимальный риск на сделку
        max_risk = max_risk_override if max_risk_override is not None else self.risk_per_trade
        
        # Ограничиваем риск с учетом дневного лимита
        remaining_daily_risk = self.max_daily_risk - self.daily_risk_used
        max_risk = min(max_risk, remaining_daily_risk)
        
        # Если стоп-лосс не указан или некорректен, используем минимальный разумный
        if stop_loss_pips <= 0:
            logger.warning("Некорректный стоп-лосс, используем значение по умолчанию 30 пипсов")
            stop_loss_pips = 30
        
        # Базовый расчет по методу фиксированного процента риска
        risk_amount = self.account_balance * max_risk
        
        # Корректируем метод расчета в зависимости от выбранного
        if self.position_sizing_method == "kelly":
            # Формула Келли: f* = (bp - q) / b, где
            # b - отношение размера выигрыша к размеру ставки
            # p - вероятность выигрыша
            # q - вероятность проигрыша (1-p)
            win_rate = self.get_win_rate()
            win_loss_ratio = self.get_average_win_loss_ratio()
            
            # Предотвращаем деление на ноль и обеспечиваем разумные значения
            if win_rate < 0.1:
                win_rate = 0.1  # Минимальный винрейт для расчета
            if win_loss_ratio < 0.5:
                win_loss_ratio = 0.5  # Минимальное соотношение
            
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Ограничиваем долю Келли для более консервативного подхода
            kelly_fraction = max(0.01, min(kelly_fraction, 0.25))  # Не более 25% от формулы Келли
            
            # Рассчитываем риск по формуле Келли
            risk_amount = self.account_balance * kelly_fraction
            max_risk = kelly_fraction
            
            logger.info(f"Расчет по Келли: win_rate={win_rate:.2f}, win_loss_ratio={win_loss_ratio:.2f}, "
                        f"kelly_fraction={kelly_fraction:.4f}, risk_amount=${risk_amount:.2f}")
            
        elif self.position_sizing_method == "optimal_f":
            # Optimal f (Ральф Винс): оптимизирует геометрический рост
            # Используем упрощенную версию, основанную на последних результатах
            win_rate = self.get_win_rate()
            win_loss_ratio = self.get_average_win_loss_ratio()
            
            # Аппроксимация optimal f (для точного расчета нужна оптимизация)
            optimal_f = win_rate - (1 / win_loss_ratio) * (1 - win_rate)
            optimal_f = max(0.01, min(optimal_f, 0.2))  # Ограничиваем для безопасности
            
            risk_amount = self.account_balance * optimal_f
            max_risk = optimal_f
            
            logger.info(f"Расчет по Optimal f: win_rate={win_rate:.2f}, win_loss_ratio={win_loss_ratio:.2f}, "
                        f"optimal_f={optimal_f:.4f}, risk_amount=${risk_amount:.2f}")
            
        elif self.position_sizing_method == "martingale":
            # Мартингейл: увеличиваем ставку после каждого проигрыша
            # ВНИМАНИЕ: высокорисковая стратегия!
            if self.consecutive_losses > 0:
                # Ограничиваем прогрессию до разумных пределов
                multiplier = min(2 ** self.consecutive_losses, 4)  # Максимум 4x от начального риска
                risk_amount = self.account_balance * max_risk * multiplier
                max_risk = max_risk * multiplier
                
                logger.info(f"Мартингейл: consecutive_losses={self.consecutive_losses}, "
                            f"multiplier={multiplier}, risk_amount=${risk_amount:.2f}")
            
        elif self.position_sizing_method == "anti_martingale":
            # Анти-мартингейл: увеличиваем ставку после каждого выигрыша
            if self.consecutive_wins > 0:
                # Более консервативная прогрессия
                multiplier = min(1 + (0.3 * self.consecutive_wins), 3)  # Максимум 3x от начального риска
                risk_amount = self.account_balance * max_risk * multiplier
                max_risk = max_risk * multiplier
                
                logger.info(f"Анти-мартингейл: consecutive_wins={self.consecutive_wins}, "
                            f"multiplier={multiplier}, risk_amount=${risk_amount:.2f}")
        
        # Рассчитываем стоимость 1 пипса для лота 1.0
        point = symbol_info["point"]
        pip_value = point * 10  # 1 пипс = 10 пунктов
        tick_value = symbol_info["trade_tick_value"]
        tick_size = symbol_info["trade_tick_size"]
        
        # Стоимость 1 пипса для лота 1.0
        pip_cost = (pip_value / tick_size) * tick_value
        
        # Расчет размера лота
        # risk_amount = lot_size * stop_loss_pips * pip_cost
        lot_size = risk_amount / (stop_loss_pips * pip_cost)
        
        # Ограничения на минимальный и максимальный лот
        min_lot = symbol_info["volume_min"]
        max_lot = symbol_info["volume_max"]
        step = symbol_info["volume_step"]
        
        # Округляем до ближайшего шага объема
        lot_size = round(lot_size / step) * step
        
        # Ограничиваем лот в пределах допустимых значений
        lot_size = max(min(lot_size, max_lot), min_lot)
        
        # Обновляем использованный риск
        self.daily_risk_used += max_risk
        
        # Логируем результат расчета
        logger.info(f"Расчет позиции: баланс=${self.account_balance}, метод={self.position_sizing_method}, "
                   f"риск={max_risk*100:.2f}%, SL={stop_loss_pips} пипсов, "
                   f"лот={lot_size}, риск в валюте=${risk_amount:.2f}")
        
        # Возвращаем размер лота, риск в валюте и риск в процентах
        return lot_size, risk_amount, max_risk
    
    def calculate_take_profit(self, stop_loss_pips, custom_rr_ratio=None):
        """
        Расчет оптимального тейк-профита на основе исторических данных
        
        Параметры:
        stop_loss_pips (int): Размер стоп-лосса в пипсах
        custom_rr_ratio (float, optional): Пользовательское соотношение риск/доходность
        
        Возвращает:
        int: Размер тейк-профита в пипсах
        """
        # Используем предоставленное соотношение риск/доходность, если оно указано
        if custom_rr_ratio is not None:
            return int(stop_loss_pips * custom_rr_ratio)
        
        # Адаптивное соотношение риск/доходность на основе винрейта
        win_rate = self.get_win_rate()
        
        # Формула для расчета оптимального соотношения R/R на основе винрейта
        # Чем ниже винрейт, тем выше должно быть соотношение R/R для положительного EV
        if win_rate <= 0.35:
            rr_ratio = 3.0  # Для низкого винрейта нужно высокое R/R
        elif win_rate <= 0.45:
            rr_ratio = 2.5
        elif win_rate <= 0.55:
            rr_ratio = 2.0
        else:
            rr_ratio = 1.5  # Для высокого винрейта можно использовать более низкое R/R
        
        # Для новых систем без статистики используем стандартное значение
        if self.total_trades < 10:
            rr_ratio = 2.0
        
        # Вычисляем размер тейк-профита
        take_profit_pips = int(stop_loss_pips * rr_ratio)
        
        logger.info(f"Расчет TP: SL={stop_loss_pips} пипсов, R/R={rr_ratio}, TP={take_profit_pips} пипсов")
        
        return take_profit_pips
    
    def should_adjust_stop_loss(self, profit_pips, stop_loss_pips):
        """
        Определение необходимости переноса стоп-лосса в безубыток или трейлинг-стопа
        
        Параметры:
        profit_pips (int): Текущая прибыль в пипсах
        stop_loss_pips (int): Размер стоп-лосса в пипсах от входа
        
        Возвращает:
        tuple: (нужно_двигать, новый_стоп_в_пипсах_от_входа)
        """
        # Обычно стоп переносится в безубыток, когда прибыль достигает 75% от стоп-лосса
        breakeven_threshold = stop_loss_pips * 0.75
        
        # Трейлинг стоп начинается, когда прибыль достигает первоначального стоп-лосса
        trailing_threshold = stop_loss_pips
        
        # Логика переноса стопа
        if profit_pips >= trailing_threshold:
            # Трейлинг-стоп: двигаем стоп на 50% от прибыли сверх порога
            trailing_amount = (profit_pips - trailing_threshold) * 0.5
            new_stop = -stop_loss_pips + trailing_amount  # отрицательное значение становится положительным
            
            logger.info(f"Рекомендация трейлинг-стопа: профит={profit_pips} пипсов, "
                       f"новый стоп={new_stop:.1f} пипсов от входа")
            
            return True, new_stop
        
        elif profit_pips >= breakeven_threshold:
            # Безубыток: переносим стоп в точку входа с небольшим запасом (1-5 пипсов)
            buffer = 2  # пипсы
            
            logger.info(f"Рекомендация переноса в безубыток: профит={profit_pips} пипсов, "
                       f"новый стоп=-{buffer} пипсов от входа")
            
            return True, -buffer  # отрицательное значение означает пипсы ниже входа
        
        # В остальных случаях не трогаем стоп
        return False, -stop_loss_pips
    
    def should_close_trade_early(self, profit_pips, stop_loss_pips, time_in_trade_hours, max_trade_duration=48):
        """
        Определение необходимости раннего закрытия сделки на основе различных факторов
        
        Параметры:
        profit_pips (int): Текущая прибыль в пипсах
        stop_loss_pips (int): Размер стоп-лосса в пипсах
        time_in_trade_hours (float): Время в сделке в часах
        max_trade_duration (int): Максимальная длительность сделки в часах
        
        Возвращает:
        tuple: (нужно_закрывать, причина)
        """
        # Закрытие по времени
        if time_in_trade_hours >= max_trade_duration:
            reason = f"Превышена максимальная длительность сделки ({time_in_trade_hours:.1f} >= {max_trade_duration} часов)"
            return True, reason
        
        # Закрытие при частичной прибыли и смене направления рынка
        # (требует дополнительного анализа)
        
        # Закрытие при достижении определенного уровня прибыли, но замедлении движения
        partial_target = stop_loss_pips * 1.5  # 150% от стоп-лосса
        
        if profit_pips >= partial_target and time_in_trade_hours > 24:
            reason = f"Достигнута частичная цель ({profit_pips:.1f} пипсов) и торговля идет более 24 часов"
            return True, reason
        
        # Нет оснований для раннего закрытия
        return False, None
    
    def get_trade_recommendations(self):
        """
        Получение рекомендаций по торговым параметрам на основе статистики
        
        Возвращает:
        dict: Рекомендации по торговле
        """
        win_rate = self.get_win_rate()
        win_loss_ratio = self.get_average_win_loss_ratio()
        
        recommendations = {
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "total_trades": self.total_trades,
            "current_drawdown": self.current_drawdown,
            "recommended_risk_per_trade": None,
            "recommended_r_r_ratio": None,
            "recommended_approach": None,
            "position_sizing_method": None,
            "confidence_level": "low" if self.total_trades < 20 else "medium" if self.total_trades < 50 else "high"
        }
        
        # Недостаточно данных для надежных рекомендаций
        if self.total_trades < 10:
            recommendations["recommended_risk_per_trade"] = 0.01  # 1% консервативный риск
            recommendations["recommended_r_r_ratio"] = 2.0  # Стандартное соотношение R:R
            recommendations["recommended_approach"] = "консервативный - недостаточно данных для анализа"
            recommendations["position_sizing_method"] = "fixed_percent"
            return recommendations
        
        # Расчет рекомендуемого риска на сделку
        if win_rate >= 0.6:
            # Высокий винрейт позволяет немного повысить риск
            recommended_risk = min(0.02, self.risk_per_trade * 1.2)
            position_method = "optimal_f"
            approach = "умеренно-агрессивный"
        elif win_rate >= 0.45:
            # Средний винрейт - стандартный риск
            recommended_risk = self.risk_per_trade
            position_method = "fixed_percent"
            approach = "сбалансированный"
        else:
            # Низкий винрейт - уменьшаем риск
            recommended_risk = max(0.005, self.risk_per_trade * 0.7)
            position_method = "kelly"
            approach = "консервативный"
        
        # Расчет рекомендуемого соотношения риск/доходность
        if win_rate < 0.4:
            recommended_r_r = 3.0  # Низкий винрейт требует высокого R:R
        elif win_rate < 0.5:
            recommended_r_r = 2.5
        elif win_rate < 0.6:
            recommended_r_r = 2.0
        else:
            recommended_r_r = 1.5  # Высокий винрейт позволяет снизить R:R
        
        # Корректировка на основе соотношения выигрыш/проигрыш
        if win_loss_ratio < 1.0:
            # Проигрыши в среднем больше выигрышей - нужно компенсировать более высоким R:R
            recommended_r_r += 0.5
            approach = "консервативный"
        
        # Корректировка на основе текущей просадки
        if self.current_drawdown > 0.1:  # Просадка > 10%
            recommended_risk = max(0.005, recommended_risk * 0.7)  # Уменьшаем риск
            approach = "восстановительный"
            position_method = "kelly"  # Более консервативный метод
        
        # Корректировка на основе серии убытков
        if self.consecutive_losses >= 3:
            recommended_risk = max(0.005, recommended_risk * 0.5)  # Существенно снижаем риск
            approach = "восстановительный после серии убытков"
        
        # Заполняем рекомендации
        recommendations["recommended_risk_per_trade"] = round(recommended_risk, 4)
        recommendations["recommended_r_r_ratio"] = recommended_r_r
        recommendations["recommended_approach"] = approach
        recommendations["position_sizing_method"] = position_method
        
        return recommendations
    
    def get_equity_curve(self):
        """
        Создает данные для графика кривой капитала
        
        Возвращает:
        pandas.DataFrame: Данные для построения графика
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        # Преобразуем историю сделок в DataFrame
        df = pd.DataFrame(self.trade_history)
        
        # Преобразуем строковое представление времени в datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Сортируем по времени
        df = df.sort_values('time')
        
        # Рассчитываем кумулятивные показатели
        df['cumulative_profit'] = df['profit'].cumsum()
        df['equity'] = df['balance_after']
        df['drawdown'] = df['drawdown'] * 100  # в процентах
        
        # Добавляем скользящие средние
        if len(df) >= 10:
            df['ma10_profit'] = df['profit'].rolling(window=10).mean()
        
        return df
    
    def get_statistics_summary(self):
        """
        Создает сводный отчет по всем статистикам риск-менеджмента
        
        Возвращает:
        dict: Статистика торговли
        """
        stats = {
            "account_balance": self.account_balance,
            "peak_balance": self.peak_balance,
            "current_drawdown": self.current_drawdown * 100,  # в процентах
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.get_win_rate() * 100,  # в процентах
            "win_loss_ratio": self.get_average_win_loss_ratio(),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "daily_risk_used": self.daily_risk_used * 100,  # в процентах
            "daily_loss": self.daily_loss,
            "daily_risk_limit": self.max_daily_risk * 100,  # в процентах
            "risk_per_trade": self.risk_per_trade * 100,  # в процентах
            "position_sizing_method": self.position_sizing_method,
            "date": datetime.now().isoformat()
        }
        
        # Добавляем рекомендации
        recommendations = self.get_trade_recommendations()
        stats.update({
            "recommended_risk": recommendations["recommended_risk_per_trade"] * 100 if recommendations["recommended_risk_per_trade"] else None,
            "recommended_rr": recommendations["recommended_r_r_ratio"],
            "recommended_approach": recommendations["recommended_approach"],
            "recommended_position_method": recommendations["position_sizing_method"]
        })
        
        # Добавляем анализ прибыльности за последние 10, 20 и 30 сделок
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            if len(df) >= 10:
                stats["last_10_trades_profit"] = df['profit'].tail(10).sum()
                stats["last_10_trades_win_rate"] = (df['profit'].tail(10) > 0).mean() * 100
            
            if len(df) >= 20:
                stats["last_20_trades_profit"] = df['profit'].tail(20).sum()
                stats["last_20_trades_win_rate"] = (df['profit'].tail(20) > 0).mean() * 100
            
            if len(df) >= 30:
                stats["last_30_trades_profit"] = df['profit'].tail(30).sum()
                stats["last_30_trades_win_rate"] = (df['profit'].tail(30) > 0).mean() * 100
        
        return stats
    
    def export_statistics(self, format="json"):
        """
        Экспорт статистики торговли в файл
        
        Параметры:
        format (str): Формат экспорта ("json", "csv", "html")
        
        Возвращает:
        str: Путь к экспортированному файлу
        """
        stats = self.get_statistics_summary()
        
        # Генерируем имя файла с датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_stats_{timestamp}.{format}"
        filepath = os.path.join(self.data_dir, filename)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=4)
        
        elif format == "csv":
            # Преобразуем вложенные структуры в плоскую таблицу
            flat_stats = {}
            for k, v in stats.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_stats[f"{k}_{sub_k}"] = sub_v
                else:
                    flat_stats[k] = v
            
            # Сохраняем как CSV
            pd.DataFrame([flat_stats]).to_csv(filepath, index=False)
        
        elif format == "html":
            # Простой HTML-отчет
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Статистика торговли {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Статистика торговли</h1>
                <p>Время создания: {timestamp}</p>
                
                <h2>Общая информация</h2>
                <table>
                    <tr><th>Показатель</th><th>Значение</th></tr>
                    <tr><td>Баланс счета</td><td>{stats['account_balance']:.2f}</td></tr>
                    <tr><td>Пиковый баланс</td><td>{stats['peak_balance']:.2f}</td></tr>
                    <tr><td>Текущая просадка</td><td class="{'negative' if stats['current_drawdown'] > 0 else ''}">{stats['current_drawdown']:.2f}%</td></tr>
                    <tr><td>Всего сделок</td><td>{stats['total_trades']}</td></tr>
                    <tr><td>Выигрышные сделки</td><td>{stats['winning_trades']}</td></tr>
                    <tr><td>Проигрышные сделки</td><td>{stats['losing_trades']}</td></tr>
                    <tr><td>Винрейт</td><td>{stats['win_rate']:.2f}%</td></tr>
                    <tr><td>Соотношение выигрыш/проигрыш</td><td>{stats['win_loss_ratio']:.2f}</td></tr>
                </table>
                
                <h2>Параметры риска</h2>
                <table>
                    <tr><th>Показатель</th><th>Значение</th></tr>
                    <tr><td>Риск на сделку</td><td>{stats['risk_per_trade']:.2f}%</td></tr>
                    <tr><td>Метод расчета позиции</td><td>{stats['position_sizing_method']}</td></tr>
                    <tr><td>Использованный дневной риск</td><td>{stats['daily_risk_used']:.2f}%</td></tr>
                    <tr><td>Лимит дневного риска</td><td>{stats['daily_risk_limit']:.2f}%</td></tr>
                </table>
                
                <h2>Рекомендации</h2>
                <table>
                    <tr><th>Показатель</th><th>Значение</th></tr>
                    <tr><td>Рекомендуемый риск на сделку</td><td>{stats['recommended_risk'] if stats['recommended_risk'] else 'Н/Д'}%</td></tr>
                    <tr><td>Рекомендуемое соотношение R:R</td><td>{stats['recommended_rr'] if stats['recommended_rr'] else 'Н/Д'}</td></tr>
                    <tr><td>Рекомендуемый подход</td><td>{stats['recommended_approach'] if stats['recommended_approach'] else 'Н/Д'}</td></tr>
                    <tr><td>Рекомендуемый метод расчета позиции</td><td>{stats['recommended_position_method'] if stats['recommended_position_method'] else 'Н/Д'}</td></tr>
                </table>
            </body>
            </html>
            """
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {format}")
        
        logger.info(f"Статистика экспортирована в {filepath}")
        return filepath

# Глобальный экземпляр риск-менеджера
_risk_manager = None

def get_risk_manager(account_balance=None, init_if_none=True, **kwargs):
    """
    Получение глобального экземпляра риск-менеджера
    
    Параметры:
    account_balance (float, optional): Текущий баланс аккаунта
    init_if_none (bool): Инициализировать экземпляр, если он не существует
    **kwargs: Дополнительные параметры для RiskManager
    
    Возвращает:
    RiskManager: Экземпляр риск-менеджера
    """
    global _risk_manager
    
    if _risk_manager is None and init_if_none:
        _risk_manager = RiskManager(account_balance=account_balance, **kwargs)
    elif account_balance is not None and _risk_manager is not None:
        # Обновляем баланс, если он предоставлен
        _risk_manager.update_balance(account_balance)
    
    return _risk_manager

def reset_risk_manager():
    """Сброс глобального экземпляра риск-менеджера"""
    global _risk_manager
    _risk_manager = None
    logger.info("Глобальный экземпляр риск-менеджера сброшен")

# Удобная функция для быстрого расчета размера позиции
def quick_position_size(account_balance, risk_percent, stop_loss_pips, symbol_info):
    """
    Быстрый расчет размера позиции с базовыми проверками
    
    Параметры:
    account_balance (float): Баланс счета
    risk_percent (float): Процент риска (0.01 = 1%)
    stop_loss_pips (int): Размер стоп-лосса в пипсах
    symbol_info (dict): Информация о символе
    
    Возвращает:
    float: Размер позиции в лотах
    """
    # Создаем временный экземпляр риск-менеджера
    temp_manager = RiskManager(account_balance=account_balance, risk_per_trade=risk_percent)
    
    # Рассчитываем размер позиции
    lot_size, _, _ = temp_manager.calculate_position_size(stop_loss_pips, 0, symbol_info)
    
    return lot_size