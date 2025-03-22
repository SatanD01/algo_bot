import pandas as pd
import numpy as np
import os
import time
import logging
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm import tqdm

# Настройка логирования
logger = logging.getLogger(__name__)

class TradeJournal:
    """
    Расширенный журнал сделок для реальной торговли с расчетом статистики
    и возможностью экспорта данных в различные форматы.
    """
    
    def __init__(self, symbol, account_id=None, journal_dir=None):
        """
        Инициализация журнала сделок
        
        Параметры:
        symbol (str): Торговый символ
        account_id (int, optional): ID торгового счета
        journal_dir (str, optional): Директория для хранения журнала
        """
        self.symbol = symbol
        self.account_id = account_id or "default"
        
        # Создаем директорию для журнала, если она не существует
        if journal_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.journal_dir = os.path.join(base_dir, "trade_journal")
        else:
            self.journal_dir = journal_dir
            
        os.makedirs(self.journal_dir, exist_ok=True)
        
        # Создаем подпапки для различных типов данных
        self.trades_dir = os.path.join(self.journal_dir, "trades")
        self.stats_dir = os.path.join(self.journal_dir, "statistics")
        self.charts_dir = os.path.join(self.journal_dir, "charts")
        self.reports_dir = os.path.join(self.journal_dir, "reports")
        
        for dir_path in [self.trades_dir, self.stats_dir, self.charts_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Инициализация DataFrame для хранения сделок
        self.trades_df = self._load_trades()
        
        # Метаданные журнала
        self.journal_meta = {
            "symbol": symbol,
            "account_id": str(self.account_id),
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "trade_count": len(self.trades_df),
            "version": "1.0.0"
        }
        
        self._save_meta()
        logger.info(f"Инициализирован журнал сделок для {symbol} (Аккаунт: {self.account_id})")
    
    def _get_trades_file_path(self):
        """Получение пути к файлу с данными сделок"""
        return os.path.join(self.trades_dir, f"{self.symbol}_{self.account_id}_trades.csv")
    
    def _get_meta_file_path(self):
        """Получение пути к файлу с метаданными"""
        return os.path.join(self.journal_dir, f"{self.symbol}_{self.account_id}_meta.json")
    
    def _load_trades(self):
        """Загрузка сделок из файла"""
        file_path = self._get_trades_file_path()
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['entry_time', 'exit_time'])
                logger.info(f"Загружено {len(df)} сделок из журнала")
                return df
            except Exception as e:
                logger.error(f"Ошибка при загрузке журнала сделок: {str(e)}")
                # Возвращаем пустой DataFrame с правильной структурой
                return self._create_empty_trades_df()
        else:
            return self._create_empty_trades_df()
    
    def _create_empty_trades_df(self):
        """Создание пустого DataFrame для сделок с правильной структурой"""
        columns = [
            'ticket', 'symbol', 'order', 'lot_size', 'entry_price', 'exit_price',
            'entry_time', 'exit_time', 'stop_loss', 'take_profit', 'result',
            'profit', 'commission', 'swap', 'net_profit', 'duration',
            'setup', 'tf', 'comment', 'tags', 'risk_reward', 'risk_percent',
            'entry_reason', 'exit_reason', 'market_condition', 'screenshot_path'
        ]
        return pd.DataFrame(columns=columns)
    
    def _save_meta(self):
        """Сохранение метаданных журнала"""
        # Обновление времени последнего изменения
        self.journal_meta["last_updated"] = datetime.now().isoformat()
        self.journal_meta["trade_count"] = len(self.trades_df)
        
        with open(self._get_meta_file_path(), 'w', encoding='utf-8') as f:
            json.dump(self.journal_meta, f, ensure_ascii=False, indent=4)
    
    def _save_trades(self):
        """Сохранение сделок в CSV файл"""
        self.trades_df.to_csv(self._get_trades_file_path(), index=False)
        self._save_meta()
        logger.info(f"Сохранено {len(self.trades_df)} сделок в журнал")
    
    def add_trade(self, trade_data):
        """
        Добавление новой сделки в журнал
        
        Параметры:
        trade_data (dict): Данные о сделке
        
        Пример:
        {
            'ticket': 123456,
            'symbol': 'EURUSD',
            'order': 'buy',  # или 'sell'
            'lot_size': 0.1,
            'entry_price': 1.12345,
            'exit_price': 1.12545,
            'entry_time': datetime(2023, 1, 1, 10, 0, 0),
            'exit_time': datetime(2023, 1, 1, 14, 30, 0),
            'stop_loss': 1.12245,
            'take_profit': 1.12645,
            'result': 'win',  # или 'loss'
            'profit': 20.0,
            'commission': -2.0,
            'swap': -0.5,
            'setup': 'BreakerBlock',
            'tf': 'H1',
            'comment': 'Strong momentum after retest',
            'tags': ['trend_following', 'high_volume'],
            'entry_reason': 'Break of key level with momentum',
            'exit_reason': 'Take profit hit',
            'market_condition': 'Trending',
            'screenshot_path': '/path/to/screenshot.png'
        }
        """
        # Проверка наличия обязательных полей
        required_fields = ['ticket', 'symbol', 'order', 'entry_price', 'entry_time']
        for field in required_fields:
            if field not in trade_data:
                logger.error(f"Не указано обязательное поле: {field}")
                return False
        
        # Добавление дополнительных расчетных полей, если они не указаны
        
        # Расчет net_profit
        if 'net_profit' not in trade_data and 'profit' in trade_data:
            commission = trade_data.get('commission', 0)
            swap = trade_data.get('swap', 0)
            trade_data['net_profit'] = trade_data['profit'] + commission + swap
        
        # Расчет duration
        if 'duration' not in trade_data and 'entry_time' in trade_data and 'exit_time' in trade_data:
            if trade_data['exit_time'] and trade_data['entry_time']:
                try:
                    duration = trade_data['exit_time'] - trade_data['entry_time']
                    trade_data['duration'] = duration.total_seconds() / 3600  # в часах
                except Exception as e:
                    logger.warning(f"Не удалось рассчитать продолжительность сделки: {str(e)}")
                    trade_data['duration'] = None
        
        # Расчет risk_reward (отношение риск/доходность)
        if 'risk_reward' not in trade_data and 'stop_loss' in trade_data and 'take_profit' in trade_data:
            try:
                if trade_data['order'] == 'buy':
                    sl_distance = trade_data['entry_price'] - trade_data['stop_loss']
                    tp_distance = trade_data['take_profit'] - trade_data['entry_price']
                else:  # sell
                    sl_distance = trade_data['stop_loss'] - trade_data['entry_price']
                    tp_distance = trade_data['entry_price'] - trade_data['take_profit']
                
                if sl_distance > 0:
                    trade_data['risk_reward'] = tp_distance / sl_distance
                else:
                    trade_data['risk_reward'] = None
            except Exception as e:
                logger.warning(f"Не удалось рассчитать отношение риск/доходность: {str(e)}")
                trade_data['risk_reward'] = None
        
        # Конвертация списка тегов в строку для хранения в CSV
        if 'tags' in trade_data and isinstance(trade_data['tags'], list):
            trade_data['tags'] = ','.join(trade_data['tags'])
        
        # Проверяем, не существует ли уже запись с таким же ticket
        if 'ticket' in self.trades_df.columns:
            existing_trade = self.trades_df[self.trades_df['ticket'] == trade_data['ticket']]
            if not existing_trade.empty:
                logger.warning(f"Сделка с ticket {trade_data['ticket']} уже существует. Обновляем данные.")
                
                # Обновление существующей записи
                for key, value in trade_data.items():
                    if key in self.trades_df.columns:
                        self.trades_df.loc[self.trades_df['ticket'] == trade_data['ticket'], key] = value
                
                logger.info(f"Обновлена сделка: {trade_data['symbol']} {trade_data['order']} ({trade_data['ticket']})")
            else:
                # Добавление новой записи
                # Создаем новый DataFrame с одной записью
                new_trade_df = pd.DataFrame([trade_data])
                
                # Объединяем с существующими данными
                self.trades_df = pd.concat([self.trades_df, new_trade_df], ignore_index=True)
                
                logger.info(f"Добавлена новая сделка: {trade_data['symbol']} {trade_data['order']} ({trade_data['ticket']})")
        else:
            # Если DataFrame пустой
            self.trades_df = pd.DataFrame([trade_data])
            logger.info(f"Добавлена первая сделка: {trade_data['symbol']} {trade_data['order']} ({trade_data['ticket']})")
        
        # Сохраняем обновленные данные
        self._save_trades()
        return True
    
    def update_trade(self, ticket, updates):
        """
        Обновление информации о существующей сделке
        
        Параметры:
        ticket (int): Номер тикета сделки
        updates (dict): Данные для обновления
        
        Возвращает:
        bool: True в случае успешного обновления, иначе False
        """
        if 'ticket' in self.trades_df.columns:
            mask = self.trades_df['ticket'] == ticket
            if not any(mask):
                logger.warning(f"Сделка с ticket {ticket} не найдена")
                return False
            
            # Обновляем указанные поля
            for key, value in updates.items():
                if key in self.trades_df.columns:
                    self.trades_df.loc[mask, key] = value
                else:
                    logger.warning(f"Поле {key} не существует в структуре данных о сделках")
            
            self._save_trades()
            logger.info(f"Обновлена сделка с ticket {ticket}")
            return True
        else:
            logger.warning("Нет данных о сделках")
            return False
    
    def close_trade(self, ticket, exit_price, exit_time, profit, result, commission=0, swap=0, exit_reason=None):
        """
        Закрытие открытой сделки
        
        Параметры:
        ticket (int): Номер тикета сделки
        exit_price (float): Цена выхода
        exit_time (datetime): Время выхода
        profit (float): Прибыль в валюте счета
        result (str): Результат ('win' или 'loss')
        commission (float, optional): Комиссия
        swap (float, optional): Своп
        exit_reason (str, optional): Причина выхода
        
        Возвращает:
        bool: True в случае успешного закрытия, иначе False
        """
        updates = {
            'exit_price': exit_price,
            'exit_time': exit_time,
            'profit': profit,
            'result': result,
            'commission': commission,
            'swap': swap,
            'net_profit': profit + commission + swap
        }
        
        if exit_reason:
            updates['exit_reason'] = exit_reason
        
        # Расчет продолжительности сделки
        if 'entry_time' in self.trades_df.columns:
            mask = self.trades_df['ticket'] == ticket
            if any(mask):
                entry_time = self.trades_df.loc[mask, 'entry_time'].iloc[0]
                if entry_time and exit_time:
                    try:
                        duration = exit_time - entry_time
                        updates['duration'] = duration.total_seconds() / 3600  # в часах
                    except Exception as e:
                        logger.warning(f"Не удалось рассчитать продолжительность сделки: {str(e)}")
        
        return self.update_trade(ticket, updates)
    
    def get_open_trades(self):
        """
        Получение списка открытых сделок
        
        Возвращает:
        DataFrame: DataFrame с открытыми сделками
        """
        if 'exit_time' in self.trades_df.columns:
            # Открытые сделки - это те, у которых exit_time is NaN
            open_trades = self.trades_df[self.trades_df['exit_time'].isna()]
            return open_trades
        else:
            return pd.DataFrame()
    
    def get_closed_trades(self, start_date=None, end_date=None):
        """
        Получение списка закрытых сделок с возможностью фильтрации по дате
        
        Параметры:
        start_date (datetime, optional): Начальная дата фильтра
        end_date (datetime, optional): Конечная дата фильтра
        
        Возвращает:
        DataFrame: DataFrame с закрытыми сделками
        """
        if 'exit_time' not in self.trades_df.columns:
            return pd.DataFrame()
        
        # Закрытые сделки - это те, у которых exit_time is not NaN
        closed_trades = self.trades_df[~self.trades_df['exit_time'].isna()]
        
        # Применяем фильтры по дате, если они указаны
        if start_date is not None:
            closed_trades = closed_trades[closed_trades['exit_time'] >= start_date]
        
        if end_date is not None:
            closed_trades = closed_trades[closed_trades['exit_time'] <= end_date]
        
        return closed_trades
    
    def get_trade_by_ticket(self, ticket):
        """
        Получение информации о сделке по ее тикету
        
        Параметры:
        ticket (int): Номер тикета сделки
        
        Возвращает:
        Series: Данные о сделке или None, если сделка не найдена
        """
        if 'ticket' in self.trades_df.columns:
            mask = self.trades_df['ticket'] == ticket
            if any(mask):
                return self.trades_df[mask].iloc[0]
        
        return None
    
    def calculate_statistics(self, start_date=None, end_date=None, include_open=False):
        """
        Расчет статистики по сделкам
        
        Параметры:
        start_date (datetime, optional): Начальная дата для расчета статистики
        end_date (datetime, optional): Конечная дата для расчета статистики
        include_open (bool): Включать ли открытые сделки в статистику
        
        Возвращает:
        dict: Словарь со статистикой
        """
        # Получаем закрытые сделки
        closed_trades = self.get_closed_trades(start_date, end_date)
        
        # Добавляем открытые сделки, если нужно
        if include_open:
            open_trades = self.get_open_trades()
            all_trades = pd.concat([closed_trades, open_trades], ignore_index=True)
        else:
            all_trades = closed_trades
        
        if all_trades.empty:
            logger.warning("Нет данных для расчета статистики")
            return {
                "total_trades": 0,
                "period_start": start_date.isoformat() if start_date else None,
                "period_end": end_date.isoformat() if end_date else None,
                "include_open": include_open,
                "timestamp": datetime.now().isoformat()
            }
        
        # Базовая статистика
        total_trades = len(all_trades)
        win_trades = len(all_trades[all_trades['result'] == 'win'])
        loss_trades = len(all_trades[all_trades['result'] == 'loss'])
        winrate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Прибыль и убытки
        total_profit = all_trades['profit'].sum() if 'profit' in all_trades.columns else 0
        total_net_profit = all_trades['net_profit'].sum() if 'net_profit' in all_trades.columns else total_profit
        total_commission = all_trades['commission'].sum() if 'commission' in all_trades.columns else 0
        total_swap = all_trades['swap'].sum() if 'swap' in all_trades.columns else 0
        
        # Средняя прибыль и убытки на сделку
        avg_profit = all_trades[all_trades['result'] == 'win']['profit'].mean() if win_trades > 0 else 0
        avg_loss = all_trades[all_trades['result'] == 'loss']['profit'].mean() if loss_trades > 0 else 0
        
        # Максимальная прибыль и убыток
        max_profit = all_trades['profit'].max() if 'profit' in all_trades.columns else 0
        min_profit = all_trades['profit'].min() if 'profit' in all_trades.columns else 0
        
        # Профит-фактор
        gross_profit = all_trades[all_trades['profit'] > 0]['profit'].sum() if 'profit' in all_trades.columns else 0
        gross_loss = abs(all_trades[all_trades['profit'] < 0]['profit'].sum()) if 'profit' in all_trades.columns else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Отношение средней прибыли к среднему убытку
        win_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        # Максимальная просадка
        if 'profit' in all_trades.columns:
            all_trades = all_trades.sort_values(by='exit_time')
            all_trades['cumulative_profit'] = all_trades['profit'].cumsum()
            all_trades['running_max'] = all_trades['cumulative_profit'].cummax()
            all_trades['drawdown'] = all_trades['cumulative_profit'] - all_trades['running_max']
            max_drawdown = abs(all_trades['drawdown'].min())
            max_drawdown_percent = (max_drawdown / all_trades['running_max'].max()) * 100 if all_trades['running_max'].max() > 0 else 0
        else:
            max_drawdown = 0
            max_drawdown_percent = 0
        
        # Серии выигрышей и проигрышей
        if 'result' in all_trades.columns:
            all_trades = all_trades.sort_values(by='exit_time')
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            current_streak_type = None
            
            for result in all_trades['result']:
                if result == current_streak_type:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = result
                
                if result == 'win':
                    max_win_streak = max(max_win_streak, current_streak)
                elif result == 'loss':
                    max_loss_streak = max(max_loss_streak, current_streak)
        else:
            max_win_streak = 0
            max_loss_streak = 0
        
        # Статистика по типам сделок (buy/sell)
        order_stats = {}
        if 'order' in all_trades.columns:
            for order_type, group in all_trades.groupby('order'):
                group_wins = len(group[group['result'] == 'win'])
                group_total = len(group)
                group_winrate = (group_wins / group_total * 100) if group_total > 0 else 0
                group_profit = group['profit'].sum() if 'profit' in group.columns else 0
                
                order_stats[order_type] = {
                    "trades": group_total,
                    "wins": group_wins,
                    "losses": group_total - group_wins,
                    "winrate": group_winrate,
                    "profit": group_profit
                }
        
        # Статистика по сетапам
        setup_stats = {}
        if 'setup' in all_trades.columns:
            for setup, group in all_trades.groupby('setup'):
                if pd.isna(setup) or setup == '':
                    continue
                
                group_wins = len(group[group['result'] == 'win'])
                group_total = len(group)
                group_winrate = (group_wins / group_total * 100) if group_total > 0 else 0
                group_profit = group['profit'].sum() if 'profit' in group.columns else 0
                
                setup_stats[setup] = {
                    "trades": group_total,
                    "wins": group_wins,
                    "losses": group_total - group_wins,
                    "winrate": group_winrate,
                    "profit": group_profit
                }
        
        # Статистика по таймфреймам
        tf_stats = {}
        if 'tf' in all_trades.columns:
            for tf, group in all_trades.groupby('tf'):
                if pd.isna(tf) or tf == '':
                    continue
                
                group_wins = len(group[group['result'] == 'win'])
                group_total = len(group)
                group_winrate = (group_wins / group_total * 100) if group_total > 0 else 0
                group_profit = group['profit'].sum() if 'profit' in group.columns else 0
                
                tf_stats[tf] = {
                    "trades": group_total,
                    "wins": group_wins,
                    "losses": group_total - group_wins,
                    "winrate": group_winrate,
                    "profit": group_profit
                }
        
        # Статистика по дням недели
        day_stats = {}
        if 'entry_time' in all_trades.columns:
            all_trades['weekday'] = all_trades['entry_time'].dt.day_name()
            for day, group in all_trades.groupby('weekday'):
                group_wins = len(group[group['result'] == 'win'])
                group_total = len(group)
                group_winrate = (group_wins / group_total * 100) if group_total > 0 else 0
                group_profit = group['profit'].sum() if 'profit' in group.columns else 0
                
                day_stats[day] = {
                    "trades": group_total,
                    "wins": group_wins,
                    "losses": group_total - group_wins,
                    "winrate": group_winrate,
                    "profit": group_profit
                }
        
        # Статистика по месяцам
        month_stats = {}
        if 'entry_time' in all_trades.columns:
            all_trades['year_month'] = all_trades['entry_time'].dt.strftime('%Y-%m')
            for month, group in all_trades.groupby('year_month'):
                group_wins = len(group[group['result'] == 'win'])
                group_total = len(group)
                group_winrate = (group_wins / group_total * 100) if group_total > 0 else 0
                group_profit = group['profit'].sum() if 'profit' in group.columns else 0
                
                month_stats[month] = {
                    "trades": group_total,
                    "wins": group_wins,
                    "losses": group_total - group_wins,
                    "winrate": group_winrate,
                    "profit": group_profit
                }
        
        # Собираем все статистические данные в один словарь
        stats = {
            "total_trades": total_trades,
            "wins": win_trades,
            "losses": loss_trades,
            "winrate": winrate,
            "total_profit": total_profit,
            "total_net_profit": total_net_profit,
            "total_commission": total_commission,
            "total_swap": total_swap,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "max_profit": max_profit,
            "min_profit": min_profit,
            "profit_factor": profit_factor,
            "win_loss_ratio": win_loss_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "period_start": start_date.isoformat() if start_date else min(all_trades['entry_time']).isoformat() if 'entry_time' in all_trades.columns else None,
            "period_end": end_date.isoformat() if end_date else max(all_trades['exit_time']).isoformat() if 'exit_time' in all_trades.columns and not all_trades['exit_time'].isna().all() else None,
            "include_open": include_open,
            "order_stats": order_stats,
            "setup_stats": setup_stats,
            "tf_stats": tf_stats,
            "day_stats": day_stats,
            "month_stats": month_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
    
    def save_statistics(self, stats, filename=None):
        """
        Сохранение статистики в файл
        
        Параметры:
        stats (dict): Словарь со статистикой
        filename (str, optional): Имя файла для сохранения
        
        Возвращает:
        str: Путь к файлу со статистикой
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.symbol}_{self.account_id}_stats_{timestamp}.json"
        
        file_path = os.path.join(self.stats_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4, default=str)
        
        logger.info(f"Статистика сохранена в {file_path}")
        return file_path
    
    def generate_performance_report(self, start_date=None, end_date=None, include_open=False):
        """
        Генерация полного отчета о производительности торговли
        
        Параметры:
        start_date (datetime, optional): Начальная дата для отчета
        end_date (datetime, optional): Конечная дата для отчета
        include_open (bool): Включать ли открытые сделки в отчет
        
        Возвращает:
        str: Путь к созданному отчету
        """
        # Рассчитываем статистику
        stats = self.calculate_statistics(start_date, end_date, include_open)
        
        # Создаем название отчета
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.reports_dir, f"{self.symbol}_{self.account_id}_report_{timestamp}.html")
        
        # Получаем сделки для указанного периода
        closed_trades = self.get_closed_trades(start_date, end_date)
        
        if include_open:
            open_trades = self.get_open_trades()
            all_trades = pd.concat([closed_trades, open_trades], ignore_index=True)
        else:
            all_trades = closed_trades
        
        if all_trades.empty:
            logger.warning("Нет данных для создания отчета")
            
            # Создаем пустой отчет с информацией об отсутствии данных
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Отчет о торговле {self.symbol} - Нет данных</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .no-data {{ text-align: center; margin-top: 100px; color: #888; }}
                    </style>
                </head>
                <body>
                    <h1>Отчет о торговле {self.symbol}</h1>
                    <p>Период: {start_date or 'Все время'} - {end_date or 'Все время'}</p>
                    <div class="no-data">
                        <h2>Нет данных для отображения</h2>
                        <p>За указанный период не было совершено сделок.</p>
                    </div>
                </body>
                </html>
                """)
            
            logger.info(f"Создан пустой отчет (нет данных): {report_file}")
            return report_file
        
        # Создаем графики для отчета
        charts_data = {}
        
        # Сохраняем статистику в файл
        stats_file = self.save_statistics(stats)
        
        # Создаем баланс-кривую
        if 'profit' in all_trades.columns and 'exit_time' in all_trades.columns:
            balance_chart_path = os.path.join(self.charts_dir, f"{self.symbol}_balance_{timestamp}.png")
            
            try:
                # Сортируем сделки по времени выхода
                sorted_trades = all_trades.sort_values(by='exit_time')
                
                # Рассчитываем накопленную прибыль
                sorted_trades['cumulative_profit'] = sorted_trades['profit'].cumsum()
                
                # Создаем график
                plt.figure(figsize=(12, 6))
                plt.plot(sorted_trades['exit_time'], sorted_trades['cumulative_profit'], 'b-')
                plt.grid(True, alpha=0.3)
                plt.title(f'Динамика баланса для {self.symbol}')
                plt.xlabel('Время')
                plt.ylabel('Прибыль')
                
                # Форматируем ось X для отображения дат
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.xticks(rotation=45)
                
                # Проверяем, есть ли положительные и отрицательные сделки
                positive_trades = sorted_trades[sorted_trades['profit'] > 0]
                negative_trades = sorted_trades[sorted_trades['profit'] < 0]
                
                # Добавляем маркеры для сделок
                if not positive_trades.empty:
                    plt.scatter(positive_trades['exit_time'], positive_trades['cumulative_profit'], color='green', marker='^', s=50, label='Прибыльные сделки')
                
                if not negative_trades.empty:
                    plt.scatter(negative_trades['exit_time'], negative_trades['cumulative_profit'], color='red', marker='v', s=50, label='Убыточные сделки')
                
                plt.legend()
                plt.tight_layout()
                
                # Сохраняем график
                plt.savefig(balance_chart_path, dpi=100)
                plt.close()
                
                charts_data['balance_chart'] = os.path.relpath(balance_chart_path, self.journal_dir)
            except Exception as e:
                logger.error(f"Ошибка при создании графика баланса: {str(e)}")
        
        # Создаем график распределения прибыли
        if 'profit' in all_trades.columns:
            profit_dist_chart_path = os.path.join(self.charts_dir, f"{self.symbol}_profit_dist_{timestamp}.png")
            
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(all_trades['profit'], bins=20, alpha=0.7, color='blue', edgecolor='black')
                plt.grid(True, alpha=0.3)
                plt.title(f'Распределение прибыли для {self.symbol}')
                plt.xlabel('Прибыль')
                plt.ylabel('Количество сделок')
                plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
                
                # Добавляем среднюю прибыль
                avg_profit = all_trades['profit'].mean()
                plt.axvline(x=avg_profit, color='green', linestyle='-', linewidth=1, label=f'Средняя прибыль: {avg_profit:.2f}')
                
                plt.legend()
                plt.tight_layout()
                
                # Сохраняем график
                plt.savefig(profit_dist_chart_path, dpi=100)
                plt.close()
                
                charts_data['profit_dist_chart'] = os.path.relpath(profit_dist_chart_path, self.journal_dir)
            except Exception as e:
                logger.error(f"Ошибка при создании графика распределения прибыли: {str(e)}")
        
        # Создаем график винрейта по сетапам
        if 'setup' in all_trades.columns and 'result' in all_trades.columns:
            setup_chart_path = os.path.join(self.charts_dir, f"{self.symbol}_setup_stats_{timestamp}.png")
            
            try:
                # Группируем данные по сетапам
                setup_groups = all_trades.groupby('setup')
                setups = []
                winrates = []
                counts = []
                
                for setup, group in setup_groups:
                    if pd.isna(setup) or setup == '':
                        continue
                    
                    winrate = (group['result'] == 'win').mean() * 100
                    setups.append(setup)
                    winrates.append(winrate)
                    counts.append(len(group))
                
                if setups:
                    plt.figure(figsize=(12, 6))
                    
                    # Создаем bar plot с винрейтом
                    ax1 = plt.gca()
                    bars = ax1.bar(setups, winrates, color='skyblue', alpha=0.7)
                    
                    # Добавляем значения над столбцами
                    for bar, wr, cnt in zip(bars, winrates, counts):
                        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                                f'{wr:.1f}% ({cnt})',
                                ha='center', va='bottom', rotation=0, fontsize=9)
                    
                    ax1.set_ylabel('Винрейт (%)')
                    ax1.set_title(f'Винрейт по сетапам для {self.symbol}')
                    
                    # Добавляем линию среднего винрейта
                    avg_winrate = (all_trades['result'] == 'win').mean() * 100
                    ax1.axhline(y=avg_winrate, color='red', linestyle='--', label=f'Средний винрейт: {avg_winrate:.1f}%')
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, axis='y', alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    
                    # Сохраняем график
                    plt.savefig(setup_chart_path, dpi=100)
                    plt.close()
                    
                    charts_data['setup_chart'] = os.path.relpath(setup_chart_path, self.journal_dir)
            except Exception as e:
                logger.error(f"Ошибка при создании графика статистики по сетапам: {str(e)}")
        
        # Создаем HTML-отчет
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Отчет о торговле {self.symbol}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .section {{ margin-bottom: 30px; }}
                        .chart-container {{ text-align: center; margin: 20px 0; }}
                        .chart {{ max-width: 100%; height: auto; }}
                        .stats-row {{ display: flex; flex-wrap: wrap; }}
                        .stats-box {{ flex: 1; min-width: 200px; background-color: #f8f8f8; border: 1px solid #ddd; 
                                     border-radius: 5px; padding: 15px; margin: 10px; text-align: center; }}
                        .win {{ color: green; }}
                        .loss {{ color: red; }}
                        .highlight {{ background-color: #fffde7; font-weight: bold; }}
                    </style>
                </head>
                <body>
                    <h1>Отчет о торговле {self.symbol}</h1>
                    <p>Период: {start_date or 'Все время'} - {end_date or 'Все время'}</p>
                    <p>Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="section">
                        <h2>Общая статистика</h2>
                        <div class="stats-row">
                            <div class="stats-box">
                                <h3>Всего сделок</h3>
                                <p style="font-size: 24px;">{stats['total_trades']}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Винрейт</h3>
                                <p style="font-size: 24px;">{stats['winrate']:.2f}%</p>
                                <p>Выигрыши: {stats['wins']} / Проигрыши: {stats['losses']}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Общая прибыль</h3>
                                <p style="font-size: 24px;" class="{'win' if stats['total_profit'] >= 0 else 'loss'}">{stats['total_profit']:.2f}</p>
                                <p>Чистая прибыль: {stats['total_net_profit']:.2f}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Профит-фактор</h3>
                                <p style="font-size: 24px;">{stats['profit_factor'] if stats['profit_factor'] != float('inf') else '∞'}</p>
                                <p>Выигрыш/Проигрыш: {stats['win_loss_ratio'] if stats['win_loss_ratio'] != float('inf') else '∞'}</p>
                            </div>
                        </div>
                        
                        <div class="stats-row">
                            <div class="stats-box">
                                <h3>Средняя прибыль</h3>
                                <p class="win">{stats['avg_profit']:.2f}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Средний убыток</h3>
                                <p class="loss">{stats['avg_loss']:.2f}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Максимальная прибыль</h3>
                                <p class="win">{stats['max_profit']:.2f}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Максимальный убыток</h3>
                                <p class="loss">{stats['min_profit']:.2f}</p>
                            </div>
                        </div>
                        
                        <div class="stats-row">
                            <div class="stats-box">
                                <h3>Макс. просадка</h3>
                                <p class="loss">{stats['max_drawdown']:.2f} ({stats['max_drawdown_percent']:.2f}%)</p>
                            </div>
                            <div class="stats-box">
                                <h3>Серия выигрышей</h3>
                                <p>{stats['max_win_streak']}</p>
                            </div>
                            <div class="stats-box">
                                <h3>Серия проигрышей</h3>
                                <p>{stats['max_loss_streak']}</p>
                            </div>
                        </div>
                    </div>
                """)
                
                # Добавляем графики
                if charts_data:
                    f.write("""
                    <div class="section">
                        <h2>Графики</h2>
                    """)
                    
                    if 'balance_chart' in charts_data:
                        f.write(f"""
                        <div class="chart-container">
                            <h3>Динамика баланса</h3>
                            <img src="../{charts_data['balance_chart']}" alt="Динамика баланса" class="chart">
                        </div>
                        """)
                    
                    if 'profit_dist_chart' in charts_data:
                        f.write(f"""
                        <div class="chart-container">
                            <h3>Распределение прибыли</h3>
                            <img src="../{charts_data['profit_dist_chart']}" alt="Распределение прибыли" class="chart">
                        </div>
                        """)
                    
                    if 'setup_chart' in charts_data:
                        f.write(f"""
                        <div class="chart-container">
                            <h3>Статистика по сетапам</h3>
                            <img src="../{charts_data['setup_chart']}" alt="Статистика по сетапам" class="chart">
                        </div>
                        """)
                    
                    f.write("</div>")
                
                # Добавляем статистику по типам ордеров
                if stats['order_stats']:
                    f.write("""
                    <div class="section">
                        <h2>Статистика по типам ордеров</h2>
                        <table>
                            <tr>
                                <th>Тип</th>
                                <th>Сделки</th>
                                <th>Выигрыши</th>
                                <th>Проигрыши</th>
                                <th>Винрейт</th>
                                <th>Прибыль</th>
                            </tr>
                    """)
                    
                    for order_type, order_data in stats['order_stats'].items():
                        profit_class = 'win' if order_data['profit'] >= 0 else 'loss'
                        f.write(f"""
                            <tr>
                                <td>{order_type}</td>
                                <td>{order_data['trades']}</td>
                                <td>{order_data['wins']}</td>
                                <td>{order_data['losses']}</td>
                                <td>{order_data['winrate']:.2f}%</td>
                                <td class="{profit_class}">{order_data['profit']:.2f}</td>
                            </tr>
                        """)
                    
                    f.write("</table></div>")
                
                # Добавляем статистику по сетапам
                if stats['setup_stats']:
                    f.write("""
                    <div class="section">
                        <h2>Статистика по сетапам</h2>
                        <table>
                            <tr>
                                <th>Сетап</th>
                                <th>Сделки</th>
                                <th>Выигрыши</th>
                                <th>Проигрыши</th>
                                <th>Винрейт</th>
                                <th>Прибыль</th>
                            </tr>
                    """)
                    
                    for setup, setup_data in stats['setup_stats'].items():
                        profit_class = 'win' if setup_data['profit'] >= 0 else 'loss'
                        f.write(f"""
                            <tr>
                                <td>{setup}</td>
                                <td>{setup_data['trades']}</td>
                                <td>{setup_data['wins']}</td>
                                <td>{setup_data['losses']}</td>
                                <td>{setup_data['winrate']:.2f}%</td>
                                <td class="{profit_class}">{setup_data['profit']:.2f}</td>
                            </tr>
                        """)
                    
                    f.write("</table></div>")
                
                # Добавляем статистику по таймфреймам
                if stats['tf_stats']:
                    f.write("""
                    <div class="section">
                        <h2>Статистика по таймфреймам</h2>
                        <table>
                            <tr>
                                <th>Таймфрейм</th>
                                <th>Сделки</th>
                                <th>Выигрыши</th>
                                <th>Проигрыши</th>
                                <th>Винрейт</th>
                                <th>Прибыль</th>
                            </tr>
                    """)
                    
                    for tf, tf_data in stats['tf_stats'].items():
                        profit_class = 'win' if tf_data['profit'] >= 0 else 'loss'
                        f.write(f"""
                            <tr>
                                <td>{tf}</td>
                                <td>{tf_data['trades']}</td>
                                <td>{tf_data['wins']}</td>
                                <td>{tf_data['losses']}</td>
                                <td>{tf_data['winrate']:.2f}%</td>
                                <td class="{profit_class}">{tf_data['profit']:.2f}</td>
                            </tr>
                        """)
                    
                    f.write("</table></div>")
                
                # Добавляем статистику по дням недели
                if stats['day_stats']:
                    f.write("""
                    <div class="section">
                        <h2>Статистика по дням недели</h2>
                        <table>
                            <tr>
                                <th>День</th>
                                <th>Сделки</th>
                                <th>Выигрыши</th>
                                <th>Проигрыши</th>
                                <th>Винрейт</th>
                                <th>Прибыль</th>
                            </tr>
                    """)
                    
                    # Сортируем дни недели
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    for day in days_order:
                        if day in stats['day_stats']:
                            day_data = stats['day_stats'][day]
                            profit_class = 'win' if day_data['profit'] >= 0 else 'loss'
                            f.write(f"""
                                <tr>
                                    <td>{day}</td>
                                    <td>{day_data['trades']}</td>
                                    <td>{day_data['wins']}</td>
                                    <td>{day_data['losses']}</td>
                                    <td>{day_data['winrate']:.2f}%</td>
                                    <td class="{profit_class}">{day_data['profit']:.2f}</td>
                                </tr>
                            """)
                    
                    f.write("</table></div>")
                
                # Добавляем статистику по месяцам
                if stats['month_stats']:
                    f.write("""
                    <div class="section">
                        <h2>Статистика по месяцам</h2>
                        <table>
                            <tr>
                                <th>Месяц</th>
                                <th>Сделки</th>
                                <th>Выигрыши</th>
                                <th>Проигрыши</th>
                                <th>Винрейт</th>
                                <th>Прибыль</th>
                            </tr>
                    """)
                    
                    # Сортируем месяцы
                    sorted_months = sorted(stats['month_stats'].keys())
                    
                    for month in sorted_months:
                        month_data = stats['month_stats'][month]
                        profit_class = 'win' if month_data['profit'] >= 0 else 'loss'
                        f.write(f"""
                            <tr>
                                <td>{month}</td>
                                <td>{month_data['trades']}</td>
                                <td>{month_data['wins']}</td>
                                <td>{month_data['losses']}</td>
                                <td>{month_data['winrate']:.2f}%</td>
                                <td class="{profit_class}">{month_data['profit']:.2f}</td>
                            </tr>
                        """)
                    
                    f.write("</table></div>")
                
                # Добавляем список последних сделок
                f.write("""
                <div class="section">
                    <h2>Последние сделки</h2>
                    <table>
                        <tr>
                            <th>Дата входа</th>
                            <th>Дата выхода</th>
                            <th>Тип</th>
                            <th>Лот</th>
                            <th>Цена входа</th>
                            <th>Цена выхода</th>
                            <th>Профит</th>
                            <th>Результат</th>
                            <th>Сетап</th>
                        </tr>
                """)
                
                # Сортируем сделки по времени выхода в обратном порядке
                sorted_trades = all_trades.sort_values(by='exit_time', ascending=False)
                
                # Показываем максимум 20 последних сделок
                for i, (_, trade) in enumerate(sorted_trades.iloc[:20].iterrows()):
                    entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M') if 'entry_time' in trade and pd.notna(trade['entry_time']) else '-'
                    exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M') if 'exit_time' in trade and pd.notna(trade['exit_time']) else '-'
                    
                    result_class = ''
                    if 'result' in trade:
                        result_class = 'win' if trade['result'] == 'win' else 'loss'
                    
                    profit_class = ''
                    if 'profit' in trade:
                        profit_class = 'win' if trade['profit'] >= 0 else 'loss'
                    
                    setup = trade['setup'] if 'setup' in trade and pd.notna(trade['setup']) else '-'
                    
                    f.write(f"""
                        <tr>
                            <td>{entry_time}</td>
                            <td>{exit_time}</td>
                            <td>{trade['order'] if 'order' in trade else '-'}</td>
                            <td>{trade['lot_size'] if 'lot_size' in trade else '-'}</td>
                            <td>{trade['entry_price'] if 'entry_price' in trade else '-'}</td>
                            <td>{trade['exit_price'] if 'exit_price' in trade else '-'}</td>
                            <td class="{profit_class}">{trade['profit'] if 'profit' in trade else '-'}</td>
                            <td class="{result_class}">{trade['result'] if 'result' in trade else '-'}</td>
                            <td>{setup}</td>
                        </tr>
                    """)
                
                f.write("</table></div>")
                
                # Ссылка на полную статистику в JSON
                f.write(f"""
                <div class="section">
                    <p>Полная статистика доступна в JSON файле: <a href="../{os.path.relpath(stats_file, self.journal_dir)}" target="_blank">Открыть статистику</a></p>
                </div>
                """)
                
                f.write("""
                </body>
                </html>
                """)
        
        except Exception as e:
            logger.error(f"Ошибка при создании HTML-отчета: {str(e)}")
            return None
        
        logger.info(f"Отчет о производительности создан: {report_file}")
        return report_file
    
    def export_to_csv(self, filename=None, filter_open=False, filter_closed=False, start_date=None, end_date=None):
        """
        Экспорт журнала сделок в CSV файл
        
        Параметры:
        filename (str, optional): Имя файла для экспорта
        filter_open (bool): Экспортировать только открытые сделки
        filter_closed (bool): Экспортировать только закрытые сделки
        start_date (datetime, optional): Начальная дата для фильтрации
        end_date (datetime, optional): Конечная дата для фильтрации
        
        Возвращает:
        str: Путь к экспортированному файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.symbol}_{self.account_id}_export_{timestamp}.csv"
        
        # Создаем директорию для экспорта, если она не существует
        export_dir = os.path.join(self.journal_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        file_path = os.path.join(export_dir, filename)
        
        # Фильтруем данные в соответствии с параметрами
        if filter_open:
            data_to_export = self.get_open_trades()
        elif filter_closed:
            data_to_export = self.get_closed_trades(start_date, end_date)
        else:
            # Объединяем открытые и закрытые сделки
            open_trades = self.get_open_trades()
            closed_trades = self.get_closed_trades(start_date, end_date)
            data_to_export = pd.concat([open_trades, closed_trades], ignore_index=True)
        
        if data_to_export.empty:
            logger.warning("Нет данных для экспорта")
            return None
        
        # Экспортируем в CSV
        data_to_export.to_csv(file_path, index=False)
        
        logger.info(f"Данные экспортированы в {file_path}")
        return file_path
    
    def export_to_mt5_format(self, filename=None):
        """
        Экспорт журнала сделок в формат, совместимый с MT5
        
        Параметры:
        filename (str, optional): Имя файла для экспорта
        
        Возвращает:
        str: Путь к экспортированному файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.symbol}_{self.account_id}_mt5export_{timestamp}.csv"
        
        # Создаем директорию для экспорта, если она не существует
        export_dir = os.path.join(self.journal_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        file_path = os.path.join(export_dir, filename)
        
        # Получаем только закрытые сделки
        closed_trades = self.get_closed_trades()
        
        if closed_trades.empty:
            logger.warning("Нет закрытых сделок для экспорта в формат MT5")
            return None
        
        # Создаем DataFrame в формате MT5
        mt5_columns = [
            "Ticket", "Open Time", "Type", "Size", "Item", "Price", "S / L", "T / P", 
            "Close Time", "Price", "Commission", "Swap", "Profit", "Comment"
        ]
        
        mt5_data = []
        
        for _, trade in closed_trades.iterrows():
            order_type = 0 if trade.get('order', '').lower() == 'buy' else 1  # 0 для buy, 1 для sell
            
            # Форматирование дат для MT5
            open_time = trade.get('entry_time', '')
            if isinstance(open_time, (datetime, pd.Timestamp)):
                open_time = open_time.strftime('%Y.%m.%d %H:%M:%S')
            
            close_time = trade.get('exit_time', '')
            if isinstance(close_time, (datetime, pd.Timestamp)):
                close_time = close_time.strftime('%Y.%m.%d %H:%M:%S')
            
            # Создаем запись в формате MT5
            row = [
                trade.get('ticket', ''),  # Ticket
                open_time,  # Open Time
                order_type,  # Type (0=buy, 1=sell)
                trade.get('lot_size', 0.0),  # Size
                trade.get('symbol', self.symbol),  # Item
                trade.get('entry_price', 0.0),  # Price
                trade.get('stop_loss', 0.0),  # S/L
                trade.get('take_profit', 0.0),  # T/P
                close_time,  # Close Time
                trade.get('exit_price', 0.0),  # Price
                trade.get('commission', 0.0),  # Commission
                trade.get('swap', 0.0),  # Swap
                trade.get('profit', 0.0),  # Profit
                trade.get('comment', '')  # Comment
            ]
            
            mt5_data.append(row)
        
        # Создаем DataFrame и экспортируем в CSV
        mt5_df = pd.DataFrame(mt5_data, columns=mt5_columns)
        mt5_df.to_csv(file_path, index=False, sep=',')
        
        logger.info(f"Данные экспортированы в формат MT5: {file_path}")
        return file_path
    
    def plot_cumulative_profit(self, save_path=None, show=False):
        """
        Построение графика накопленной прибыли
        
        Параметры:
        save_path (str, optional): Путь для сохранения графика
        show (bool): Показать график (в интерактивном режиме)
        
        Возвращает:
        str: Путь к сохраненному графику или None, если не удалось построить график
        """
        closed_trades = self.get_closed_trades()
        
        if closed_trades.empty or 'profit' not in closed_trades.columns:
            logger.warning("Нет данных для построения графика накопленной прибыли")
            return None
        
        try:
            # Сортируем сделки по времени выхода
            sorted_trades = closed_trades.sort_values(by='exit_time')
            
            # Рассчитываем накопленную прибыль
            sorted_trades['cumulative_profit'] = sorted_trades['profit'].cumsum()
            
            # Создаем график
            plt.figure(figsize=(12, 6))
            plt.plot(sorted_trades['exit_time'], sorted_trades['cumulative_profit'], 'b-', linewidth=2)
            plt.grid(True, alpha=0.3)
            plt.title(f'Накопленная прибыль для {self.symbol}')
            plt.xlabel('Время')
            plt.ylabel('Прибыль')
            
            # Форматируем ось X для отображения дат
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Добавляем маркеры для выигрышных и проигрышных сделок
            win_trades = sorted_trades[sorted_trades['result'] == 'win']
            loss_trades = sorted_trades[sorted_trades['result'] == 'loss']
            
            if not win_trades.empty:
                plt.scatter(win_trades['exit_time'], win_trades['cumulative_profit'], color='green', marker='^', s=50, label='Выигрыши')
            
            if not loss_trades.empty:
                plt.scatter(loss_trades['exit_time'], loss_trades['cumulative_profit'], color='red', marker='v', s=50, label='Проигрыши')
            
            # Добавляем линию нулевой прибыли
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Добавляем линию тренда
            if len(sorted_trades) > 1:
                x = np.arange(len(sorted_trades))
                y = sorted_trades['cumulative_profit'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(sorted_trades['exit_time'], p(x), "r--", alpha=0.7, label=f'Тренд: {z[0]:.2f}x + {z[1]:.2f}')
            
            plt.legend()
            plt.tight_layout()
            
            # Сохраняем график, если указан путь
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.charts_dir, f"{self.symbol}_cumulative_profit_{timestamp}.png")
            
            plt.savefig(save_path, dpi=100)
            
            # Показываем график, если требуется
            if show:
                plt.show()
            else:
                plt.close()
            
            logger.info(f"График накопленной прибыли сохранен: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Ошибка при построении графика накопленной прибыли: {str(e)}")
            if 'plt' in locals() or 'plt' in globals():
                plt.close()
            return None
    
    def plot_wins_vs_losses(self, save_path=None, show=False):
        """
        Построение сравнительных графиков выигрышей и проигрышей
        
        Параметры:
        save_path (str, optional): Путь для сохранения графика
        show (bool): Показать график (в интерактивном режиме)
        
        Возвращает:
        str: Путь к сохраненному графику или None, если не удалось построить график
        """
        closed_trades = self.get_closed_trades()
        
        if closed_trades.empty or 'result' not in closed_trades.columns:
            logger.warning("Нет данных для построения графика выигрышей/проигрышей")
            return None
        
        try:
            # Создаем фигуру с несколькими подграфиками
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Распределение по результатам (круговая диаграмма)
            result_counts = closed_trades['result'].value_counts()
            win_count = result_counts.get('win', 0)
            loss_count = result_counts.get('loss', 0)
            total = win_count + loss_count
            winrate = (win_count / total * 100) if total > 0 else 0
            
            axs[0, 0].pie([win_count, loss_count], labels=['Выигрыши', 'Проигрыши'], 
                          autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
            axs[0, 0].set_title(f'Распределение результатов (Винрейт: {winrate:.2f}%)')
            
            # 2. Распределение прибыли по выигрышам/проигрышам
            if 'profit' in closed_trades.columns:
                win_profits = closed_trades[closed_trades['result'] == 'win']['profit']
                loss_profits = closed_trades[closed_trades['result'] == 'loss']['profit']
                
                bin_count = min(20, max(5, int(total / 5)))  # Адаптивное количество бинов
                
                # Гистограмма прибыли для выигрышных сделок
                if not win_profits.empty:
                    axs[0, 1].hist(win_profits, bins=bin_count, color='green', alpha=0.7, label='Выигрыши')
                
                # Гистограмма прибыли для проигрышных сделок
                if not loss_profits.empty:
                    axs[0, 1].hist(loss_profits, bins=bin_count, color='red', alpha=0.7, label='Проигрыши')
                
                axs[0, 1].set_title('Распределение прибыли')
                axs[0, 1].set_xlabel('Прибыль')
                axs[0, 1].set_ylabel('Количество сделок')
                axs[0, 1].legend()
                axs[0, 1].grid(True, alpha=0.3)
            
            # 3. Средняя прибыль и убыток
            if 'profit' in closed_trades.columns:
                avg_win = win_profits.mean() if not win_profits.empty else 0
                avg_loss = loss_profits.mean() if not loss_profits.empty else 0
                
                bars = axs[1, 0].bar(['Средний выигрыш', 'Средний проигрыш'], [avg_win, avg_loss], 
                                   color=['green', 'red'], alpha=0.7)
                
                # Добавляем значения над столбцами
                for bar in bars:
                    height = bar.get_height()
                    if height < 0:
                        va = 'top'
                        y_offset = -5
                    else:
                        va = 'bottom'
                        y_offset = 5
                    
                    axs[1, 0].text(bar.get_x() + bar.get_width()/2., height + (y_offset * 0.01),
                                 f'{height:.2f}', ha='center', va=va)
                
                axs[1, 0].set_title('Средняя прибыль/убыток')
                axs[1, 0].set_ylabel('Прибыль')
                axs[1, 0].grid(True, alpha=0.3)
                
                # Добавляем линию нулевой прибыли
                axs[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # 4. Продолжительность сделок (выигрыши vs проигрыши)
            if 'duration' in closed_trades.columns:
                win_durations = closed_trades[closed_trades['result'] == 'win']['duration']
                loss_durations = closed_trades[closed_trades['result'] == 'loss']['duration']
                
                # Удаляем выбросы для лучшей визуализации
                if not win_durations.empty:
                    win_q1, win_q3 = np.percentile(win_durations, [25, 75])
                    win_iqr = win_q3 - win_q1
                    win_bounds = (win_q1 - 1.5 * win_iqr, win_q3 + 1.5 * win_iqr)
                    win_durations = win_durations[(win_durations >= win_bounds[0]) & (win_durations <= win_bounds[1])]
                
                if not loss_durations.empty:
                    loss_q1, loss_q3 = np.percentile(loss_durations, [25, 75])
                    loss_iqr = loss_q3 - loss_q1
                    loss_bounds = (loss_q1 - 1.5 * loss_iqr, loss_q3 + 1.5 * loss_iqr)
                    loss_durations = loss_durations[(loss_durations >= loss_bounds[0]) & (loss_durations <= loss_bounds[1])]
                
                bin_count = min(15, max(5, int(total / 8)))  # Адаптивное количество бинов
                
                if not win_durations.empty:
                    axs[1, 1].hist(win_durations, bins=bin_count, color='green', alpha=0.7, label='Выигрыши')
                
                if not loss_durations.empty:
                    axs[1, 1].hist(loss_durations, bins=bin_count, color='red', alpha=0.7, label='Проигрыши')
                
                axs[1, 1].set_title('Продолжительность сделок (часы)')
                axs[1, 1].set_xlabel('Продолжительность (часы)')
                axs[1, 1].set_ylabel('Количество сделок')
                axs[1, 1].legend()
                axs[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем график, если указан путь
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.charts_dir, f"{self.symbol}_wins_vs_losses_{timestamp}.png")
            
            plt.savefig(save_path, dpi=100)
            
            # Показываем график, если требуется
            if show:
                plt.show()
            else:
                plt.close()
            
            logger.info(f"График сравнения выигрышей и проигрышей сохранен: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Ошибка при построении графика выигрышей/проигрышей: {str(e)}")
            if 'plt' in locals() or 'plt' in globals():
                plt.close()
            return None
    
    def print_summary(self, days=None):
        """
        Вывод краткой сводки по торговле за последние дни
        
        Параметры:
        days (int, optional): Количество дней для анализа. Если None, использует все данные.
        """
        start_date = None
        if days is not None:
            start_date = datetime.now() - timedelta(days=days)
        
        # Получаем статистику
        stats = self.calculate_statistics(start_date=start_date)
        
        if stats['total_trades'] == 0:
            print(f"Нет данных о сделках{f' за последние {days} дней' if days else ''}.")
            return
        
        # Выводим краткую сводку
        print(f"\n=== Сводка по торговле {self.symbol}{f' за последние {days} дней' if days else ''} ===")
        print(f"Всего сделок: {stats['total_trades']}")
        print(f"Выигрыши/проигрыши: {stats['wins']}/{stats['losses']} (Винрейт: {stats['winrate']:.2f}%)")
        print(f"Общая прибыль: {stats['total_profit']:.2f}")
        print(f"Профит-фактор: {stats['profit_factor'] if stats['profit_factor'] != float('inf') else '∞'}")
        print(f"Средняя прибыль на сделку: {stats['total_profit'] / stats['total_trades']:.2f}")
        print(f"Макс. просадка: {stats['max_drawdown']:.2f} ({stats['max_drawdown_percent']:.2f}%)")
        
        # Выводим топ-3 лучших сетапа, если они есть
        if stats['setup_stats']:
            print("\nТоп-3 лучших сетапа:")
            top_setups = sorted(stats['setup_stats'].items(), key=lambda x: x[1]['profit'], reverse=True)[:3]
            for setup_name, setup_data in top_setups:
                print(f"- {setup_name}: {setup_data['profit']:.2f} ({setup_data['winrate']:.2f}% винрейт, {setup_data['trades']} сделок)")
        
        # Выводим информацию по типам ордеров
        if stats['order_stats']:
            print("\nПо типам ордеров:")
            for order_type, order_data in stats['order_stats'].items():
                print(f"- {order_type}: {order_data['profit']:.2f} ({order_data['winrate']:.2f}% винрейт, {order_data['trades']} сделок)")
        
        # Выводим лучший и худший день недели
        if stats['day_stats']:
            best_day = max(stats['day_stats'].items(), key=lambda x: x[1]['profit'])
            worst_day = min(stats['day_stats'].items(), key=lambda x: x[1]['profit'])
            
            print("\nЛучший/худший день недели:")
            print(f"- Лучший: {best_day[0]} с прибылью {best_day[1]['profit']:.2f}")
            print(f"- Худший: {worst_day[0]} с прибылью {worst_day[1]['profit']:.2f}")
        
        print("\nПоследние 5 сделок:")
        closed_trades = self.get_closed_trades(start_date=start_date)
        if not closed_trades.empty:
            recent_trades = closed_trades.sort_values(by='exit_time', ascending=False).head(5)
            for i, (_, trade) in enumerate(recent_trades.iterrows(), 1):
                exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M') if 'exit_time' in trade and pd.notna(trade['exit_time']) else '-'
                result = trade['result'] if 'result' in trade else '-'
                profit = trade['profit'] if 'profit' in trade else 0
                
                result_sign = '+' if result == 'win' else '-' if result == 'loss' else ''
                print(f"{i}. {exit_time}: {trade['order'] if 'order' in trade else '-'} {trade['symbol'] if 'symbol' in trade else self.symbol} {result_sign}{profit:.2f}")