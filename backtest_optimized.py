import pandas as pd
import numpy as np
import time
import logging
import os
import gc
from datetime import datetime, timedelta
from tqdm import tqdm
import multiprocessing
from functools import partial
from config import SYMBOL, BACKTEST_START, BACKTEST_END, TIMEFRAMES, INITIAL_BALANCE, RISK_PER_TRADE
from data_fetcher import get_historical_data
from strategy import find_trade_signal
from mt5_connector import connect_mt5, disconnect_mt5

# Директория для результатов
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
os.makedirs(results_dir, exist_ok=True)

# Импорт модуля визуализации с детальной обработкой ошибок
VISUALIZATION_AVAILABLE = False
try:
    # Пробуем импортировать необходимые библиотеки
    import PyQt5
    import pyqtgraph
    
    # Если базовые библиотеки импортированы успешно, пробуем импортировать визуализатор
    try:
        from backtest_visualizer import start_visualization, update_visualization, register_trade
        VISUALIZATION_AVAILABLE = True
        print("Модуль визуализации успешно импортирован")
    except ImportError as e:
        print(f"Не удалось импортировать модуль визуализации: {e}")
except ImportError as e:
    print(f"Не найдены библиотеки для визуализации: {e}")
    print("Установите PyQt5 и PyQtGraph для визуализации: pip install PyQt5 pyqtgraph")

# Глобальная переменная для управления частотой обновления визуализации
VISUALIZATION_UPDATE_FREQUENCY = 10  # Обновление каждые 10 шагов

# Директория для логов
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Имя файла логов с датой
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename = os.path.join(log_dir, f"{current_date}_backtest_log.txt")

# Настройка уровня логирования
log_level = logging.INFO

# Настраиваем логирование в файл и в консоль
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def preprocess_data(start_date=None, end_date=None, symbol=None, timeframes=None):
    """
    Предварительная загрузка и обработка всех данных для бэктеста
    
    Параметры:
    start_date (datetime): Дата начала бэктеста. Если None, используется BACKTEST_START из конфига
    end_date (datetime): Дата окончания бэктеста. Если None, используется BACKTEST_END из конфига
    symbol (str): Символ для бэктеста. Если None, используется SYMBOL из конфига
    timeframes (list): Список таймфреймов. Если None, используется TIMEFRAMES из конфига
    
    Возвращает:
    tuple: (timeframe_data, timeline, candle_indices) или None в случае ошибки
    """
    start_time = time.time()
    
    # Используем значения по умолчанию из конфига, если параметры не заданы
    if start_date is None:
        start_date = BACKTEST_START
    if end_date is None:
        end_date = BACKTEST_END
    if symbol is None:
        symbol = SYMBOL
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    if not connect_mt5():
        logging.error("Ошибка подключения к MT5 при загрузке данных")
        return None
    
    try:
        logging.info(f"Загрузка исторических данных для {symbol} ({', '.join(timeframes)})")
        print(f"Загрузка исторических данных для {symbol}...")
        
        # Загружаем данные для всех таймфреймов
        timeframe_data = {}
        candle_indices = {}  # Словарь для быстрого поиска индексов
        
        for tf in timeframes:
            print(f"  Загрузка данных для {tf}...")
            df = get_historical_data(symbol, timeframe=tf, start_date=start_date, end_date=end_date)
            if df is None or len(df) < 20:
                logging.warning(f"Недостаточно данных для {tf}")
                continue
                
            # Оптимизируем DataFrame для быстрого доступа
            df['time'] = pd.to_datetime(df['time'])
            
            # Создаем индекс для быстрого поиска по времени
            candle_indices[tf] = {}
            for idx, row in df.iterrows():
                candle_indices[tf][row['time']] = idx
            
            # Сохраняем данные
            timeframe_data[tf] = df
            print(f"  Загружено {len(df)} свечей для {tf}")
        
        # Создаем временную шкалу для бэктеста
        smallest_tf = min(timeframes, key=lambda x: timeframes.index(x))
        if smallest_tf in timeframe_data:
            timeline = timeframe_data[smallest_tf]['time'].tolist()
            
            # Отбрасываем первые 100 свечей для наличия исторических данных для анализа
            valid_timeline = timeline[100:]
            logging.info(f"Создана временная шкала из {len(valid_timeline)} точек")
            
            elapsed_time = time.time() - start_time
            logging.info(f"Предобработка данных завершена за {elapsed_time:.2f} секунд")
            
            return timeframe_data, valid_timeline, candle_indices
        else:
            logging.error("Не удалось создать временную шкалу для бэктеста")
            return None
            
    finally:
        disconnect_mt5()  # Отключаемся от MT5 после загрузки всех данных

def find_candle_at_time(timeframe_data, candle_indices, tf, current_time):
    """
    Быстрый поиск свечи по времени с использованием индекса
    
    Параметры:
    timeframe_data (dict): Данные по всем таймфреймам
    candle_indices (dict): Индексы свечей по времени
    tf (str): Таймфрейм
    current_time (datetime): Искомое время
    
    Возвращает:
    pandas.Series: Найденная свеча или None
    """
    if tf not in timeframe_data or tf not in candle_indices:
        return None
    
    # Пробуем точное совпадение по времени
    if current_time in candle_indices[tf]:
        idx = candle_indices[tf][current_time]
        if isinstance(idx, int):  # Убедиться, что idx - целое число
            return timeframe_data[tf].iloc[idx]
        else:
            # Если idx не целое число, пробуем конвертировать
            try:
                idx = int(idx)
                return timeframe_data[tf].iloc[idx]
            except:
                return None
    
    # Ищем ближайшую предыдущую свечу
    previous_times = [t for t in candle_indices[tf].keys() if t <= current_time]
    if previous_times:
        closest_time = max(previous_times)
        idx = candle_indices[tf][closest_time]
        return timeframe_data[tf].iloc[idx]
    
    return None

def sequential_backtest(timeline, timeframe_data, candle_indices, visualize_realtime=False,
                       initial_balance=None, risk_per_trade=None, symbol=None, debug_mode=False,
                       visualization_mode="full"):
    """
    Выполнение последовательного бэктеста
    
    Параметры:
    timeline (list): Временная шкала для бэктеста
    timeframe_data (dict): Данные по всем таймфреймам
    candle_indices (dict): Индексы свечей по времени
    visualize_realtime (bool): Использовать ли визуализацию в реальном времени
    initial_balance (float): Начальный баланс. Если None, используется INITIAL_BALANCE из конфига
    risk_per_trade (float): Риск на сделку. Если None, используется RISK_PER_TRADE из конфига
    symbol (str): Символ для бэктеста. Если None, используется SYMBOL из конфига
    debug_mode (bool): Режим отладки с дополнительными сообщениями
    visualization_mode (str): Режим визуализации ("full", "simple", "performance")
    
    Возвращает:
    tuple: (результаты, история сделок)
    """
    # Используем значения по умолчанию из конфига, если параметры не заданы
    if initial_balance is None:
        initial_balance = INITIAL_BALANCE
    if risk_per_trade is None:
        risk_per_trade = RISK_PER_TRADE
    if symbol is None:
        symbol = SYMBOL
    
    results = []
    balance = initial_balance
    open_trades = []  # Текущие открытые сделки
    last_signal_time = {}  # Отслеживание последних сигналов по таймфреймам
    trade_history = {
        "entries": [],
        "exits": [],
        "stop_losses": [],
        "take_profits": []
    }
    
    # Инициализация визуализации, если она включена
    visualizer = None
    if visualize_realtime and VISUALIZATION_AVAILABLE:
        try:
            visualizer = start_visualization(timeframe_data, symbol, mode=visualization_mode)
            if debug_mode:
                print("Визуализация инициализирована успешно")
        except Exception as e:
            if debug_mode:
                print(f"Ошибка при инициализации визуализации: {e}")
            visualize_realtime = False
    
    # Определяем наименьший таймфрейм
    tf_keys = list(timeframe_data.keys())
    smallest_tf = min(tf_keys, key=lambda x: TIMEFRAMES.index(x) if x in TIMEFRAMES else float('inf'))
    
    if debug_mode:
        print(f"Используем наименьший таймфрейм: {smallest_tf}")
        print(f"Начинаем бэктест для {len(timeline)} точек времени")
    
    # Для отслеживания уникальных входов
    unique_entries = set()
    
    # Счетчик для обновления визуализации
    visualization_counter = 0
    
    # Кэш последних данных о свечах для ускорения доступа
    candle_cache = {}
    
    # Инициализируем прогресс-бар
    with tqdm(total=len(timeline), desc="Бэктест", unit="бар") as pbar:
        # Обрабатываем каждую точку временной шкалы
        for i, current_time in enumerate(timeline):
            # Обновляем прогресс-бар каждые 100 точек для производительности
            if i % 100 == 0:
                pbar.update(100 if i > 0 else 1)
                if debug_mode:
                    print(f"Обработано {i} точек из {len(timeline)}")
            
            # Получаем текущую свечу с использованием кэша
            cache_key = f"{smallest_tf}_{current_time}"
            if cache_key in candle_cache:
                current_candle = candle_cache[cache_key]
            else:
                current_candle = find_candle_at_time(timeframe_data, candle_indices, smallest_tf, current_time)
                candle_cache[cache_key] = current_candle
                
                # Ограничиваем размер кэша для экономии памяти
                if len(candle_cache) > 1000:
                    # Удаляем самые старые записи
                    old_keys = list(candle_cache.keys())[:100]
                    for key in old_keys:
                        del candle_cache[key]
            
            if current_candle is None:
                continue
            
            # Обновляем визуализацию, если она включена
            if visualize_realtime and visualizer:
                visualization_counter += 1
                if visualization_counter % VISUALIZATION_UPDATE_FREQUENCY == 0:
                    try:
                        update_visualization(visualizer, current_time, balance, open_trades)
                    except Exception as e:
                        if debug_mode:
                            print(f"Ошибка при обновлении визуализации: {e}")
            
            # Проверяем и обновляем открытые сделки
            trade_closed = False
            for trade_idx in reversed(range(len(open_trades))):
                trade = open_trades[trade_idx]
                
                # Проверка стоп-лосса и тейк-профита
                if trade['order'] == "buy":
                    # Проверка стоп-лосса
                    if current_candle['low'] <= trade['entry_price'] - trade['stop_loss']:
                        # Стоп-лосс сработал
                        trade['result'] = "loss"
                        trade['exit_price'] = trade['entry_price'] - trade['stop_loss']
                        trade['exit_time'] = current_time
                        trade['profit'] = -trade['risk_amount']
                        balance += trade['profit']
                        trade_closed = True
                        
                        # Добавляем в историю для визуализации
                        exit_data = {
                            "time": trade['exit_time'],
                            "price": trade['exit_price'],
                            "type": "stop_loss",
                            "order": trade['order'],
                            "profit": trade['profit']
                        }
                        trade_history["exits"].append(exit_data)
                        
                        # Обновляем визуализацию при закрытии сделки
                        if visualize_realtime and visualizer:
                            try:
                                register_trade(visualizer, 'exit', exit_data)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Ошибка при обновлении визуализации: {e}")
                        
                        # Добавляем результат в историю
                        results.append({
                            "entry_time": trade['entry_time'],
                            "order": trade['order'],
                            "tf": trade['tf'],
                            "result": trade['result'],
                            "profit": trade['profit'],
                            "lot_size": trade['lot_size'],
                            "entry_price": trade['entry_price'],
                            "exit_price": trade['exit_price'],
                            "exit_time": trade['exit_time'],
                            "stop_loss": trade['stop_loss'],
                            "take_profit": trade['take_profit'],
                            "balance": balance,
                            "setup": trade.get('setup', 'Standard')
                        })
                        
                        # Удаляем сделку из открытых
                        open_trades.pop(trade_idx)
                        
                        if debug_mode:
                            print(f"SL сработал для BUY: {current_time}, цена: {trade['exit_price']}")
                        
                    # Проверка тейк-профита
                    elif current_candle['high'] >= trade['entry_price'] + trade['take_profit']:
                        # Тейк-профит сработал
                        trade['result'] = "win"
                        trade['exit_price'] = trade['entry_price'] + trade['take_profit']
                        trade['exit_time'] = current_time
                        trade['profit'] = trade['risk_amount'] * (trade['take_profit'] / trade['stop_loss'])
                        balance += trade['profit']
                        trade_closed = True
                        
                        # Добавляем в историю для визуализации
                        exit_data = {
                            "time": trade['exit_time'],
                            "price": trade['exit_price'],
                            "type": "take_profit",
                            "order": trade['order'],
                            "profit": trade['profit']
                        }
                        trade_history["exits"].append(exit_data)
                        
                        # Обновляем визуализацию при закрытии сделки
                        if visualize_realtime and visualizer:
                            try:
                                register_trade(visualizer, 'exit', exit_data)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Ошибка при обновлении визуализации: {e}")
                        
                        # Добавляем результат в историю
                        results.append({
                            "entry_time": trade['entry_time'],
                            "order": trade['order'],
                            "tf": trade['tf'],
                            "result": trade['result'],
                            "profit": trade['profit'],
                            "lot_size": trade['lot_size'],
                            "entry_price": trade['entry_price'],
                            "exit_price": trade['exit_price'],
                            "exit_time": trade['exit_time'],
                            "stop_loss": trade['stop_loss'],
                            "take_profit": trade['take_profit'],
                            "balance": balance,
                            "setup": trade.get('setup', 'Standard')
                        })
                        
                        # Удаляем сделку из открытых
                        open_trades.pop(trade_idx)
                        
                        if debug_mode:
                            print(f"TP сработал для BUY: {current_time}, цена: {trade['exit_price']}")
                        
                else:  # sell
                    # Проверка стоп-лосса
                    if current_candle['high'] >= trade['entry_price'] + trade['stop_loss']:
                        # Стоп-лосс сработал
                        trade['result'] = "loss"
                        trade['exit_price'] = trade['entry_price'] + trade['stop_loss']
                        trade['exit_time'] = current_time
                        trade['profit'] = -trade['risk_amount']
                        balance += trade['profit']
                        trade_closed = True
                        
                        # Добавляем в историю для визуализации
                        exit_data = {
                            "time": trade['exit_time'],
                            "price": trade['exit_price'],
                            "type": "stop_loss",
                            "order": trade['order'],
                            "profit": trade['profit']
                        }
                        trade_history["exits"].append(exit_data)
                        
                        # Обновляем визуализацию при закрытии сделки
                        if visualize_realtime and visualizer:
                            try:
                                register_trade(visualizer, 'exit', exit_data)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Ошибка при обновлении визуализации: {e}")
                        
                        # Добавляем результат в историю
                        results.append({
                            "entry_time": trade['entry_time'],
                            "order": trade['order'],
                            "tf": trade['tf'],
                            "result": trade['result'],
                            "profit": trade['profit'],
                            "lot_size": trade['lot_size'],
                            "entry_price": trade['entry_price'],
                            "exit_price": trade['exit_price'],
                            "exit_time": trade['exit_time'],
                            "stop_loss": trade['stop_loss'],
                            "take_profit": trade['take_profit'],
                            "balance": balance,
                            "setup": trade.get('setup', 'Standard')
                        })
                        
                        # Удаляем сделку из открытых
                        open_trades.pop(trade_idx)
                        
                        if debug_mode:
                            print(f"SL сработал для SELL: {current_time}, цена: {trade['exit_price']}")
                        
                    # Проверка тейк-профита
                    elif current_candle['low'] <= trade['entry_price'] - trade['take_profit']:
                        # Тейк-профит сработал
                        trade['result'] = "win"
                        trade['exit_price'] = trade['entry_price'] - trade['take_profit']
                        trade['exit_time'] = current_time
                        trade['profit'] = trade['risk_amount'] * (trade['take_profit'] / trade['stop_loss'])
                        balance += trade['profit']
                        trade_closed = True
                        
                        # Добавляем в историю для визуализации
                        exit_data = {
                            "time": trade['exit_time'],
                            "price": trade['exit_price'],
                            "type": "take_profit",
                            "order": trade['order'],
                            "profit": trade['profit']
                        }
                        trade_history["exits"].append(exit_data)
                        
                        # Обновляем визуализацию при закрытии сделки
                        if visualize_realtime and visualizer:
                            try:
                                register_trade(visualizer, 'exit', exit_data)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Ошибка при обновлении визуализации: {e}")
                        
                        # Добавляем результат в историю
                        results.append({
                            "entry_time": trade['entry_time'],
                            "order": trade['order'],
                            "tf": trade['tf'],
                            "result": trade['result'],
                            "profit": trade['profit'],
                            "lot_size": trade['lot_size'],
                            "entry_price": trade['entry_price'],
                            "exit_price": trade['exit_price'],
                            "exit_time": trade['exit_time'],
                            "stop_loss": trade['stop_loss'],
                            "take_profit": trade['take_profit'],
                            "balance": balance,
                            "setup": trade.get('setup', 'Standard')
                        })
                        
                        # Удаляем сделку из открытых
                        open_trades.pop(trade_idx)
                        
                        if debug_mode:
                            print(f"TP сработал для SELL: {current_time}, цена: {trade['exit_price']}")
            
            # Если нет открытой сделки, ищем новый сигнал
            if not open_trades:
                # Получаем список таймфреймов, сортированный по уровню (от старшего к младшему)
                sorted_timeframes = sorted(
                    [tf for tf in timeframe_data.keys()],
                    key=lambda x: -TIMEFRAMES.index(x) if x in TIMEFRAMES else -999
                )
                
                # Проверяем сигналы на всех таймфреймах (от старшего к младшему)
                for tf in sorted_timeframes:
                    # Пропускаем проверку, если еще не прошло достаточно времени с предыдущего сигнала на этом таймфрейме
                    if tf in last_signal_time and current_time < last_signal_time[tf]:
                        continue
                    
                    # Получаем данные для текущего таймфрейма до текущего момента времени
                    filtered_df = timeframe_data[tf][timeframe_data[tf]['time'] <= current_time].copy()
                    
                    # Пропускаем, если недостаточно данных для анализа
                    if len(filtered_df) < 30:
                        continue
                        
                    # Ищем сигнал на основе данных до текущего момента
                    signal = find_trade_signal(filtered_df)
                    
                    # Если сигнал найден и нет открытой сделки
                    if signal:
                        # Обновляем время последнего сигнала для этого таймфрейма
                        last_signal_time[tf] = current_time + timedelta(hours=1)  # Задержка 1 час перед следующим сигналом
                        
                        # Проверяем, что сигнал подтверждается и на младшем таймфрейме, если это старший ТФ
                        signal_confirmed = True
                        if tf in ["D1", "H4", "H1"]:
                            confirmation_tf = "M15"  # Используем M15 для подтверждения
                            if confirmation_tf in timeframe_data:
                                confirm_df = timeframe_data[confirmation_tf][timeframe_data[confirmation_tf]['time'] <= current_time].copy()
                                if len(confirm_df) >= 30:
                                    confirm_signal = find_trade_signal(confirm_df)
                                    if not confirm_signal or confirm_signal['type'] != signal['type']:
                                        signal_confirmed = False
                        
                        if not signal_confirmed:
                            continue
                        
                        # Получаем стоп-лосс и тейк-профит из сигнала
                        stop_loss = signal.get('stop_loss', 0.0003)  # Дефолтное значение 30 пипсов
                        take_profit = signal.get('take_profit', stop_loss * 3)  # Дефолтное соотношение 1:3
                        
                        # Ограничиваем риск максимум 1% от баланса
                        actual_risk_per_trade = min(risk_per_trade, 0.01)
                        
                        # Рассчитываем размер позиции на основе риска
                        risk_amount = balance * actual_risk_per_trade  # Риск в валюте счета
                        stop_loss_pips = int(stop_loss / 0.0001)  # Конвертируем в пипсы
                        
                        # Ограничиваем максимальный размер лота
                        max_lot = 0.5  # Максимальный размер лота для любой сделки
                        
                        # Рассчитываем размер лота
                        lot_size = risk_amount / (stop_loss_pips * 10)
                        lot_size = round(lot_size, 2)
                        
                        # Применяем ограничения
                        lot_size = min(max(lot_size, 0.01), max_lot)
                        
                        # Проверяем, не слишком ли маленький стоп-лосс (менее 10 пипсов)
                        if stop_loss_pips < 10:
                            if debug_mode:
                                print(f"Пропуск сигнала из-за слишком маленького SL: {stop_loss_pips} пипсов")
                            continue
                        
                        # Формируем уникальный ключ для проверки дублирования
                        entry_key = f"{current_time}_{tf}_{signal['type']}"
                        
                        # Пропускаем, если уже была такая сделка
                        if entry_key in unique_entries:
                            if debug_mode:
                                print(f"Пропуск дублирующегося сигнала: {entry_key}")
                            continue
                        
                        # Добавляем ключ в множество
                        unique_entries.add(entry_key)
                        
                        if debug_mode:
                            print(f"Найден сигнал: {signal['type']} на {tf}, цена: {signal['level']}, setup: {signal.get('setup', 'Standard')}")
                        
                        # Добавляем информацию о входе в историю для визуализации
                        entry_data = {
                            "time": current_time,
                            "price": signal['level'],
                            "order": signal['type'],
                            "tf": tf,
                            "setup": signal.get('setup', 'Standard')
                        }
                        trade_history["entries"].append(entry_data)
                        
                        # Добавляем информацию о стоп-лоссе и тейк-профите
                        if signal['type'] == "buy":
                            sl_price = signal['level'] - stop_loss
                            tp_price = signal['level'] + take_profit
                        else:  # sell
                            sl_price = signal['level'] + stop_loss
                            tp_price = signal['level'] - take_profit
                        
                        sl_data = {
                            "time": current_time,
                            "price": sl_price,
                            "order": signal['type']
                        }
                        trade_history["stop_losses"].append(sl_data)
                        
                        tp_data = {
                            "time": current_time,
                            "price": tp_price,
                            "order": signal['type']
                        }
                        trade_history["take_profits"].append(tp_data)
                        
                        # Обновляем визуализацию при открытии сделки
                        if visualize_realtime and visualizer:
                            try:
                                register_trade(visualizer, 'entry', entry_data)
                                register_trade(visualizer, 'sl', sl_data)
                                register_trade(visualizer, 'tp', tp_data)
                            except Exception as e:
                                if debug_mode:
                                    print(f"Ошибка при обновлении визуализации при открытии сделки: {e}")
                        
                        # Создаем новую сделку
                        new_trade = {
                            "order": signal['type'],
                            "entry_price": signal['level'],
                            "entry_time": current_time,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "lot_size": lot_size,
                            "tf": tf,
                            "risk_amount": risk_amount,
                            "setup": signal.get('setup', 'Standard')
                        }
                        
                        # Добавляем сделку в список открытых
                        open_trades.append(new_trade)
                        
                        # Нашли сигнал и открыли сделку, прекращаем проверку таймфреймов
                        break
    
    # Закрываем оставшиеся открытые сделки в конце периода
    final_time = timeline[-1] if timeline else None
    for trade in open_trades:
        # Получаем последнюю цену
        final_price = None
        for tf in [smallest_tf, "M15", "H1", "H4", "D1"]:
            if tf in timeframe_data and not timeframe_data[tf].empty:
                final_price = timeframe_data[tf].iloc[-1]['close']
                break
        
        if final_price is None:
            logging.warning("Не удалось получить финальную цену для закрытия открытых сделок")
            continue
        
        # Рассчитываем результат
        if trade['order'] == "buy":
            profit = (final_price - trade['entry_price']) / trade['stop_loss'] * trade['risk_amount']
            result = "win" if final_price > trade['entry_price'] else "loss"
        else:  # sell
            profit = (trade['entry_price'] - final_price) / trade['stop_loss'] * trade['risk_amount']
            result = "win" if final_price < trade['entry_price'] else "loss"
        
        balance += profit
        
        # Добавляем в историю для визуализации
        exit_data = {
            "time": final_time,
            "price": final_price,
            "type": "end_of_test",
            "order": trade['order'],
            "profit": profit
        }
        trade_history["exits"].append(exit_data)
        
        # Обновляем визуализацию при закрытии сделки
        if visualize_realtime and visualizer:
            try:
                register_trade(visualizer, 'exit', exit_data)
            except Exception as e:
                if debug_mode:
                    print(f"Ошибка при обновлении визуализации: {e}")
        
        # Добавляем в результаты
        results.append({
            "entry_time": trade['entry_time'],
            "order": trade['order'],
            "tf": trade['tf'],
            "result": result,
            "profit": profit,
            "lot_size": trade['lot_size'],
            "entry_price": trade['entry_price'],
            "exit_price": final_price,
            "exit_time": final_time,
            "stop_loss": trade['stop_loss'],
            "take_profit": trade['take_profit'],
            "balance": balance,
            "setup": trade.get('setup', 'Standard')
        })
        
        if debug_mode:
            print(f"Закрыта последняя сделка в конце периода: {trade['order']} с результатом {result}")
    
    if debug_mode:
        print(f"Бэктест завершен. Всего сделок: {len(results)}")
    
    return results, trade_history

def process_chunk_with_params(timeline, timeframe_data, candle_indices, 
                             chunk_start_idx, chunk_end_idx, 
                             initial_balance=None, risk_per_trade=None, symbol=None, 
                             debug_mode=False, chunk_index=0):
    """
    Обработка части временной шкалы для параллельного бэктеста
    
    Параметры:
    timeline (list): Полная временная шкала
    timeframe_data (dict): Данные по всем таймфреймам
    candle_indices (dict): Индексы свечей по времени
    chunk_start_idx (int): Начальный индекс участка
    chunk_end_idx (int): Конечный индекс участка
    initial_balance (float): Начальный баланс
    risk_per_trade (float): Риск на сделку
    symbol (str): Символ для бэктеста
    debug_mode (bool): Режим отладки
    chunk_index (int): Индекс чанка для отслеживания порядка
    
    Возвращает:
    tuple: (результаты, история сделок, chunk_index)
    """
    try:
        # Проверяем типы индексов
        if not isinstance(chunk_start_idx, int):
            logging.error(f"Ошибка в process_chunk_with_params для чанка {chunk_index}: chunk_start_idx не является целым числом (тип: {type(chunk_start_idx).__name__}, значение: {chunk_start_idx})")
            # Пытаемся преобразовать к int
            try:
                chunk_start_idx = int(chunk_start_idx)
            except:
                return [], {"entries": [], "exits": [], "stop_losses": [], "take_profits": []}, chunk_index
        
        if not isinstance(chunk_end_idx, int):
            logging.error(f"Ошибка в process_chunk_with_params для чанка {chunk_index}: chunk_end_idx не является целым числом (тип: {type(chunk_end_idx).__name__}, значение: {chunk_end_idx})")
            # Пытаемся преобразовать к int
            try:
                chunk_end_idx = int(chunk_end_idx)
            except:
                return [], {"entries": [], "exits": [], "stop_losses": [], "take_profits": []}, chunk_index
        
        # Выделяем чанк из временной шкалы
        chunk = timeline[chunk_start_idx:chunk_end_idx]
        
        if debug_mode:
            logging.info(f"Начата обработка чанка {chunk_index}: {chunk_start_idx}-{chunk_end_idx} ({len(chunk)} точек)")
        
        # Выполняем последовательный бэктест для чанка
        results, trade_history = sequential_backtest(
            chunk, timeframe_data, candle_indices, 
            visualize_realtime=False,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            symbol=symbol,
            debug_mode=debug_mode
        )
        
        # Добавляем к результатам информацию об индексе чанка
        for result in results:
            result['chunk_index'] = chunk_index
            
        if debug_mode:
            logging.info(f"Завершена обработка чанка {chunk_index}: найдено {len(results)} сделок")
        
        # Возвращаем результаты и историю сделок для этого чанка, а также индекс чанка
        return results, trade_history, chunk_index
    except Exception as e:
        logging.error(f"Ошибка в process_chunk_with_params для чанка {chunk_index}: {str(e)}")
        if debug_mode:
            import traceback
            logging.error(traceback.format_exc())
        # Возвращаем пустые результаты в случае ошибки
        return [], {"entries": [], "exits": [], "stop_losses": [], "take_profits": []}, chunk_index

def parallel_backtest(timeline, timeframe_data, candle_indices, num_processes, 
                     initial_balance=None, risk_per_trade=None, symbol=None, debug_mode=False):
    """
    Параллельное выполнение бэктеста на нескольких процессах с улучшенной обработкой ошибок
    и корректным объединением результатов
    
    Параметры:
    timeline (list): Временная шкала для бэктеста
    timeframe_data (dict): Данные по всем таймфреймам
    candle_indices (dict): Индексы свечей по времени
    num_processes (int): Количество процессов
    initial_balance (float): Начальный баланс. Если None, используется INITIAL_BALANCE из конфига
    risk_per_trade (float): Риск на сделку. Если None, используется RISK_PER_TRADE из конфига
    symbol (str): Символ для бэктеста. Если None, используется SYMBOL из конфига
    debug_mode (bool): Режим отладки с дополнительными выводами
    
    Возвращает:
    tuple: (результаты, история сделок)
    """
    start_time = time.time()
    
    # Используем значения по умолчанию из конфига, если параметры не заданы
    if initial_balance is None:
        initial_balance = INITIAL_BALANCE
    if risk_per_trade is None:
        risk_per_trade = RISK_PER_TRADE
    if symbol is None:
        symbol = SYMBOL
    
    # Регулируем число процессов в зависимости от размера данных
    timeline_length = len(timeline)
    if timeline_length < num_processes * 100:  # Если данных мало, снижаем число процессов
        recommended_processes = max(1, timeline_length // 100)
        logging.warning(f"Слишком мало данных для {num_processes} процессов. Используем {recommended_processes} процессов.")
        num_processes = recommended_processes
    
    # Определяем оптимальные границы чанков, учитывая временные интервалы
    # Стараемся, чтобы границы чанков не разрывали торговые дни
    try:
        # Простое равномерное разделение
        chunk_size = timeline_length // num_processes
        boundaries = [(i*chunk_size, min((i+1)*chunk_size, timeline_length)) 
                     for i in range(num_processes)]
        
        # Дополнительно логируем информацию о границах для отладки
        if debug_mode:
            for i, (start, end) in enumerate(boundaries):
                logging.info(f"Чанк {i}: индексы {start}-{end} (размер: {end-start})")
    except Exception as e:
        logging.error(f"Ошибка при определении границ чанков: {str(e)}")
        if debug_mode:
            import traceback
            logging.error(traceback.format_exc())
        # Если не удалось определить границы, используем равномерное разделение
        chunk_size = timeline_length // num_processes
        boundaries = [(i*chunk_size, min((i+1)*chunk_size, timeline_length)) 
                      for i in range(num_processes)]
    
    logging.info(f"Запуск параллельной обработки на {num_processes} процессах...")
    
    # Создаем аргументы для каждого процесса
    starmap_args = []
    for i, (start, end) in enumerate(boundaries):
        args = (timeline, timeframe_data, candle_indices, start, end, 
                initial_balance, risk_per_trade, symbol, debug_mode, i)
        starmap_args.append(args)
    
    # Запускаем параллельную обработку с обработкой ошибок
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk_with_params, starmap_args)
    except Exception as e:
        logging.error(f"Ошибка при параллельной обработке: {str(e)}")
        if debug_mode:
            import traceback
            logging.error(traceback.format_exc())
        # В случае ошибки пытаемся выполнить последовательный бэктест
        logging.info("Пробуем выполнить последовательный бэктест...")
        results_seq, history_seq = sequential_backtest(
            timeline, timeframe_data, candle_indices, 
            visualize_realtime=False,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            symbol=symbol,
            debug_mode=debug_mode
        )
        return results_seq, history_seq
    
    # Объединяем результаты с их сортировкой и учетом индекса чанка
    all_results = []
    combined_history = {
        "entries": [],
        "exits": [],
        "stop_losses": [],
        "take_profits": []
    }
    
    logging.info("Объединение результатов параллельных процессов...")
    for chunk_results, chunk_history, chunk_index in results:
        all_results.extend(chunk_results)
        combined_history["entries"].extend(chunk_history["entries"])
        combined_history["exits"].extend(chunk_history["exits"])
        combined_history["stop_losses"].extend(chunk_history["stop_losses"])
        combined_history["take_profits"].extend(chunk_history["take_profits"])
    
    # Сортируем объединенные результаты по времени входа и индексу чанка
    if all_results:  # Проверяем, что список не пустой
        if 'entry_time' in all_results[0]:
            all_results = sorted(all_results, key=lambda x: (x["entry_time"], x.get("chunk_index", 0)))
        
        # Пересчитываем баланс последовательно
        balance = initial_balance
        for i, result in enumerate(all_results):
            # Удаляем временный индекс чанка перед возвратом результата
            if 'chunk_index' in result:
                del result['chunk_index']
                
            balance += result["profit"]
            all_results[i]["balance"] = balance
    
    # Сортируем истории сделок по времени
    if combined_history["entries"]:
        if 'time' in combined_history["entries"][0]:
            combined_history["entries"] = sorted(combined_history["entries"], key=lambda x: x["time"])
    
    if combined_history["exits"]:
        if 'time' in combined_history["exits"][0]:
            combined_history["exits"] = sorted(combined_history["exits"], key=lambda x: x["time"])
    
    if combined_history["stop_losses"]:
        if 'time' in combined_history["stop_losses"][0]:
            combined_history["stop_losses"] = sorted(combined_history["stop_losses"], key=lambda x: x["time"])
    
    if combined_history["take_profits"]:
        if 'time' in combined_history["take_profits"][0]:
            combined_history["take_profits"] = sorted(combined_history["take_profits"], key=lambda x: x["time"])
    
    elapsed_time = time.time() - start_time
    logging.info(f"Параллельный бэктест завершен за {elapsed_time:.2f} секунд. Всего сделок: {len(all_results)}")
    return all_results, combined_history

def optimized_backtest(parallel=True, num_processes=None, visualize_realtime=False,
                      start_date=None, end_date=None, symbol=None, timeframes=None,
                      initial_balance=None, risk_per_trade=None, debug_mode=False,
                      visualization_mode="full", save_charts=True):
    """
    Выполнение оптимизированного бэктеста с возможностью параллельной обработки.
    
    Параметры:
    parallel (bool): Использовать ли параллельную обработку
    num_processes (int): Количество процессов для параллельной обработки
    visualize_realtime (bool): Использовать ли визуализацию в реальном времени
    start_date (datetime): Дата начала бэктеста. Если None, используется BACKTEST_START из конфига
    end_date (datetime): Дата окончания бэктеста. Если None, используется BACKTEST_END из конфига
    symbol (str): Символ для бэктеста. Если None, используется SYMBOL из конфига
    timeframes (list): Список таймфреймов. Если None, используется TIMEFRAMES из конфига
    initial_balance (float): Начальный баланс. Если None, используется INITIAL_BALANCE из конфига
    risk_per_trade (float): Риск на сделку. Если None, используется RISK_PER_TRADE из конфига
    debug_mode (bool): Режим отладки с дополнительными выводами
    visualization_mode (str): Режим визуализации ("full", "simple", "performance")
    save_charts (bool): Сохранять ли графики после бэктеста
    
    Возвращает:
    DataFrame: Результаты бэктеста
    """
    # Если используется визуализация, отключаем параллельную обработку
    if visualize_realtime and VISUALIZATION_AVAILABLE:
        if parallel:
            print("Внимание: При использовании визуализации в реальном времени параллельная обработка отключена")
            parallel = False
    
    # Используем значения по умолчанию из конфига, если параметры не заданы
    if start_date is None:
        start_date = BACKTEST_START
    if end_date is None:
        end_date = BACKTEST_END
    if symbol is None:
        symbol = SYMBOL
    if timeframes is None:
        timeframes = TIMEFRAMES
    if initial_balance is None:
        initial_balance = INITIAL_BALANCE
    if risk_per_trade is None:
        risk_per_trade = RISK_PER_TRADE

    start_time = time.time()
    
    # Загружаем и предобрабатываем данные
    logging.info("Запуск оптимизированного бэктеста")
    print(f"\nЗапуск оптимизированного бэктеста для {symbol}")
    
    data = preprocess_data(start_date=start_date, end_date=end_date, symbol=symbol, timeframes=timeframes)
    if data is None:
        logging.error("Не удалось загрузить данные для бэктеста")
        return None
    
    timeframe_data, timeline, candle_indices = data
    print(f"Данные загружены, начинаем бэктест для {len(timeline)} временных точек")
    
    # Если включена визуализация, выводим сообщение о доступности
    if visualize_realtime:
        if VISUALIZATION_AVAILABLE:
            print(f"Визуализация в реальном времени включена (режим: {visualization_mode})")
        else:
            print("Визуализация в реальном времени недоступна. Установите PyQt5 и PyQtGraph.")
            visualize_realtime = False
    
    # Если параллельная обработка включена и временная шкала достаточно большая и не включена визуализация
    if parallel and len(timeline) > 1000 and not visualize_realtime:
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)  # Оставляем 1 ядро для ОС
        
        print(f"Используем параллельную обработку с {num_processes} процессами")
        logging.info(f"Используем параллельную обработку с {num_processes} процессами")
        
        # Выполняем параллельный бэктест с передачей всех нужных параметров
        all_results, trade_history = parallel_backtest(
            timeline, timeframe_data, candle_indices, num_processes, 
            initial_balance=initial_balance, risk_per_trade=risk_per_trade, 
            symbol=symbol, debug_mode=debug_mode
        )
    else:
        # Последовательная обработка
        if parallel and visualize_realtime:
            print("Параллельная обработка отключена из-за использования визуализации")
        else:
            print("Используем последовательную обработку")
        
        logging.info("Используем последовательную обработку")
        
        all_results, trade_history = sequential_backtest(
            timeline, timeframe_data, candle_indices, visualize_realtime,
            initial_balance=initial_balance, risk_per_trade=risk_per_trade,
            symbol=symbol, debug_mode=debug_mode, visualization_mode=visualization_mode
        )
    
    # Очищаем память
    gc.collect()
    
    # Преобразуем результаты в DataFrame
    if not all_results:
        logging.warning("Бэктест не дал результатов")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    
    # Преобразуем даты
    if 'entry_time' in results_df.columns:
        results_df['entry_time'] = pd.to_datetime(results_df['entry_time'])
    if 'exit_time' in results_df.columns:
        results_df['exit_time'] = pd.to_datetime(results_df['exit_time'])
    
    # Сохраняем полные результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f"backtest_results_{symbol}_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    
    # Сохраняем историю сделок для последующей визуализации
    entries_file = os.path.join(results_dir, f"backtest_entries_{symbol}.csv")
    exits_file = os.path.join(results_dir, f"backtest_exits_{symbol}.csv")
    sl_file = os.path.join(results_dir, f"backtest_sl_{symbol}.csv")
    tp_file = os.path.join(results_dir, f"backtest_tp_{symbol}.csv")
    
    pd.DataFrame(trade_history["entries"]).to_csv(entries_file, index=False)
    pd.DataFrame(trade_history["exits"]).to_csv(exits_file, index=False)
    pd.DataFrame(trade_history["stop_losses"]).to_csv(sl_file, index=False)
    pd.DataFrame(trade_history["take_profits"]).to_csv(tp_file, index=False)
    
    # Рассчитываем и выводим статистику
    if not results_df.empty:
        total_trades = len(results_df)
        wins = results_df[results_df["result"] == "win"].shape[0] if "result" in results_df.columns else 0
        winrate = (wins / total_trades) * 100 if total_trades > 0 else 0
        final_balance = results_df['balance'].iloc[-1] if 'balance' in results_df.columns else initial_balance
        
        # Расчет просадки
        results_df['cumulative_profit'] = results_df['profit'].cumsum()
        results_df['peak'] = results_df['cumulative_profit'].cummax()
        results_df['drawdown'] = (results_df['peak'] - results_df['cumulative_profit']) / results_df['peak'] * 100
        max_drawdown = results_df['drawdown'].max()
        
        elapsed_time = time.time() - start_time
        
        logging.info(f"Бэктест завершен за {elapsed_time:.2f} секунд")
        logging.info(f"Всего сделок: {total_trades}")
        logging.info(f"Побед: {wins}, Винрейт: {winrate:.2f}%")
        logging.info(f"Итоговый баланс: {final_balance:.2f}")
        logging.info(f"Максимальная просадка: {max_drawdown:.2f}%")
        
        print(f"\nБэктест завершен за {timedelta(seconds=int(elapsed_time))}")
        print(f"Всего сделок: {total_trades}")
        print(f"Побед: {wins}, Винрейт: {winrate:.2f}%")
        print(f"Итоговый баланс: {final_balance:.2f}")
        print(f"Максимальная просадка: {max_drawdown:.2f}%")
        print(f"Результаты сохранены в {results_file}")
        
        # Предлагаем использовать стандартную визуализацию
        if not visualize_realtime:
            print("\nДля визуализации результатов используйте команду:")
            print("from visualization import visualize_backtest")
            print("visualize_backtest()")
        
        # Запускаем визуализатор для просмотра результатов, если он не был запущен в реальном времени
        if not visualize_realtime and save_charts:
            try:
                from visualization import visualize_backtest
                print("Запускаем визуализацию результатов бэктеста...")
                visualize_backtest(symbol=symbol, with_prices=True)
            except Exception as e:
                print(f"Ошибка при запуске визуализации: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
    
    return results_df

if __name__ == "__main__":
    # На Windows обязательно оборачиваем в if __name__ == "__main__" для multiprocessing
    # Определим количество процессов в зависимости от системы
    cpu_count = multiprocessing.cpu_count()
    recommended_processes = max(1, cpu_count - 1)  # Оставляем одно ядро для ОС
    
    print(f"Обнаружено {cpu_count} ядер процессора, рекомендуется использовать {recommended_processes} процессов")
    print("Выберите режим бэктеста:")
    print("1. Параллельный бэктест (рекомендуется для больших периодов)")
    print("2. Последовательный бэктест (более точный, но медленнее)")
    
    choice = input("Ваш выбор (1/2): ").strip()
    
    # Дополнительные настройки для бэктеста
    save_charts = input("Сохранить графики после бэктеста? (y/n): ").strip().lower() == 'y'
    
    # Спрашиваем про визуализацию в реальном времени
    visualize_realtime = input("Использовать визуализацию в реальном времени? (y/n): ").strip().lower() == 'y'
    visualization_mode = "full"
    if visualize_realtime:
        print("Выберите режим визуализации:")
        print("1. Полный (все индикаторы и информация)")
        print("2. Простой (только основные графики и сделки)")
        print("3. Производительность (минимальные визуальные эффекты для максимальной скорости)")
        viz_choice = input("Ваш выбор (1/2/3): ").strip()
        
        if viz_choice == "2":
            visualization_mode = "simple"
        elif viz_choice == "3":
            visualization_mode = "performance"
    
    # Настройки периода бэктеста
    custom_period = input("Использовать пользовательский период для бэктеста? (y/n): ").strip().lower() == 'y'
    
    # Параметры для передачи в функцию бэктеста
    backtest_params = {
        'visualize_realtime': visualize_realtime,
        'visualization_mode': visualization_mode,
        'save_charts': save_charts
    }
    
    if custom_period:
        try:
            start_str = input("Введите дату начала (YYYY-MM-DD): ").strip()
            end_str = input("Введите дату окончания (YYYY-MM-DD): ").strip()
            
            backtest_start = datetime.strptime(start_str, "%Y-%m-%d")
            backtest_end = datetime.strptime(end_str, "%Y-%m-%d")
            
            # Добавляем параметры в словарь
            backtest_params['start_date'] = backtest_start
            backtest_params['end_date'] = backtest_end
            
            print(f"Установлен период бэктеста: {backtest_start.strftime('%Y-%m-%d')} - {backtest_end.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"Ошибка при обработке дат: {e}. Используем период по умолчанию.")
    
    # Настройки баланса и риска
    custom_balance = input("Использовать пользовательский начальный баланс? (y/n): ").strip().lower() == 'y'
    if custom_balance:
        try:
            balance_str = input(f"Введите начальный баланс [{INITIAL_BALANCE}]: ").strip()
            if balance_str:
                initial_balance = float(balance_str)
                backtest_params['initial_balance'] = initial_balance
                print(f"Установлен начальный баланс: {initial_balance}")
        except Exception as e:
            print(f"Ошибка при обработке баланса: {e}. Используем значение по умолчанию.")
    
    custom_risk = input("Использовать пользовательский риск на сделку? (y/n): ").strip().lower() == 'y'
    if custom_risk:
        try:
            risk_str = input(f"Введите риск на сделку в % [{RISK_PER_TRADE*100}%]: ").strip()
            if risk_str:
                risk_per_trade = float(risk_str) / 100  # Переводим из % в доли
                backtest_params['risk_per_trade'] = risk_per_trade
                print(f"Установлен риск на сделку: {risk_per_trade*100}%")
        except Exception as e:
            print(f"Ошибка при обработке риска: {e}. Используем значение по умолчанию.")
    
    # Режим отладки
    debug_mode = input("Включить режим отладки? (y/n): ").strip().lower() == 'y'
    backtest_params['debug_mode'] = debug_mode
    
    # Запускаем бэктест в соответствии с выбором
    if choice == "1":
        # Для визуализации в реальном времени рекомендуем последовательный бэктест
        if visualize_realtime:
            print("Внимание: При использовании визуализации в реальном времени рекомендуется последовательный бэктест")
            confirm = input("Продолжить с параллельным бэктестом? (y/n): ").strip().lower()
            if confirm != 'y':
                choice = "2"  # Переключаемся на последовательный бэктест
        
        if choice == "1":  # Если пользователь все же выбрал параллельный
            num_processes = input(f"Введите количество процессов [{recommended_processes}]: ").strip()
            num_processes = int(num_processes) if num_processes else recommended_processes
            
            backtest_params.update({
                'parallel': True,
                'num_processes': num_processes
            })
    else:
        backtest_params.update({
            'parallel': False
        })
    
    # Запускаем бэктест
    start_time = time.time()
    results = optimized_backtest(**backtest_params)
    elapsed_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {timedelta(seconds=int(elapsed_time))}")