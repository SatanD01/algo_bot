import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import matplotlib
import os
import shutil
import gc
import time
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
from config import SYMBOL, BACKTEST_START, BACKTEST_END, TIMEFRAMES
from data_fetcher import get_historical_data
from mt5_connector import connect_mt5, disconnect_mt5

# === Настройки визуализации ===
# Настройки визуализации
CHART_DPI = 150  # Разрешение графиков
CHART_FORMAT = 'png'  # Формат сохранения (png, svg, pdf)
USE_INTERACTIVE = False  # Использовать интерактивные графики
PARALLEL_PROCESSING = True  # Использовать параллельную обработку
MAX_WORKERS = 4  # Максимальное количество параллельных процессов
MAX_MEMORY_PERCENT = 80  # Максимальное использование памяти (%)

matplotlib.use('Agg')  # Использование неинтерактивного бэкенда для предотвращения ошибок в потоках

# Настраиваем логирование
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Директория для логов
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Файл логов с ротацией
log_file = os.path.join(logs_dir, "visualization.log")
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Проверка наличия необходимых библиотек
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("Библиотека psutil не установлена. Мониторинг памяти отключен.")

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False
    logger.warning("Библиотека mplfinance не установлена. Используется стандартный метод отрисовки свечей.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Библиотека plotly не установлена. Интерактивные графики не будут созданы.")
    USE_INTERACTIVE = False

def get_memory_usage():
    """
    Получение текущего использования памяти
    
    Возвращает:
    float: Процент использования памяти или None, если psutil не доступен
    """
    if HAS_PSUTIL:
        return psutil.virtual_memory().percent
    return None

def check_memory_usage(threshold=MAX_MEMORY_PERCENT):
    """
    Проверка использования памяти и принудительная сборка мусора при превышении порога
    
    Параметры:
    threshold (float): Пороговое значение использования памяти в процентах
    
    Возвращает:
    bool: True, если память в норме, False, если есть проблемы
    """
    if not HAS_PSUTIL:
        return True
        
    memory_percent = get_memory_usage()
    if memory_percent > threshold:
        logger.warning(f"Высокое использование памяти: {memory_percent:.1f}%. Выполняется сборка мусора.")
        gc.collect()
        new_memory_percent = get_memory_usage()
        logger.info(f"Использование памяти после сборки мусора: {new_memory_percent:.1f}%")
        if new_memory_percent > threshold:
            logger.error(f"Критическое использование памяти: {new_memory_percent:.1f}%")
            return False
    return True

def create_results_directory():
    """
    Создает директорию для сохранения результатов визуализации с учетом даты и времени.
    
    Возвращает:
    tuple: (results_dir, data_dir, charts_dir, interactive_dir)
    """
    # Создаем основную директорию для результатов, если ее нет
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Создаем подпапку с датой и временем
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(base_results_dir, f"{SYMBOL}_{current_timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Создаем подпапки для разных типов данных
    charts_dir = os.path.join(results_dir, "charts")
    data_dir = os.path.join(results_dir, "data")
    interactive_dir = os.path.join(results_dir, "interactive") if USE_INTERACTIVE else None
    
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    if USE_INTERACTIVE:
        os.makedirs(interactive_dir, exist_ok=True)
    
    logger.info(f"Создана директория для результатов: {results_dir}")
    
    return results_dir, data_dir, charts_dir, interactive_dir

def find_latest_backtest_files():
    try:
        # Ищем самый свежий файл результатов в директории backtest_results
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
        result_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_results_') and f.endswith('.csv')]
        
        if not result_files:
            logger.warning("Файлы с результатами бэктеста не найдены!")
            return None, None, None, None, None
        
        latest_result_file = max(result_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
        latest_result_file = os.path.join(results_dir, latest_result_file)
        logger.info(f"Найден файл результатов: {latest_result_file}")
        
        # Находим соответствующие файлы данных
        entries_file = os.path.join(results_dir, f'backtest_entries_{SYMBOL}.csv')
        exits_file = os.path.join(results_dir, f'backtest_exits_{SYMBOL}.csv')
        sl_file = os.path.join(results_dir, f'backtest_sl_{SYMBOL}.csv')
        tp_file = os.path.join(results_dir, f'backtest_tp_{SYMBOL}.csv')
        
        return latest_result_file, entries_file, exits_file, sl_file, tp_file
    except Exception as e:
        logger.error(f"Ошибка при поиске файлов бэктеста: {str(e)}")
        return None, None, None, None, None

def copy_files_to_data_dir(files, data_dir):
    """
    Копирует файлы в директорию данных
    
    Параметры:
    files (list): Список файлов для копирования
    data_dir (str): Директория назначения
    
    Возвращает:
    dict: Словарь с новыми путями к файлам
    """
    new_paths = {}
    
    for file_path in files:
        if file_path and os.path.exists(file_path):
            new_file_path = os.path.join(data_dir, os.path.basename(file_path))
            shutil.copy2(file_path, new_file_path)
            new_paths[file_path] = new_file_path
            logger.info(f"Файл скопирован: {file_path} -> {new_file_path}")
        else:
            if file_path:
                logger.warning(f"Файл не найден: {file_path}")
    
    return new_paths

def load_and_optimize_df(file_path, time_column='time', max_rows=None):
    """
    Загружает DataFrame из CSV и оптимизирует использование памяти
    
    Параметры:
    file_path (str): Путь к файлу
    time_column (str): Название столбца с временем
    max_rows (int): Максимальное количество строк для загрузки
    
    Возвращает:
    DataFrame: Загруженные данные или None в случае ошибки
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Файл {file_path} не найден")
            return None
        
        # Определяем типы данных для экономии памяти
        dtype_dict = {
            'order': 'category',
            'type': 'category',
            'result': 'category',
            'tf': 'category',
            'setup': 'category'
        }
        
        # Загружаем только нужное количество строк, если указано
        df = pd.read_csv(file_path, dtype=dtype_dict, nrows=max_rows)
        
        # Преобразуем временную колонку
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Оптимизируем численные столбцы
        for col in df.columns:
            # Преобразуем int64 в int32 или int16, если возможно
            if df[col].dtype == 'int64':
                # Проверяем диапазон значений
                col_min, col_max = df[col].min(), df[col].max()
                if col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # Преобразуем float64 в float32, если возможно
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype(np.float32)
        
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
        return None

def load_backtest_data(data_dir, max_rows=None):
    """
    Загрузка результатов бэктеста и данных свечей из указанной директории
    
    Параметры:
    data_dir (str): Директория с данными
    max_rows (int): Максимальное количество строк для загрузки
    
    Возвращает:
    tuple: (results, entries, exits, stop_losses, take_profits)
    """
    # Ищем файлы в директории
    result_files = [f for f in os.listdir(data_dir) if f.startswith('backtest_results_') and f.endswith('.csv')]
    
    if not result_files:
        logger.warning("Файлы с результатами бэктеста не найдены в директории данных!")
        return None, None, None, None, None
    
    try:
        latest_result_file = os.path.join(data_dir, max(result_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x))))
        logger.info(f"Загрузка результатов из файла: {latest_result_file}")
        
        # Загружаем результаты бэктеста
        results = load_and_optimize_df(latest_result_file, 'entry_time', max_rows)
        
        # Загружаем данные о входах и выходах
        entries_file = os.path.join(data_dir, f'backtest_entries_{SYMBOL}.csv')
        exits_file = os.path.join(data_dir, f'backtest_exits_{SYMBOL}.csv')
        sl_file = os.path.join(data_dir, f'backtest_sl_{SYMBOL}.csv')
        tp_file = os.path.join(data_dir, f'backtest_tp_{SYMBOL}.csv')
        
        entries = load_and_optimize_df(entries_file, 'time')
        exits = load_and_optimize_df(exits_file, 'time')
        stop_losses = load_and_optimize_df(sl_file, 'time')
        take_profits = load_and_optimize_df(tp_file, 'time')
        
        # Проверяем использование памяти
        check_memory_usage()
        
        return results, entries, exits, stop_losses, take_profits
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None

def get_price_data(start_date=None, end_date=None, symbol=None, timeframes=None):
    """
    Получение данных цен для построения графиков с оптимизацией
    
    Параметры:
    start_date (datetime, str, optional): Начальная дата. Если None, используется BACKTEST_START
    end_date (datetime, str, optional): Конечная дата. Если None, используется BACKTEST_END
    symbol (str, optional): Торговый символ. Если None, используется SYMBOL
    timeframes (list, optional): Список таймфреймов. Если None, используется выбранные из TIMEFRAMES
    
    Возвращает:
    dict: Словарь с данными по таймфреймам или None в случае ошибки
    """
    if not connect_mt5():
        logger.error("Не удалось подключиться к MT5")
        return None
    
    if symbol is None:
        symbol = SYMBOL
    
    # Преобразуем строки в datetime, если они переданы как строки
    if start_date is not None and isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Неверный формат даты: {start_date}")
            start_date = BACKTEST_START
    
    if end_date is not None and isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Неверный формат даты: {end_date}")
            end_date = BACKTEST_END
    
    # Установка значений по умолчанию
    if start_date is None:
        start_date = BACKTEST_START
    
    if end_date is None:
        end_date = BACKTEST_END
        
    if timeframes is None:
        # Используем основные таймфреймы для визуализации
        all_tfs = ["M5", "M15", "H1", "H4", "D1"]
        timeframes = [tf for tf in all_tfs if tf in TIMEFRAMES]
    
    logger.info(f"Загрузка ценовых данных для {symbol}...")
    
    # Загружаем данные для разных таймфреймов
    price_data = {}
    
    try:
        # При параллельной обработке используем ThreadPoolExecutor
        if PARALLEL_PROCESSING and len(timeframes) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(timeframes), MAX_WORKERS)) as executor:
                future_to_tf = {
                    executor.submit(
                        get_historical_data, 
                        symbol, 
                        tf, 
                        start_date, 
                        end_date
                    ): tf for tf in timeframes
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_tf), 
                                   total=len(future_to_tf),
                                   desc="Загрузка данных"):
                    tf = future_to_tf[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            # Оптимизируем DataFrame для экономии памяти
                            for col in df.columns:
                                if col not in ['time']:
                                    if df[col].dtype == 'float64':
                                        df[col] = df[col].astype('float32')
                                    elif df[col].dtype == 'int64':
                                        # Проверяем диапазон значений
                                        if df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
                                            df[col] = df[col].astype('int32')
                                        
                            price_data[tf] = df
                            logger.info(f"Загружено {len(df)} свечей для {tf}")
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке данных для {tf}: {str(e)}")
        else:
            # Последовательная загрузка
            for tf in tqdm(timeframes, desc="Загрузка данных"):
                df = get_historical_data(symbol, timeframe=tf, start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    # Оптимизируем DataFrame для экономии памяти
                    for col in df.columns:
                        if col not in ['time']:
                            if df[col].dtype == 'float64':
                                df[col] = df[col].astype('float32')
                    
                    price_data[tf] = df
                    logger.info(f"Загружено {len(df)} свечей для {tf}")
        
        # Проверяем использование памяти
        check_memory_usage()
        
        return price_data
    
    except Exception as e:
        logger.error(f"Ошибка при получении данных цен: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
    finally:
        disconnect_mt5()

def plot_balance_equity(results, charts_dir, interactive_dir=None):
    """
    Построение графика баланса и средств с возможностью интерактивности
    и улучшенным сглаживанием для более плавного отображения
    
    Параметры:
    results (DataFrame): Результаты бэктеста
    charts_dir (str): Путь к директории для сохранения графиков
    interactive_dir (str, optional): Путь к директории для интерактивных графиков
    """
    if results is None or results.empty:
        logger.warning("Нет данных для построения графика баланса")
        return
    
    start_time = time.time()
    
    try:
        # Создаем локальную копию данных для работы
        df = results.copy()
        
        # Проверяем и преобразуем столбец exit_time
        if 'exit_time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['exit_time']):
                df['exit_time'] = pd.to_datetime(df['exit_time'])
        else:
            logger.error("В данных отсутствует столбец 'exit_time'")
            return
        
        # Сортируем данные по времени
        df = df.sort_values(by='exit_time')
        
        # Вычисляем накопленную прибыль
        df['cumulative_profit'] = df['profit'].cumsum() + df['balance'].iloc[0]
        
        # Создаем фигуру большего размера для лучшей детализации
        plt.figure(figsize=(18, 10))
        
        # Создаем ежедневные данные с заполнением пробелов для плавности
        # Определяем минимальную и максимальную даты
        min_date = df['exit_time'].min().replace(hour=0, minute=0, second=0)
        max_date = df['exit_time'].max() + pd.Timedelta(days=1)
        
        # Создаем равномерный временной ряд с ежедневными интервалами
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        smooth_df = pd.DataFrame(index=date_range)
        
        # Преобразуем индекс в колонку для слияния
        smooth_df['date'] = smooth_df.index
        
        # Создаем колонки с датами (без времени) для сопоставления
        df['date'] = df['exit_time'].dt.floor('D')
        
        # Группируем по дате, сохраняя последнюю запись для каждого дня
        daily_df = df.groupby('date', as_index=False).last()
        
        # Объединяем с полным диапазоном дат
        merged_df = pd.merge(smooth_df, daily_df, on='date', how='left')
        
        # Заполняем пропущенные значения
        merged_df['balance'] = merged_df['balance'].ffill()
        merged_df['cumulative_profit'] = merged_df['cumulative_profit'].ffill()
        
        # Заполняем начальные значения, если есть пропуски в начале
        initial_balance = df['balance'].iloc[0] - df['profit'].iloc[0] if not df.empty else 10000
        merged_df['balance'] = merged_df['balance'].fillna(initial_balance)
        merged_df['cumulative_profit'] = merged_df['cumulative_profit'].fillna(initial_balance)
        
        # Рисуем линии баланса и накопленной прибыли
        plt.plot(merged_df['date'], merged_df['balance'], '-', color='blue', linewidth=2, label='Баланс')
        plt.plot(merged_df['date'], merged_df['cumulative_profit'], '--', color='orange', linewidth=1.5, label='Накопленная прибыль')
        
        # Создаем словари для точек выигрышей и проигрышей
        wins_x = []
        wins_y = []
        losses_x = []
        losses_y = []
        
        # Отображаем маркеры для каждой сделки, правильно размещая их на линии баланса
        for idx, row in df.iterrows():
            # Находим ближайшую дату в merged_df к текущей сделке
            date_diff = abs(merged_df['date'] - row['exit_time'])
            date_idx = date_diff.idxmin() if not date_diff.empty else None
            
            if date_idx is not None:
                # Используем значение баланса из merged_df для этой точки
                plot_balance = merged_df.loc[date_idx, 'balance']
                
                # Собираем точки в соответствующие списки
                if row['result'] == 'win':
                    wins_x.append(row['exit_time'])
                    wins_y.append(plot_balance)
                else:
                    losses_x.append(row['exit_time'])
                    losses_y.append(plot_balance)
        
        # Отображаем точки выигрышей и проигрышей
        if wins_x:
            plt.scatter(wins_x, wins_y, color='green', s=50, marker='o', alpha=0.7, label='Выигрыши', zorder=10)
        if losses_x:
            plt.scatter(losses_x, losses_y, color='red', s=50, marker='x', alpha=0.7, label='Проигрыши', zorder=10)
        
        # Добавляем статистику
        if not df.empty:
            initial_balance = df['balance'].iloc[0] - df['profit'].iloc[0]
            final_balance = df['balance'].iloc[-1]
            profit_percent = ((final_balance / initial_balance) - 1) * 100
            
            plt.annotate(
                f"Начальный баланс: {initial_balance:.2f}\n"
                f"Конечный баланс: {final_balance:.2f}\n"
                f"Рост: {profit_percent:.2f}%",
                xy=(0.02, 0.92),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Настройки графика
        plt.title(f'Динамика баланса для {SYMBOL}', fontsize=14)
        plt.xlabel('Время', fontsize=12)
        plt.ylabel('Баланс', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Форматирование оси X с равномерными метками
        ax = plt.gca()
        
        # Установка основного формата даты
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Определение количества меток для хорошей читабельности
        num_ticks = min(15, len(merged_df))
        
        # Устанавливаем локатор с автоматическим выбором интервала
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=min(5, len(merged_df)), maxticks=num_ticks))
        
        # Поворачиваем метки для лучшей читабельности
        plt.xticks(rotation=45)
        
        # Установка плотной компоновки
        plt.tight_layout()
        
        # Сохраняем график в папку charts
        chart_file = os.path.join(charts_dir, f'balance_chart_{SYMBOL}.{CHART_FORMAT}')
        plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
        logger.info(f"График баланса сохранен в {chart_file}")
        plt.close()
        
        # Если включены интерактивные графики, создаем их с помощью Plotly
        if USE_INTERACTIVE and interactive_dir and HAS_PLOTLY:
            try:
                # Создаем фигуру с двумя осями Y
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Добавляем линию баланса (более плавно)
                fig.add_trace(
                    go.Scatter(
                        x=merged_df['date'], 
                        y=merged_df['balance'],
                        mode='lines',
                        name='Баланс',
                        line=dict(color='blue', width=2, shape='spline', smoothing=1.3)
                    )
                )
                
                # Добавляем накопленную прибыль (более плавно)
                fig.add_trace(
                    go.Scatter(
                        x=merged_df['date'], 
                        y=merged_df['cumulative_profit'],
                        mode='lines',
                        name='Накопленная прибыль',
                        line=dict(color='orange', width=1.5, dash='dash', shape='spline', smoothing=1.3)
                    )
                )
                
                # Добавляем выигрыши и проигрыши
                if wins_x:
                    fig.add_trace(
                        go.Scatter(
                            x=wins_x,
                            y=wins_y,
                            mode='markers',
                            name='Выигрыши',
                            marker=dict(color='green', size=10, symbol='circle')
                        )
                    )
                
                if losses_x:
                    fig.add_trace(
                        go.Scatter(
                            x=losses_x,
                            y=losses_y,
                            mode='markers',
                            name='Проигрыши',
                            marker=dict(color='red', size=10, symbol='x')
                        )
                    )
                
                # Добавляем просадку
                if 'drawdown' not in df.columns:
                    df['drawdown'] = (df['cumulative_profit'].cummax() - df['cumulative_profit']) / df['cumulative_profit'].cummax() * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=df['exit_time'],
                        y=df['drawdown'],
                        mode='lines',
                        name='Просадка (%)',
                        line=dict(color='red', width=1.5),
                        visible='legendonly'  # Скрыто по умолчанию
                    ),
                    secondary_y=True
                )
                
                # Настраиваем макет
                fig.update_layout(
                    title=f'Динамика баланса для {SYMBOL} (интерактивный)',
                    xaxis_title='Время',
                    yaxis_title='Баланс',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template='plotly_white',
                    width=1200,
                    height=700
                )
                
                # Настраиваем вторичную ось Y
                fig.update_yaxes(title_text="Просадка (%)", secondary_y=True)
                
                # Добавляем больше точек данных для оси X
                fig.update_xaxes(
                    tickmode='auto',
                    nticks=20,  # Больше делений для более детального отображения
                    rangeslider_visible=False  # Можно включить, если нужен слайдер
                )
                
                # Сохраняем интерактивный график
                interactive_file = os.path.join(interactive_dir, f'interactive_balance_{SYMBOL}.html')
                fig.write_html(interactive_file, include_plotlyjs='cdn')
                logger.info(f"Интерактивный график баланса сохранен в {interactive_file}")
            except Exception as e:
                logger.error(f"Ошибка при создании интерактивного графика: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Очищаем память
        del df, merged_df
        if 'wins_x' in locals(): del wins_x
        if 'wins_y' in locals(): del wins_y
        if 'losses_x' in locals(): del losses_x
        if 'losses_y' in locals(): del losses_y
        gc.collect()
        
        logger.info(f"Построение графика баланса заняло {time.time() - start_time:.2f} секунд")
    
    except Exception as e:
        logger.error(f"Ошибка при построении графика баланса: {str(e)}")
        logger.error(traceback.format_exc())
        if 'plt' in locals() or 'plt' in globals():
            plt.close()  # Закрываем график в случае ошибки

def plot_drawdown(results, charts_dir, interactive_dir=None):
    """
    Построение графика просадки
    
    Параметры:
    results (DataFrame): Результаты бэктеста
    charts_dir (str): Путь к директории для сохранения графиков
    interactive_dir (str, optional): Путь к директории для интерактивных графиков
    """
    if results is None or results.empty:
        logger.warning("Нет данных для построения графика просадки")
        return
    
    start_time = time.time()
    
    try:
        # Создаем локальную копию данных
        df = results.copy()
        
        # Вычисляем текущий максимум и текущую просадку
        df['cumulative_profit'] = df['profit'].cumsum() + df['balance'].iloc[0]
        df['running_max'] = df['cumulative_profit'].cummax()
        df['drawdown'] = (df['running_max'] - df['cumulative_profit']) / df['running_max'] * 100
        df['drawdown_abs'] = df['running_max'] - df['cumulative_profit']
        
        # Находим максимальную просадку и ее дату
        max_dd_idx = df['drawdown'].idxmax()
        max_dd = df.loc[max_dd_idx, 'drawdown']
        max_dd_date = df.loc[max_dd_idx, 'exit_time']
        
        # Создаем график с matplotlib
        plt.figure(figsize=(14, 7))
        
        # Основной график просадки в процентах
        ax1 = plt.gca()
        ax1.plot(df['exit_time'], df['drawdown'], color='red', linewidth=1.5)
        ax1.fill_between(df['exit_time'], df['drawdown'], 0, color='red', alpha=0.3)
        ax1.set_title(f'Просадка для {SYMBOL}')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Просадка (%)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Инвертируем ось Y для наглядности
        
        # Добавляем аннотацию о максимальной просадке
        date_str = max_dd_date.strftime('%Y-%m-%d') if isinstance(max_dd_date, (datetime, pd.Timestamp)) else str(max_dd_date)
        
        ax1.annotate(
            f'Макс. просадка: {max_dd:.2f}%\nДата: {date_str}',
            xy=(max_dd_date, max_dd),
            xytext=(30, 30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='darkred'),
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
        
        # Добавим вторую ось Y для абсолютной просадки
        ax2 = ax1.twinx()
        ax2.plot(df['exit_time'], df['drawdown_abs'], color='blue', linestyle='--', linewidth=1.0, alpha=0.7)
        ax2.set_ylabel('Абсолютная просадка', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Форматирование оси X
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Сохраняем график в папку charts
        chart_file = os.path.join(charts_dir, f'drawdown_chart_{SYMBOL}.{CHART_FORMAT}')
        plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
        logger.info(f"График просадки сохранен в {chart_file}")
        plt.close()
        
        # Создаем интерактивный график, если нужно
        if USE_INTERACTIVE and interactive_dir and HAS_PLOTLY:
            # Создаем фигуру с двумя осями Y
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Продолжение функции plot_drawdown
            # Добавляем график просадки в процентах
            fig.add_trace(
                go.Scatter(
                    x=df['exit_time'],
                    y=df['drawdown'],
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    mode='lines',
                    name='Просадка (%)',
                    line=dict(color='red', width=2)
                )
            )
            
            # Добавляем график абсолютной просадки
            fig.add_trace(
                go.Scatter(
                    x=df['exit_time'],
                    y=df['drawdown_abs'],
                    mode='lines',
                    name='Абсолютная просадка',
                    line=dict(color='blue', width=1.5, dash='dash')
                ),
                secondary_y=True
            )
            
            # Находим максимальную просадку
            if not pd.isna(max_dd_idx):
                fig.add_trace(
                    go.Scatter(
                        x=[max_dd_date],
                        y=[max_dd],
                        mode='markers+text',
                        name='Макс. просадка',
                        marker=dict(color='darkred', size=10, symbol='diamond'),
                        text=[f"{max_dd:.2f}%"],
                        textposition="top center"
                    )
                )
            
            # Настраиваем макет
            fig.update_layout(
                title=f'Просадка для {SYMBOL} (интерактивный)',
                xaxis_title='Время',
                yaxis_title='Просадка (%)',
                yaxis=dict(autorange="reversed"),  # Инвертируем ось Y
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white'
            )
            
            # Настраиваем вторичную ось Y
            fig.update_yaxes(title_text="Абсолютная просадка", secondary_y=True)
            
            # Сохраняем интерактивный график
            interactive_file = os.path.join(interactive_dir, f'interactive_drawdown_{SYMBOL}.html')
            fig.write_html(interactive_file, include_plotlyjs='cdn')
            logger.info(f"Интерактивный график просадки сохранен в {interactive_file}")
        
        # Очищаем память
        del df
        gc.collect()
        
        logger.info(f"Построение графика просадки заняло {time.time() - start_time:.2f} секунд")
    
    except Exception as e:
        logger.error(f"Ошибка при построении графика просадки: {str(e)}")
        logger.error(traceback.format_exc())
        if 'plt' in locals() or 'plt' in globals():
            plt.close()  # Закрываем график в случае ошибки

def plot_trades_distribution(results, charts_dir, interactive_dir=None):
    """
    Построение распределения сделок с оптимизацией и расширенной аналитикой
    
    Параметры:
    results (DataFrame): Результаты бэктеста
    charts_dir (str): Путь к директории для сохранения графиков
    interactive_dir (str, optional): Путь к директории для интерактивных графиков
    """
    if results is None or results.empty:
        logger.warning("Нет данных для построения распределения сделок")
        return
    
    start_time = time.time()
    
    try:
        # Создаем фигуру с 6 подграфиками
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Распределение по типу ордера (Buy/Sell)
        order_counts = results['order'].value_counts()
        axs[0, 0].pie(order_counts, labels=order_counts.index, autopct='%1.1f%%',
                   colors=['royalblue', 'tomato'], startangle=90)
        axs[0, 0].set_title('Распределение по типу ордера')
        
        # 2. Распределение по результату (Win/Loss)
        result_counts = results['result'].value_counts()
        axs[0, 1].pie(result_counts, labels=result_counts.index, autopct='%1.1f%%',
                   colors=['limegreen', 'crimson'], startangle=90)
        axs[0, 1].set_title('Распределение по результату')
        
        # 3. Распределение по таймфреймам
        if 'tf' in results.columns:
            tf_counts = results['tf'].value_counts()
            axs[0, 2].pie(tf_counts, labels=tf_counts.index, autopct='%1.1f%%',
                       startangle=90)
            axs[0, 2].set_title('Распределение по таймфреймам')
        
        # 4. Распределение прибыли
        profit_data = results['profit'].copy()
        # Ограничиваем выбросы для лучшей визуализации
        q1, q3 = np.percentile(profit_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        profit_data = profit_data[(profit_data >= lower_bound) & (profit_data <= upper_bound)]
        
        axs[1, 0].hist(profit_data, bins=20, color='seagreen', alpha=0.7)
        axs[1, 0].set_title('Распределение прибыли')
        axs[1, 0].set_xlabel('Прибыль')
        axs[1, 0].set_ylabel('Количество сделок')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 5. Распределение по сетапам
        if 'setup' in results.columns:
            setup_counts = results['setup'].value_counts()
            axs[1, 1].bar(setup_counts.index, setup_counts.values, color='slateblue', alpha=0.7)
            axs[1, 1].set_title('Распределение по сетапам')
            axs[1, 1].set_xlabel('Сетап')
            axs[1, 1].set_ylabel('Количество сделок')
            axs[1, 1].tick_params(axis='x', rotation=45)
            axs[1, 1].grid(True, alpha=0.3)
        
        # 6. Винрейт по сетапам с исправленным параметром observed=True
        if 'setup' in results.columns:
            setup_winrate = results.groupby('setup', observed=True)['result'].apply(
                lambda x: 100 * sum(x == 'win') / len(x) if len(x) > 0 else 0
            ).sort_values(ascending=False)
            
            axs[1, 2].bar(setup_winrate.index, setup_winrate.values, color='darkorange', alpha=0.7)
            axs[1, 2].set_title('Винрейт по сетапам (%)')
            axs[1, 2].set_xlabel('Сетап')
            axs[1, 2].set_ylabel('Винрейт (%)')
            axs[1, 2].tick_params(axis='x', rotation=45)
            axs[1, 2].grid(True, alpha=0.3)
            # Добавляем горизонтальную линию для среднего винрейта
            avg_winrate = 100 * sum(results['result'] == 'win') / len(results)
            axs[1, 2].axhline(y=avg_winrate, color='red', linestyle='--', label=f'Средний винрейт: {avg_winrate:.1f}%')
            axs[1, 2].legend()
        
        plt.tight_layout()
        
        # Сохраняем график в папку charts
        chart_file = os.path.join(charts_dir, f'trades_distribution_{SYMBOL}.{CHART_FORMAT}')
        plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
        logger.info(f"График распределения сделок сохранен в {chart_file}")
        plt.close()
        
        # Создаем интерактивные графики, если нужно
        if USE_INTERACTIVE and interactive_dir and HAS_PLOTLY:
            # Интерактивное распределение типов ордеров
            fig_order = go.Figure(data=[go.Pie(
                labels=order_counts.index,
                values=order_counts.values,
                hole=.3,
                marker_colors=['royalblue', 'tomato']
            )])
            fig_order.update_layout(title=f'Распределение по типу ордера ({SYMBOL})')
            order_file = os.path.join(interactive_dir, f'interactive_order_types_{SYMBOL}.html')
            fig_order.write_html(order_file, include_plotlyjs='cdn')
            
            # Интерактивное распределение результатов
            fig_result = go.Figure(data=[go.Pie(
                labels=result_counts.index,
                values=result_counts.values,
                hole=.3,
                marker_colors=['limegreen', 'crimson']
            )])
            fig_result.update_layout(title=f'Распределение по результату ({SYMBOL})')
            result_file = os.path.join(interactive_dir, f'interactive_results_{SYMBOL}.html')
            fig_result.write_html(result_file, include_plotlyjs='cdn')
            
            # Интерактивное распределение по сетапам, если есть
            if 'setup' in results.columns:
                fig_setup = go.Figure(data=[
                    go.Bar(
                        x=setup_counts.index,
                        y=setup_counts.values,
                        marker_color='slateblue'
                    )
                ])
                fig_setup.update_layout(
                    title=f'Распределение по сетапам ({SYMBOL})',
                    xaxis_title='Сетап',
                    yaxis_title='Количество сделок'
                )
                setup_file = os.path.join(interactive_dir, f'interactive_setups_{SYMBOL}.html')
                fig_setup.write_html(setup_file, include_plotlyjs='cdn')
            
            logger.info(f"Интерактивные графики распределения сохранены")
        
        # Очищаем память
        if 'profit_data' in locals(): del profit_data
        gc.collect()
        
        logger.info(f"Построение распределения сделок заняло {time.time() - start_time:.2f} секунд")
    
    except Exception as e:
        logger.error(f"Ошибка при построении распределения сделок: {str(e)}")
        logger.error(traceback.format_exc())
        if 'plt' in locals() or 'plt' in globals():
            plt.close()  # Закрываем график в случае ошибки

def plot_price_with_trades(price_data, entries, exits, stop_losses, take_profits, timeframe, charts_dir, interactive_dir=None):
    """
    Построение графика цены с отметками сделок с использованием matplotlib
    
    Параметры:
    price_data (dict): Данные цен по таймфреймам
    entries (DataFrame): Данные о входах в позицию
    exits (DataFrame): Данные о выходах из позиции
    stop_losses (DataFrame): Данные о стоп-лоссах
    take_profits (DataFrame): Данные о тейк-профитах
    timeframe (str): Таймфрейм для построения
    charts_dir (str): Путь к директории для сохранения графиков
    interactive_dir (str, optional): Путь к директории для интерактивных графиков
    """
    if price_data is None or timeframe not in price_data:
        logger.warning(f"Нет данных для построения графика цены на таймфрейме {timeframe}")
        return
    
    start_time = time.time()
    
    try:
        # Получаем данные цены для выбранного таймфрейма
        df = price_data[timeframe].copy()
        
        # Проверяем, что данные в правильном формате
        if 'time' not in df.columns:
            logger.error(f"В данных отсутствует столбец 'time'")
            return
        
        # Создаем диапазон дат для ограничения графика
        entries_times = entries['time'].tolist() if entries is not None and not entries.empty else []
        exits_times = exits['time'].tolist() if exits is not None and not exits.empty else []
        all_times = entries_times + exits_times
        
        if all_times:
            min_date = max(min(all_times) - timedelta(days=5), df['time'].min())
            max_date = min(max(all_times) + timedelta(days=5), df['time'].max())
        else:
            # Если нет сделок, берем последний месяц данных
            max_date = df['time'].max()
            min_date = max_date - timedelta(days=30)
        
        # Фильтруем данные по выбранному диапазону дат
        df = df[(df['time'] >= min_date) & (df['time'] <= max_date)]
        
        if df.empty:
            logger.warning(f"Нет данных в выбранном диапазоне дат для таймфрейма {timeframe}")
            return
        
        # Версия с использованием mplfinance (без отметок сделок)
        if HAS_MPLFINANCE:
            try:
                # Подготавливаем данные для mplfinance
                df_mpf = df.copy()
                df_mpf = df_mpf.set_index('time')
                
                # Переименовываем колонки в формат mplfinance
                df_mpf = df_mpf.rename(columns={
                    'open': 'Open', 
                    'high': 'High', 
                    'low': 'Low', 
                    'close': 'Close', 
                    'tick_volume': 'Volume'
                })
                
                # Настройки для отображения графика без дополнительных отметок
                kwargs = dict(
                    type='candle',
                    volume=False,
                    title=f'\n{SYMBOL} ({timeframe})',
                    ylabel='Цена',
                    ylabel_lower='Объем',
                    figsize=(16, 9),
                    style='yahoo',
                    datetime_format='%Y-%m-%d %H:%M',
                    warn_too_much_data=len(df_mpf) + 1000  # Отключаем предупреждение о большом количестве данных
                )
                
                # Создаем и сохраняем график
                fig, _ = mpf.plot(df_mpf, **kwargs, returnfig=True)
                chart_file = os.path.join(charts_dir, f'price_chart_{SYMBOL}_{timeframe}.{CHART_FORMAT}')
                fig.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"График цены сохранен в {chart_file} (без отметок сделок)")
                
                # Создаем дополнительный график с отметками сделок с помощью matplotlib
                plt.figure(figsize=(16, 9))
                
                # Получаем массив данных
                x = np.arange(len(df))
                
                # Построение линии цены закрытия
                plt.plot(x, df['close'], color='blue', alpha=0.7, linewidth=1)
                
                # Функция для поиска ближайшего индекса для времени
                def find_nearest_index(timestamp):
                    # Найти ближайшую дату в датафрейме df
                    idx = (df['time'] - timestamp).abs().idxmin()
                    return df.index.get_loc(idx)
                
                # Добавляем отметки входов в позицию
                if entries is not None and not entries.empty:
                    entries_in_range = entries[(entries['time'] >= min_date) & (entries['time'] <= max_date)]
                    
                    for _, entry in entries_in_range.iterrows():
                        entry_time = entry['time']
                        # Находим ближайший индекс
                        try:
                            nearest_idx = find_nearest_index(entry_time)
                            
                            if entry['order'] == 'buy':
                                marker = '^'
                                color = 'green'
                            else:  # sell
                                marker = 'v'
                                color = 'red'
                            
                            plt.scatter(nearest_idx, entry['price'], color=color, s=100, marker=marker, 
                                        label=f"{entry['order']} entry ({entry.get('setup', 'Standard')})")
                            
                            # Добавляем стоп-лосс и тейк-профит, если они есть
                            if stop_losses is not None:
                                sl_for_entry = stop_losses[stop_losses['time'] == entry_time]
                                for _, sl in sl_for_entry.iterrows():
                                    plt.axhline(y=sl['price'], color='red', linestyle='--', alpha=0.3, 
                                                xmin=nearest_idx/len(x), xmax=1.0)
                                    plt.text(nearest_idx, sl['price'], 'SL', backgroundcolor='white')
                            
                            if take_profits is not None:
                                tp_for_entry = take_profits[take_profits['time'] == entry_time]
                                for _, tp in tp_for_entry.iterrows():
                                    plt.axhline(y=tp['price'], color='green', linestyle='--', alpha=0.3,
                                                xmin=nearest_idx/len(x), xmax=1.0)
                                    plt.text(nearest_idx, tp['price'], 'TP', backgroundcolor='white')
                        except Exception as e:
                            logger.warning(f"Не удалось отобразить вход для времени {entry_time}: {str(e)}")
                
                # Добавляем отметки выходов из позиции
                if exits is not None and not exits.empty:
                    exits_in_range = exits[(exits['time'] >= min_date) & (exits['time'] <= max_date)]
                    
                    for _, exit_data in exits_in_range.iterrows():
                        exit_time = exit_data['time']
                        # Находим ближайший индекс
                        try:
                            nearest_idx = find_nearest_index(exit_time)
                            
                            if exit_data['type'] == 'take_profit':
                                marker = 'o'
                                color = 'green'
                                label = 'TP exit'
                            elif exit_data['type'] == 'stop_loss':
                                marker = 'x'
                                color = 'red'
                                label = 'SL exit'
                            else:
                                marker = 's'
                                color = 'blue'
                                label = 'Close exit'
                            
                            plt.scatter(nearest_idx, exit_data['price'], color=color, s=100, marker=marker, label=label)
                        except Exception as e:
                            logger.warning(f"Не удалось отобразить выход для времени {exit_time}: {str(e)}")
                
                # Форматирование графика
                plt.title(f'График цены {SYMBOL} ({timeframe}) с отметками сделок')
                plt.xlabel('Время')
                plt.ylabel('Цена')
                plt.grid(True, alpha=0.3)
                
                # Устанавливаем метки дат на оси X
                # Используем только часть дат для читаемости
                num_ticks = min(10, len(df))
                tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
                plt.xticks(tick_indices, [df['time'].iloc[i].strftime('%Y-%m-%d %H:%M') for i in tick_indices], rotation=45)
                
                # Удаляем дублирующиеся метки в легенде
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:  # Проверяем, что есть метки для легенды
                    plt.legend(by_label.values(), by_label.keys(), loc='best')
                
                plt.tight_layout()
                
                # Сохраняем график с отметками в отдельный файл
                chart_file_trades = os.path.join(charts_dir, f'price_chart_{SYMBOL}_{timeframe}_trades.{CHART_FORMAT}')
                plt.savefig(chart_file_trades, dpi=CHART_DPI, bbox_inches='tight')
                logger.info(f"График цены с отметками сделок сохранен в {chart_file_trades}")
                plt.close()
                
            except Exception as e:
                logger.error(f"Ошибка при построении графика цены с mplfinance для {timeframe}: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            # Вариант с использованием только matplotlib
            plt.figure(figsize=(16, 9))
            
            # Строим линейный график цены закрытия
            plt.plot(df['time'], df['close'], color='blue', alpha=0.7)
            
            # Добавляем отметки входов в позицию
            if entries is not None and not entries.empty:
                entries_in_range = entries[(entries['time'] >= min_date) & (entries['time'] <= max_date)]
                
                for _, entry in entries_in_range.iterrows():
                    entry_time = entry['time']
                    # Проверяем, что время находится в диапазоне графика
                    if entry_time >= df['time'].min() and entry_time <= df['time'].max():
                        if entry['order'] == 'buy':
                            marker = '^'
                            color = 'green'
                        else:  # sell
                            marker = 'v'
                            color = 'red'
                        
                        plt.scatter(entry_time, entry['price'], color=color, s=100, marker=marker, 
                                    label=f"{entry['order']} entry ({entry.get('setup', 'Standard')})")
                        
                        # Добавляем стоп-лосс и тейк-профит, если они есть
                        if stop_losses is not None:
                            sl_for_entry = stop_losses[stop_losses['time'] == entry_time]
                            for _, sl in sl_for_entry.iterrows():
                                plt.axhline(y=sl['price'], color='red', linestyle='--', alpha=0.3)
                                plt.text(entry_time, sl['price'], 'SL', backgroundcolor='white')
                        
                        if take_profits is not None:
                            tp_for_entry = take_profits[take_profits['time'] == entry_time]
                            for _, tp in tp_for_entry.iterrows():
                                plt.axhline(y=tp['price'], color='green', linestyle='--', alpha=0.3)
                                plt.text(entry_time, tp['price'], 'TP', backgroundcolor='white')
            
            # Добавляем отметки выходов из позиции
            if exits is not None and not exits.empty:
                exits_in_range = exits[(exits['time'] >= min_date) & (exits['time'] <= max_date)]
                
                for _, exit_data in exits_in_range.iterrows():
                    exit_time = exit_data['time']
                    # Проверяем, что время находится в диапазоне графика
                    if exit_time >= df['time'].min() and exit_time <= df['time'].max():
                        if exit_data['type'] == 'take_profit':
                            marker = 'o'
                            color = 'green'
                            label = 'TP exit'
                        elif exit_data['type'] == 'stop_loss':
                            marker = 'x'
                            color = 'red'
                            label = 'SL exit'
                        else:
                            marker = 's'
                            color = 'blue'
                            label = 'Close exit'
                        
                        plt.scatter(exit_time, exit_data['price'], color=color, s=100, marker=marker, label=label)
            
            # Форматирование графика
            plt.title(f'График цены {SYMBOL} ({timeframe}) с отметками сделок')
            plt.xlabel('Время')
            plt.ylabel('Цена')
            plt.grid(True, alpha=0.3)
            
            # Форматируем метки оси X
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Удаляем дублирующиеся метки в легенде
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:  # Проверяем, что есть метки для легенды
                plt.legend(by_label.values(), by_label.keys(), loc='best')
            
            plt.tight_layout()
            
            # Сохраняем график в папку charts
            chart_file = os.path.join(charts_dir, f'price_chart_{SYMBOL}_{timeframe}.{CHART_FORMAT}')
            plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
            logger.info(f"График цены с отметками сделок сохранен в {chart_file}")
            plt.close()
        
        # Создаем интерактивный график с Plotly, если нужно
        if USE_INTERACTIVE and interactive_dir and HAS_PLOTLY:
            # Создаем DataFrame для свечей в формате Plotly
            df_plot = df.copy()
            
            # Создаем свечной график
            fig = go.Figure(data=[go.Candlestick(
                x=df_plot['time'],
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Цена'
            )])
            
            # Добавляем отметки для входов
            if entries is not None and not entries.empty:
                entries_in_range = entries[(entries['time'] >= min_date) & (entries['time'] <= max_date)]
                
                buy_entries = entries_in_range[entries_in_range['order'] == 'buy']
                if not buy_entries.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_entries['time'],
                        y=buy_entries['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green'
                        ),
                        name='Buy Entry'
                    ))
                
                sell_entries = entries_in_range[entries_in_range['order'] == 'sell']
                if not sell_entries.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_entries['time'],
                        y=sell_entries['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red'
                        ),
                        name='Sell Entry'
                    ))
            
            # Добавляем отметки для выходов
            if exits is not None and not exits.empty:
                exits_in_range = exits[(exits['time'] >= min_date) & (exits['time'] <= max_date)]
                
                # Группируем по типу выхода
                for exit_type, group in exits_in_range.groupby('type'):
                    marker_color = 'green' if exit_type == 'take_profit' else 'red' if exit_type == 'stop_loss' else 'blue'
                    marker_symbol = 'circle' if exit_type == 'take_profit' else 'x' if exit_type == 'stop_loss' else 'square'
                    
                    fig.add_trace(go.Scatter(
                        x=group['time'],
                        y=group['price'],
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            size=10,
                            color=marker_color
                        ),
                        name=f'{exit_type} Exit'
                    ))
            
            # Добавляем линии стоп-лоссов и тейк-профитов
            if stop_losses is not None and not stop_losses.empty:
                sl_in_range = stop_losses[(stop_losses['time'] >= min_date) & (stop_losses['time'] <= max_date)]
                for _, sl in sl_in_range.iterrows():
                    fig.add_shape(
                        type='line',
                        x0=sl['time'],
                        y0=sl['price'],
                        x1=max_date,
                        y1=sl['price'],
                        line=dict(
                            color='red',
                            width=1,
                            dash='dash'
                        ),
                        name='Stop Loss'
                    )
            
            if take_profits is not None and not take_profits.empty:
                tp_in_range = take_profits[(take_profits['time'] >= min_date) & (take_profits['time'] <= max_date)]
                for _, tp in tp_in_range.iterrows():
                    fig.add_shape(
                        type='line',
                        x0=tp['time'],
                        y0=tp['price'],
                        x1=max_date,
                        y1=tp['price'],
                        line=dict(
                            color='green',
                            width=1,
                            dash='dash'
                        ),
                        name='Take Profit'
                    )
            
            # Настраиваем макет
            fig.update_layout(
                title=f'{SYMBOL} ({timeframe}) с отметками сделок',
                xaxis_title='Время',
                yaxis_title='Цена',
                xaxis_rangeslider_visible=False,
                template='plotly_white'
            )
            
            # Сохраняем интерактивный график
            interactive_file = os.path.join(interactive_dir, f'interactive_price_{SYMBOL}_{timeframe}.html')
            fig.write_html(interactive_file, include_plotlyjs='cdn')
            logger.info(f"Интерактивный график цены сохранен в {interactive_file}")
        
        # Очищаем память
        del df
        gc.collect()
        
        logger.info(f"Построение графика цены для {timeframe} заняло {time.time() - start_time:.2f} секунд")
    
    except Exception as e:
        logger.error(f"Ошибка при построении графика цены для {timeframe}: {str(e)}")
        logger.error(traceback.format_exc())
        if 'plt' in locals() or 'plt' in globals():
            plt.close()  # Закрываем график в случае ошибки

def create_profit_factor_by_setup(results, data_dir, charts_dir, interactive_dir=None):
    """
    Создание таблицы с профит-фактором по каждому сетапу с оптимизацией
    
    Параметры:
    results (DataFrame): Результаты бэктеста
    data_dir (str): Путь к директории для сохранения данных
    charts_dir (str): Путь к директории для сохранения графиков
    interactive_dir (str, optional): Путь к директории для интерактивных графиков
    
    Возвращает:
    DataFrame: DataFrame со статистикой по сетапам
    """
    if results is None or results.empty:
        logger.warning("Нет данных для анализа профит-фактора")
        return None
    
    if 'setup' not in results.columns:
        logger.warning("В данных отсутствует информация о сетапах")
        return None
    
    start_time = time.time()
    
    try:
        # Убедимся, что результаты содержат копию данных, а не ссылку
        results_copy = results.copy()
        
        # Преобразуем категориальные колонки в строковые, чтобы избежать проблем с категориями
        if hasattr(results_copy['setup'].dtype, 'categories'):
            results_copy['setup'] = results_copy['setup'].astype(str)
        
        if hasattr(results_copy['result'].dtype, 'categories'):
            results_copy['result'] = results_copy['result'].astype(str)
        
        # Создаем DataFrame для статистики по сетапам
        # Используем GroupBy для эффективной агрегации с параметром observed=True
        setup_stats = results_copy.groupby('setup', observed=True).apply(lambda x: pd.Series({
            'trades': len(x),
            'wins': sum(x['result'] == 'win'),
            'losses': sum(x['result'] == 'loss'),
            'winrate': 100 * sum(x['result'] == 'win') / len(x) if len(x) > 0 else 0,
            'gross_profit': x.loc[x['profit'] > 0, 'profit'].sum() if not x.loc[x['profit'] > 0].empty else 0,
            'gross_loss': abs(x.loc[x['profit'] < 0, 'profit'].sum()) if not x.loc[x['profit'] < 0].empty else 0
        }), include_groups=False).reset_index()
        
        # Добавляем профит-фактор и другие метрики после GroupBy операции
        setup_stats['profit_factor'] = setup_stats.apply(
            lambda row: row['gross_profit'] / row['gross_loss'] 
                         if row['gross_loss'] > 0 else float('inf'),
            axis=1
        )
        
        # Вычисляем средние значения для всех категорий
        win_means = {}
        loss_means = {}
        max_wins = {}
        min_losses = {}
        
        # Вычисляем значения для каждой группы отдельно
        for setup in setup_stats['setup'].unique():
            setup_data = results_copy[results_copy['setup'] == setup]
            
            win_data = setup_data[setup_data['result'] == 'win']
            loss_data = setup_data[setup_data['result'] == 'loss']
            
            win_means[setup] = win_data['profit'].mean() if not win_data.empty else 0
            loss_means[setup] = loss_data['profit'].mean() if not loss_data.empty else 0
            max_wins[setup] = setup_data['profit'].max() if not setup_data.empty else 0
            min_losses[setup] = setup_data['profit'].min() if not setup_data.empty else 0
        
        # Применяем значения к DataFrame
        setup_stats['avg_win'] = setup_stats['setup'].apply(lambda x: win_means.get(x, 0))
        setup_stats['avg_loss'] = setup_stats['setup'].apply(lambda x: loss_means.get(x, 0))
        setup_stats['max_win'] = setup_stats['setup'].apply(lambda x: max_wins.get(x, 0))
        setup_stats['min_loss'] = setup_stats['setup'].apply(lambda x: min_losses.get(x, 0))
        
        # Вычисляем ожидание (expectancy)
        setup_stats['expectancy'] = setup_stats.apply(
            lambda row: (row['winrate']/100 * row['avg_win'] + (1-row['winrate']/100) * row['avg_loss']),
            axis=1
        )
        
        # Сортируем по профит-фактору
        setup_stats = setup_stats.sort_values(by='profit_factor', ascending=False)
        
        # Сохраняем в CSV
        stats_file = os.path.join(data_dir, f'setup_statistics_{SYMBOL}.csv')
        setup_stats.to_csv(stats_file, index=False)
        logger.info(f"Таблица со статистикой по сетапам сохранена в {stats_file}")
        
        # Строим визуализацию
        plt.figure(figsize=(16, 12))
        
        # 1. Профит-фактор по сетапам
        plt.subplot(2, 2, 1)
        bars = plt.bar(setup_stats['setup'], setup_stats['profit_factor'], color='purple', alpha=0.7)
        
        # Добавляем метки значений
        for bar in bars:
            height = bar.get_height()
            text_value = "∞" if height == float('inf') else f"{height:.2f}"
            plt.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                    text_value, ha='center', va='top', rotation=0, fontsize=9)
        
        plt.title('Профит-фактор по сетапам')
        plt.xlabel('Сетап')
        plt.ylabel('Профит-фактор')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.axhline(y=1, color='red', linestyle='--', label='Безубыточность')
        plt.legend()
        
        # 2. Количество сделок по сетапам
        plt.subplot(2, 2, 2)
        plt.bar(setup_stats['setup'], setup_stats['trades'], color='blue', alpha=0.7)
        plt.title('Количество сделок по сетапам')
        plt.xlabel('Сетап')
        plt.ylabel('Количество сделок')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Средний выигрыш/проигрыш по сетапам
        plt.subplot(2, 2, 3)
        ind = np.arange(len(setup_stats))
        width = 0.35
        
        plt.bar(ind - width/2, setup_stats['avg_win'], width, label='Средний выигрыш', color='green', alpha=0.7)
        plt.bar(ind + width/2, setup_stats['avg_loss'], width, label='Средний проигрыш', color='red', alpha=0.7)
        
        plt.title('Средний выигрыш/проигрыш по сетапам')
        plt.xlabel('Сетап')
        plt.ylabel('Прибыль/Убыток')
        plt.grid(True, alpha=0.3)
        plt.xticks(ind, setup_stats['setup'], rotation=45)
        plt.legend()
        
        # 4. Винрейт по сетапам
        plt.subplot(2, 2, 4)
        bars = plt.bar(setup_stats['setup'], setup_stats['winrate'], color='orange', alpha=0.7)
        
        # Добавляем метки значений
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                    f"{height:.1f}%", ha='center', va='top', rotation=0, fontsize=9)
        
        plt.title('Винрейт по сетапам (%)')
        plt.xlabel('Сетап')
        plt.ylabel('Винрейт (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Добавляем горизонтальную линию для среднего винрейта
        avg_winrate = results['result'].value_counts(normalize=True).get('win', 0) * 100
        plt.axhline(y=avg_winrate, color='red', linestyle='--', label=f'Средний винрейт: {avg_winrate:.1f}%')
        plt.legend()
        
        plt.tight_layout()
        
        # Сохраняем график в папку charts
        chart_file = os.path.join(charts_dir, f'setup_profit_analysis_{SYMBOL}.{CHART_FORMAT}')
        plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
        logger.info(f"График анализа прибыли по сетапам сохранен в {chart_file}")
        plt.close()
        
        # Создаем интерактивные графики, если нужно
        if USE_INTERACTIVE and interactive_dir and HAS_PLOTLY:
            # Создаем комбинированный график для анализа сетапов
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Профит-фактор по сетапам', 
                    'Количество сделок по сетапам',
                    'Средний выигрыш/проигрыш по сетапам', 
                    'Винрейт по сетапам (%)'
                ),
                vertical_spacing=0.12
            )
            
            # 1. Профит-фактор по сетапам
            profit_factor_values = []
            for pf in setup_stats['profit_factor']:
                if pf == float('inf'):
                    profit_factor_values.append(10)  # Ограничиваем бесконечность для отображения
                else:
                    profit_factor_values.append(min(pf, 10))  # Ограничиваем максимальное значение для лучшего отображения
            
            fig.add_trace(
                go.Bar(
                    x=setup_stats['setup'],
                    y=profit_factor_values,
                    text=[f"∞" if pf == float('inf') else f"{pf:.2f}" for pf in setup_stats['profit_factor']],
                    textposition='auto',
                    marker_color='purple',
                    name='Профит-фактор'
                ),
                row=1, col=1
            )
            
            # Линия безубыточности
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=1,
                x1=len(setup_stats['setup'])-0.5,
                y1=1,
                line=dict(color="red", width=2, dash="dash"),
                row=1, col=1
            )
            
            # 2. Количество сделок по сетапам
            fig.add_trace(
                go.Bar(
                    x=setup_stats['setup'],
                    y=setup_stats['trades'],
                    text=setup_stats['trades'],
                    textposition='auto',
                    marker_color='blue',
                    name='Количество сделок'
                ),
                row=1, col=2
            )
            
            # 3. Средний выигрыш/проигрыш по сетапам
            fig.add_trace(
                go.Bar(
                    x=setup_stats['setup'],
                    y=setup_stats['avg_win'],
                    name='Средний выигрыш',
                    marker_color='green'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=setup_stats['setup'],
                    y=setup_stats['avg_loss'],
                    name='Средний проигрыш',
                    marker_color='red'
                ),
                row=2, col=1
            )
            
            # 4. Винрейт по сетапам
            fig.add_trace(
                go.Bar(
                    x=setup_stats['setup'],
                    y=setup_stats['winrate'],
                    text=[f"{wr:.1f}%" for wr in setup_stats['winrate']],
                    textposition='auto',
                    marker_color='orange',
                    name='Винрейт'
                ),
                row=2, col=2
            )
            
            # Линия среднего винрейта
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=avg_winrate,
                x1=len(setup_stats['setup'])-0.5,
                y1=avg_winrate,
                line=dict(color="red", width=2, dash="dash"),
                row=2, col=2
            )
            
            # Настраиваем макет
            fig.update_layout(
                title=f'Анализ сетапов для {SYMBOL}',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Обновляем оси X для всех подграфиков
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(tickangle=45, row=i, col=j)
            
            # Сохраняем интерактивный график
            interactive_file = os.path.join(interactive_dir, f'interactive_setup_analysis_{SYMBOL}.html')
            fig.write_html(interactive_file, include_plotlyjs='cdn')
            logger.info(f"Интерактивный график анализа сетапов сохранен в {interactive_file}")
        
        # Очищаем память
        gc.collect()
        
        logger.info(f"Анализ профит-фактора по сетапам занял {time.time() - start_time:.2f} секунд")
        
        return setup_stats
    
    except Exception as e:
        logger.error(f"Ошибка при анализе профит-фактора по сетапам: {str(e)}")
        logger.error(traceback.format_exc())
        if 'plt' in locals() or 'plt' in globals():
            plt.close()
        return None

def create_monthly_analysis(results, data_dir, charts_dir, interactive_dir=None):
    """
    Создание анализа результатов по месяцам с оптимизацией производительности
    
    Параметры:
    results (DataFrame): Результаты бэктеста
    data_dir (str): Путь к директории для сохранения данных
    charts_dir (str): Путь к директории для сохранения графиков
    interactive_dir (str, optional): Путь к директории для интерактивных графиков
    
    Возвращает:
    DataFrame: DataFrame с месячной статистикой
    """
    if results is None or results.empty:
        logger.warning("Нет данных для месячного анализа")
        return None
    
    start_time = time.time()
    
    try:
        # Создаем локальную копию данных
        df = results.copy()
        
        # Добавляем столбцы для месяца и года
        df['year'] = df['entry_time'].dt.year
        df['month'] = df['entry_time'].dt.month
        
        # Группируем по году и месяцу с параметром observed=True
        monthly_stats = df.groupby(['year', 'month'], observed=True).agg(
            trades=('result', 'count'),
            wins=('result', lambda x: sum(x == 'win')),
            losses=('result', lambda x: sum(x == 'loss')),
            winrate=('result', lambda x: 100 * sum(x == 'win') / len(x) if len(x) > 0 else 0),
            profit=('profit', 'sum'),
            # Расчет профит-фактора
            gross_profit=('profit', lambda x: x[x > 0].sum() if any(x > 0) else 0),
            gross_loss=('profit', lambda x: abs(x[x < 0].sum()) if any(x < 0) else 0)
        ).reset_index()
        
        # Добавляем профит-фактор как отдельный шаг
        monthly_stats['profit_factor'] = monthly_stats.apply(
            lambda row: row['gross_profit'] / row['gross_loss'] 
                       if row['gross_loss'] > 0 else float('inf'), 
            axis=1
        )
        
        # Создаем столбец с названием месяца для отображения
        month_names = {1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн', 
                      7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'}
        monthly_stats['month_name'] = monthly_stats['month'].map(month_names)
        monthly_stats['period'] = monthly_stats.apply(lambda x: f"{x['year']}-{x['month_name']}", axis=1)
        
        # Сохраняем в CSV
        stats_file = os.path.join(data_dir, f'monthly_statistics_{SYMBOL}.csv')
        monthly_stats.to_csv(stats_file, index=False)
        logger.info(f"Таблица с месячной статистикой сохранена в {stats_file}")
        
        # Строим визуализацию
        plt.figure(figsize=(14, 14))
        
        # 1. Прибыль по месяцам
        plt.subplot(3, 1, 1)
        bars = plt.bar(monthly_stats['period'], monthly_stats['profit'], color='teal', alpha=0.7)
        
        # Добавляем метки значений
        for bar in bars:
            height = bar.get_height()
            color = 'green' if height >= 0 else 'red'
            plt.text(bar.get_x() + bar.get_width()/2., height * (0.95 if height >= 0 else 1.05),
                    f"{height:.2f}", ha='center', va='top' if height >= 0 else 'bottom', 
                    rotation=0, fontsize=8, color=color)
        
        plt.title('Прибыль по месяцам')
        plt.xlabel('Месяц')
        plt.ylabel('Прибыль')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Горизонтальная линия для нулевой прибыли
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
        
        # 2. Количество сделок по месяцам
        plt.subplot(3, 1, 2)
        bars = plt.bar(monthly_stats['period'], monthly_stats['trades'], color='navy', alpha=0.7)
        
        # Добавляем текст для выигрышей и проигрышей
        for i, bar in enumerate(bars):
            height = bar.get_height()
            wins = monthly_stats.iloc[i]['wins']
            losses = monthly_stats.iloc[i]['losses']
            plt.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f"W: {wins}\nL: {losses}", ha='center', va='center', 
                    rotation=0, fontsize=8, color='white')
        
        plt.title('Количество сделок по месяцам')
        plt.xlabel('Месяц')
        plt.ylabel('Количество сделок')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Винрейт по месяцам
        plt.subplot(3, 1, 3)
        bars = plt.bar(monthly_stats['period'], monthly_stats['winrate'], color='goldenrod', alpha=0.7)
        
        # Добавляем метки значений
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                    f"{height:.1f}%", ha='center', va='top', rotation=0, fontsize=8)
        
        plt.title('Винрейт по месяцам (%)')
        plt.xlabel('Месяц')
        plt.ylabel('Винрейт (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Добавляем горизонтальную линию для среднего винрейта
        avg_winrate = df['result'].value_counts(normalize=True).get('win', 0) * 100
        plt.axhline(y=avg_winrate, color='red', linestyle='--', label=f'Средний винрейт: {avg_winrate:.1f}%')
        plt.legend()
        
        plt.tight_layout()
        
        # Сохраняем график в папку charts
        chart_file = os.path.join(charts_dir, f'monthly_analysis_{SYMBOL}.{CHART_FORMAT}')
        plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
        logger.info(f"График месячного анализа сохранен в {chart_file}")
        plt.close()
        
        # Создаем интерактивный график, если нужно
        if USE_INTERACTIVE and interactive_dir and HAS_PLOTLY:
            # Создаем комбинированный график с подграфиками
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Прибыль по месяцам', 
                    'Количество сделок по месяцам', 
                    'Винрейт по месяцам (%)'
                ),
                vertical_spacing=0.1,
                row_heights=[0.33, 0.33, 0.33]
            )
            
            # 1. Прибыль по месяцам
            colors = ['green' if p >= 0 else 'red' for p in monthly_stats['profit']]
            
            fig.add_trace(
                go.Bar(
                    x=monthly_stats['period'],
                    y=monthly_stats['profit'],
                    text=[f"{p:.2f}" for p in monthly_stats['profit']],
                    textposition='auto',
                    marker_color=colors,
                    name='Прибыль'
                ),
                row=1, col=1
            )
            
            # Линия нулевой прибыли
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=0,
                x1=len(monthly_stats['period'])-0.5,
                y1=0,
                line=dict(color="red", width=1, dash="solid"),
                row=1, col=1
            )
            
            # 2. Количество сделок по месяцам
            fig.add_trace(
                go.Bar(
                    x=monthly_stats['period'],
                    y=monthly_stats['trades'],
                    text=[f"W:{w}<br>L:{l}" for w, l in zip(monthly_stats['wins'], monthly_stats['losses'])],
                    textposition='auto',
                    marker_color='navy',
                    name='Сделки'
                ),
                row=2, col=1
            )
            
            # 3. Винрейт по месяцам
            fig.add_trace(
                go.Bar(
                    x=monthly_stats['period'],
                    y=monthly_stats['winrate'],
                    text=[f"{wr:.1f}%" for wr in monthly_stats['winrate']],
                    textposition='auto',
                    marker_color='goldenrod',
                    name='Винрейт'
                ),
                row=3, col=1
            )
            
            # Линия среднего винрейта
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=avg_winrate,
                x1=len(monthly_stats['period'])-0.5,
                y1=avg_winrate,
                line=dict(color="red", width=2, dash="dash"),
                row=3, col=1
            )
            
            # Аннотация для среднего винрейта
            fig.add_annotation(
                x=len(monthly_stats['period'])-1,
                y=avg_winrate,
                text=f"Средний: {avg_winrate:.1f}%",
                showarrow=False,
                font=dict(color="red"),
                row=3, col=1
            )
            
            # Настраиваем макет
            fig.update_layout(
                title=f'Месячный анализ для {SYMBOL}',
                height=800,
                showlegend=False,
                template='plotly_white'
            )
            
            # Обновляем оси X для всех подграфиков
            for i in range(1, 4):
                fig.update_xaxes(tickangle=45, row=i, col=1)
            
            # Сохраняем интерактивный график
            interactive_file = os.path.join(interactive_dir, f'interactive_monthly_analysis_{SYMBOL}.html')
            fig.write_html(interactive_file, include_plotlyjs='cdn')
            logger.info(f"Интерактивный график месячного анализа сохранен в {interactive_file}")
        
        # Очищаем память
        del df
        gc.collect()
        
        logger.info(f"Анализ по месяцам занял {time.time() - start_time:.2f} секунд")
        
        return monthly_stats
    
    except Exception as e:
        logger.error(f"Ошибка при анализе по месяцам: {str(e)}")
        logger.error(traceback.format_exc())
        if 'plt' in locals() or 'plt' in globals():
            plt.close()  # Закрываем график в случае ошибки
        return None

def create_summary_file(results, data_dir, setup_stats=None, monthly_stats=None):
    """
    Создание файла с общей статистикой и подробным анализом
    
    Параметры:
    results (DataFrame): Результаты бэктеста
    data_dir (str): Путь к директории для сохранения данных
    setup_stats (DataFrame, optional): Статистика по сетапам
    monthly_stats (DataFrame, optional): Статистика по месяцам
    """
    if results is None or results.empty:
        logger.warning("Нет данных для создания общей статистики")
        return
    
    start_time = time.time()
    
    try:
        # Создаем копию результатов для безопасной работы
        results_copy = results.copy()
        
        # Преобразуем категориальные колонки в строковые для безопасной обработки
        for col in results_copy.select_dtypes(include=['category']).columns:
            results_copy[col] = results_copy[col].astype(str)
        
        # Рассчитываем общую статистику
        total_trades = len(results_copy)
        wins = sum(results_copy['result'] == 'win')
        losses = total_trades - wins
        winrate = 100 * wins / total_trades if total_trades > 0 else 0
        
        # Расчет прибыли и убытков
        gross_profit = results_copy.loc[results_copy['profit'] > 0, 'profit'].sum() if not results_copy.loc[results_copy['profit'] > 0].empty else 0
        gross_loss = abs(results_copy.loc[results_copy['profit'] < 0, 'profit'].sum()) if not results_copy.loc[results_copy['profit'] < 0].empty else 0
        net_profit = gross_profit - gross_loss
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Расчет профита на сделку
        avg_profit = results_copy['profit'].mean()
        avg_win = results_copy.loc[results_copy['result'] == 'win', 'profit'].mean() if not results_copy.loc[results_copy['result'] == 'win'].empty else 0
        avg_loss = results_copy.loc[results_copy['result'] == 'loss', 'profit'].mean() if not results_copy.loc[results_copy['result'] == 'loss'].empty else 0
        
        # Расчет максимальной просадки
        results_copy['cumulative_profit'] = results_copy['profit'].cumsum() + results_copy['balance'].iloc[0]
        results_copy['running_max'] = results_copy['cumulative_profit'].cummax()
        results_copy['drawdown'] = (results_copy['running_max'] - results_copy['cumulative_profit']) / results_copy['running_max'] * 100
        results_copy['drawdown_abs'] = results_copy['running_max'] - results_copy['cumulative_profit']
        max_drawdown = results_copy['drawdown'].max()
        max_drawdown_abs = results_copy['drawdown_abs'].max()
        
        # Расчет начального и конечного баланса
        initial_balance = results_copy['balance'].iloc[0] - results_copy['profit'].iloc[0]
        final_balance = results_copy['balance'].iloc[-1]
        
        # Расчет отношения выигрыша к проигрышу
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Расчет ожидания (expectancy)
        expectancy = (winrate/100 * avg_win + (1-winrate/100) * avg_loss)
        
        # Расчет статистики по последовательным выигрышам и проигрышам
        streak_data = results_copy['result'].copy()
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for result in streak_data:
            if result == 'win':
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        # Расчет времени торговли
        trading_period_str = "Нет данных"
        avg_trades_per_day = 0
        trading_start_str = "Нет данных"
        trading_end_str = "Нет данных"
        
        if 'entry_time' in results_copy.columns and 'exit_time' in results_copy.columns:
            # Убедимся, что даты имеют правильный тип
            try:
                # Преобразуем в datetime, если нужно
                if isinstance(results_copy['entry_time'].iloc[0], str):
                    results_copy['entry_time'] = pd.to_datetime(results_copy['entry_time'])
                if isinstance(results_copy['exit_time'].iloc[0], str):
                    results_copy['exit_time'] = pd.to_datetime(results_copy['exit_time'])
                
                trading_start = results_copy['entry_time'].min()
                trading_end = results_copy['exit_time'].max()
                
                if isinstance(trading_start, (datetime, pd.Timestamp)) and isinstance(trading_end, (datetime, pd.Timestamp)):
                    trading_period = trading_end - trading_start
                    trading_days = trading_period.days
                    
                    # Средние сделки в день
                    avg_trades_per_day = total_trades / max(trading_days, 1)
                    
                    # Время торговли в строковом формате
                    trading_period_str = f"{trading_days} дней"
                    if trading_days > 30:
                        trading_months = trading_days / 30.44  # Среднее количество дней в месяце
                        trading_period_str = f"{trading_days} дней ({trading_months:.1f} месяцев)"
                    
                    trading_start_str = trading_start.strftime('%Y-%m-%d')
                    trading_end_str = trading_end.strftime('%Y-%m-%d')
                else:
                    # Если даты не представлены в formattable типе
                    trading_start_str = str(trading_start)
                    trading_end_str = str(trading_end)
                    trading_period_str = "Не удалось вычислить период"
            except Exception as e:
                logger.warning(f"Ошибка при вычислении периода торговли: {str(e)}")
                trading_start_str = str(results_copy['entry_time'].min())
                trading_end_str = str(results_copy['exit_time'].max())
                trading_period_str = "Ошибка вычисления"
        
        # Создаем файл с общей статистикой
        with open(os.path.join(data_dir, f'summary_{SYMBOL}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"=== Общая статистика бэктеста для {SYMBOL} ===\n\n")
            
            if 'entry_time' in results_copy.columns and 'exit_time' in results_copy.columns:
                f.write(f"Период: {trading_start_str} - {trading_end_str} ({trading_period_str})\n\n")
            else:
                f.write(f"Период: {BACKTEST_START.strftime('%Y-%m-%d')} - {BACKTEST_END.strftime('%Y-%m-%d')}\n\n")
            
            f.write("--- Основные показатели ---\n")
            f.write(f"Всего сделок: {total_trades}\n")
            f.write(f"Выигрышных сделок: {wins} ({winrate:.2f}%)\n")
            f.write(f"Проигрышных сделок: {losses} ({100 - winrate:.2f}%)\n")
            f.write(f"Начальный баланс: {initial_balance:.2f}\n")
            f.write(f"Конечный баланс: {final_balance:.2f}\n")
            f.write(f"Чистая прибыль: {net_profit:.2f} ({(net_profit/initial_balance)*100:.2f}%)\n")
            
            if profit_factor == float('inf'):
                f.write(f"Профит-фактор: ∞ (нет убыточных сделок)\n")
            else:
                f.write(f"Профит-фактор: {profit_factor:.2f}\n")
                
            if win_loss_ratio == float('inf'):
                f.write(f"Отношение выигрыша к проигрышу: ∞ (нет проигрышных сделок)\n")
            else:
                f.write(f"Отношение выигрыша к проигрышу: {win_loss_ratio:.2f}\n")
                
            f.write(f"Ожидание (expectancy): {expectancy:.2f}\n")
            f.write(f"Максимальная просадка: {max_drawdown:.2f}% ({max_drawdown_abs:.2f})\n")
            f.write(f"Средняя прибыль на сделку: {avg_profit:.2f}\n\n")
            
            f.write("--- Серии сделок ---\n")
            f.write(f"Максимальная серия выигрышей: {max_win_streak}\n")
            f.write(f"Максимальная серия проигрышей: {max_loss_streak}\n")
            f.write(f"Среднее количество сделок в день: {avg_trades_per_day:.2f}\n\n")
            
            f.write("--- Детализация прибыли ---\n")
            f.write(f"Валовая прибыль: {gross_profit:.2f}\n")
            f.write(f"Валовой убыток: {gross_loss:.2f}\n")
            f.write(f"Средний выигрыш: {avg_win:.2f}\n")
            f.write(f"Средний проигрыш: {avg_loss:.2f}\n\n")
            
            # Анализ по типам ордеров с параметром observed=True
            if 'order' in results_copy.columns:
                f.write("--- Статистика по типам ордеров ---\n")
                order_stats = results_copy.groupby('order', observed=True).agg(
                    trades=('result', 'count'),
                    wins=('result', lambda x: sum(x == 'win')),
                    profit=('profit', 'sum')
                )
                
                for order_type, stats in order_stats.iterrows():
                    winrate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                    f.write(f"{order_type.capitalize()}: {stats['trades']} сделок, "
                            f"Винрейт: {winrate:.2f}%, Прибыль: {stats['profit']:.2f}\n")
                f.write("\n")
            
            # Анализ по таймфреймам с параметром observed=True
            if 'tf' in results_copy.columns:
                f.write("--- Статистика по таймфреймам ---\n")
                tf_stats = results_copy.groupby('tf', observed=True).agg(
                    trades=('result', 'count'),
                    wins=('result', lambda x: sum(x == 'win')),
                    profit=('profit', 'sum')
                )
                
                for tf, stats in tf_stats.iterrows():
                    winrate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                    f.write(f"{tf}: {stats['trades']} сделок, "
                            f"Винрейт: {winrate:.2f}%, Прибыль: {stats['profit']:.2f}\n")
                f.write("\n")
            
            # Статистика по сетапам, если доступна
            if 'setup' in results_copy.columns:
                f.write("--- Статистика по сетапам ---\n")
                
                if setup_stats is not None:
                    for _, row in setup_stats.iterrows():
                        setup = row['setup']
                        trades = row['trades']
                        winrate = row['winrate']
                        profit = results_copy[results_copy['setup'] == setup]['profit'].sum()
                        
                        # Профит-фактор, если доступен
                        pf = row.get('profit_factor', None)
                        if pf is not None:
                            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
                            f.write(f"{setup}: {trades} сделок, Винрейт: {winrate:.2f}%, "
                                    f"Прибыль: {profit:.2f}, PF: {pf_str}\n")
                        else:
                            f.write(f"{setup}: {trades} сделок, Винрейт: {winrate:.2f}%, "
                                    f"Прибыль: {profit:.2f}\n")
                else:
                    # Если нет готовой статистики по сетапам, генерируем ее
                    setup_group_stats = results_copy.groupby('setup', observed=True).agg(
                        trades=('result', 'count'),
                        wins=('result', lambda x: sum(x == 'win')),
                        profit=('profit', 'sum')
                    )
                    
                    for setup, stats in setup_group_stats.iterrows():
                        winrate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                        f.write(f"{setup}: {stats['trades']} сделок, "
                                f"Винрейт: {winrate:.2f}%, Прибыль: {stats['profit']:.2f}\n")
                f.write("\n")
            
            # Помесячная статистика, если доступна
            if monthly_stats is not None and not monthly_stats.empty:
                f.write("--- Помесячная статистика ---\n")
                f.write("Период         | Сделки | Винрейт | Прибыль   | Профит-фактор\n")
                f.write("---------------|--------|---------|-----------|-------------\n")
                
                for _, row in monthly_stats.iterrows():
                    period = row['period']
                    trades = row['trades']
                    winrate = row['winrate']
                    profit = row['profit']
                    pf = row['profit_factor']
                    
                    pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
                    
                    f.write(f"{period:14} | {trades:6d} | {winrate:6.2f}% | {profit:9.2f} | {pf_str:13}\n")
                
                # Добавляем итоговую строку
                f.write("---------------|--------|---------|-----------|-------------\n")
                f.write(f"{'ИТОГО':14} | {total_trades:6d} | {winrate:6.2f}% | {net_profit:9.2f} | {profit_factor if profit_factor != float('inf') else '∞':13}\n\n")
            
            # Заметки по улучшению стратегии
            f.write("--- Рекомендации по улучшению стратегии ---\n")
            
            if profit_factor < 1.0:
                f.write("- Стратегия убыточна (профит-фактор < 1.0). Рассмотрите полную переработку стратегии.\n")
            elif profit_factor < 1.5:
                f.write("- Профит-фактор ниже оптимального. Требуется оптимизация параметров и фильтров.\n")
            
            if winrate < 40:
                f.write("- Низкий винрейт. Рекомендуется улучшить фильтры для входа в рынок.\n")
            
            if max_drawdown > 20:
                f.write("- Высокая максимальная просадка. Рассмотрите улучшение управления рисками.\n")
            
            if max_loss_streak > 5:
                f.write("- Длинные серии убыточных сделок. Рекомендуется добавить фильтры для рыночных условий.\n")
            
            # Дополнительные рекомендации на основе статистики по сетапам
            if setup_stats is not None and not setup_stats.empty:
                # Находим лучшие и худшие сетапы
                best_setup = setup_stats.iloc[0]['setup']  # Лучший по профит-фактору
                worst_setups = setup_stats[setup_stats['profit_factor'] < 1.0]['setup'].tolist()
                
                if worst_setups:
                    f.write(f"- Убыточные сетапы: {', '.join(worst_setups)}. Рассмотрите их исключение или улучшение.\n")
                
                f.write(f"- Наиболее эффективный сетап: {best_setup}. Сфокусируйтесь на его использовании.\n")
            
            f.write("\nОтчет создан: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        logger.info(f"Файл с общей статистикой сохранен в {os.path.join(data_dir, f'summary_{SYMBOL}.txt')}")
        logger.info(f"Создание файла статистики заняло {time.time() - start_time:.2f} секунд")
    
    except Exception as e:
        logger.error(f"Ошибка при создании файла статистики: {str(e)}")
        logger.error(traceback.format_exc())

def visualize_backtest(max_rows=None, timeframes_to_plot=None, start_date=None, end_date=None):
    """
    Основная функция для визуализации результатов бэктеста с оптимизацией памяти и производительности
    
    Параметры:
    max_rows (int, optional): Максимальное количество строк для загрузки из CSV
    timeframes_to_plot (list, optional): Список таймфреймов для построения графиков цены
    start_date (datetime, str, optional): Начальная дата для анализа
    end_date (datetime, str, optional): Конечная дата для анализа
    """
    logger.info("\n=== Начало визуализации результатов бэктеста ===\n")
    
    global_start_time = time.time()
    
    # Если не указаны таймфреймы для графиков, используем стандартные
    if timeframes_to_plot is None:
        timeframes_to_plot = ["M5", "M15", "H1"]
    
    try:
        # Шаг 1: Создаем директории для результатов
        results_dir, data_dir, charts_dir, interactive_dir = create_results_directory()
        
        # Шаг 2: Находим файлы с результатами бэктеста
        result_file, entries_file, exits_file, sl_file, tp_file = find_latest_backtest_files()
        
        if result_file is None:
            logger.error("Не удалось найти результаты бэктеста. Визуализация прервана.")
            return
        
        # Шаг 3: Копируем файлы в директорию данных
        files_to_copy = [result_file, entries_file, exits_file, sl_file, tp_file]
        new_file_paths = copy_files_to_data_dir(files_to_copy, data_dir)
        
        # Шаг 4: Загружаем данные
        logger.info("Загрузка данных бэктеста...")
        results, entries, exits, stop_losses, take_profits = load_backtest_data(data_dir, max_rows)
        
        if results is None:
            logger.error("Не удалось загрузить данные бэктеста. Визуализация прервана.")
            return
        
        logger.info(f"Загружено {len(results)} записей результатов бэктеста.")
        
        # Проверяем использование памяти
        check_memory_usage()
        
        # Преобразуем даты в datetime объекты, если они являются строками
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Неверный формат даты start_date: {start_date}. Используется None.")
                start_date = None
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()
        
        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Неверный формат даты end_date: {end_date}. Используется None.")
                end_date = None
        elif isinstance(end_date, pd.Timestamp):
            end_date = end_date.to_pydatetime()
        
        # Изменяем параллельную обработку для графиков - создаем данные параллельно, но строим графики последовательно
        if PARALLEL_PROCESSING:
            logger.info("Использование параллельной обработки для данных, но последовательного создания графиков...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Подготовка данных для анализа
                # 1. Копируем результаты для безопасной работы
                results_copy = results.copy()
                
                # 2. Преобразуем категориальные колонки
                for col in results_copy.select_dtypes(include=['category']).columns:
                    results_copy[col] = results_copy[col].astype(str)
                
                # 3. Подготовка данных для сетапов
                setup_data_future = None
                if 'setup' in results_copy.columns:
                    setup_data_future = executor.submit(
                        lambda: {
                            setup: {
                                'trades': len(group),
                                'wins': sum(group['result'] == 'win'),
                                'losses': sum(group['result'] == 'loss'),
                                'winrate': 100 * sum(group['result'] == 'win') / len(group) if len(group) > 0 else 0,
                                'gross_profit': group.loc[group['profit'] > 0, 'profit'].sum() if not group.loc[group['profit'] > 0].empty else 0,
                                'gross_loss': abs(group.loc[group['profit'] < 0, 'profit'].sum()) if not group.loc[group['profit'] < 0].empty else 0
                            }
                            for setup, group in results_copy.groupby('setup')
                        }
                    )
                
                # 4. Подготовка данных для месячной статистики
                monthly_data_future = None
                if 'entry_time' in results_copy.columns:
                    # Преобразуем к datetime, если нужно
                    if not isinstance(results_copy['entry_time'].iloc[0], (datetime, pd.Timestamp)):
                        results_copy['entry_time'] = pd.to_datetime(results_copy['entry_time'])
                    
                    # Добавляем колонки для месяца и года
                    results_copy['year'] = results_copy['entry_time'].dt.year
                    results_copy['month'] = results_copy['entry_time'].dt.month
                    
                    # Вычисляем статистику по месяцам
                    monthly_data_future = executor.submit(
                        lambda: results_copy.groupby(['year', 'month'], observed=True).agg(
                            trades=('result', 'count'),
                            wins=('result', lambda x: sum(x == 'win')),
                            losses=('result', lambda x: sum(x == 'loss')),
                            winrate=('result', lambda x: 100 * sum(x == 'win') / len(x) if len(x) > 0 else 0),
                            profit=('profit', 'sum'),
                            gross_profit=('profit', lambda x: x[x > 0].sum() if any(x > 0) else 0),
                            gross_loss=('profit', lambda x: abs(x[x < 0].sum()) if any(x < 0) else 0)
                        ).reset_index()
                    )
                
                # 5. Определяем диапазон дат для загрузки ценовых данных
                if start_date is None and 'entry_time' in results.columns:
                    if isinstance(results['entry_time'].iloc[0], (datetime, pd.Timestamp)):
                        start_date = results['entry_time'].min()
                    elif isinstance(results['entry_time'].iloc[0], str):
                        try:
                            start_date = pd.to_datetime(results['entry_time'].min())
                        except:
                            start_date = None
                
                if end_date is None and 'exit_time' in results.columns:
                    if isinstance(results['exit_time'].iloc[0], (datetime, pd.Timestamp)):
                        end_date = results['exit_time'].max()
                    elif isinstance(results['exit_time'].iloc[0], str):
                        try:
                            end_date = pd.to_datetime(results['exit_time'].max())
                        except:
                            end_date = None
                
                # 6. Загружаем данные цен
                price_data_future = executor.submit(
                    get_price_data, start_date, end_date, SYMBOL, timeframes_to_plot
                )
                
                # 7. Получаем результаты обработки данных
                price_data = price_data_future.result()
                
                setup_data = None
                if setup_data_future:
                    setup_data_raw = setup_data_future.result()
                    
                    # Преобразуем словарь в DataFrame
                    setup_stats = pd.DataFrame.from_dict(setup_data_raw, orient='index').reset_index()
                    setup_stats.columns = ['setup', 'trades', 'wins', 'losses', 'winrate', 'gross_profit', 'gross_loss']
                    
                    # Добавляем профит-фактор
                    setup_stats['profit_factor'] = setup_stats.apply(
                        lambda row: row['gross_profit'] / row['gross_loss'] 
                                     if row['gross_loss'] > 0 else float('inf'),
                        axis=1
                    )
                    
                    # Сортируем по профит-фактору
                    setup_stats = setup_stats.sort_values(by='profit_factor', ascending=False)
                else:
                    setup_stats = None
                
                monthly_stats = None
                if monthly_data_future:
                    monthly_stats_raw = monthly_data_future.result()
                    
                    if monthly_stats_raw is not None and not monthly_stats_raw.empty:
                        # Добавляем профит-фактор
                        monthly_stats_raw['profit_factor'] = monthly_stats_raw.apply(
                            lambda row: row['gross_profit'] / row['gross_loss'] 
                                       if row['gross_loss'] > 0 else float('inf'), 
                            axis=1
                        )
                        
                        # Создаем столбец с названием месяца для отображения
                        month_names = {1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн', 
                                      7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'}
                        monthly_stats_raw['month_name'] = monthly_stats_raw['month'].map(month_names)
                        monthly_stats_raw['period'] = monthly_stats_raw.apply(
                            lambda x: f"{x['year']}-{x['month_name']}",
                            axis=1
                        )
                        
                        monthly_stats = monthly_stats_raw
                
                # Теперь последовательно создаем графики и отчеты
                logger.info("Создание графиков и отчетов...")
                
                # 8. Создаем основные графики
                plot_balance_equity(results, charts_dir, interactive_dir)
                plot_drawdown(results, charts_dir, interactive_dir)
                plot_trades_distribution(results, charts_dir, interactive_dir)
                
                # 9. Создаем подробный анализ по сетапам, если есть данные
                if setup_stats is not None:
                    create_profit_factor_by_setup(results, data_dir, charts_dir, interactive_dir)
                
                # 10. Создаем месячный анализ, если есть данные
                if monthly_stats is not None:
                    create_monthly_analysis(results, data_dir, charts_dir, interactive_dir)
                
                # 11. Создаем итоговый отчет
                create_summary_file(results, data_dir, setup_stats, monthly_stats)
                
                # 12. Строим графики цены с отметками сделок, если есть данные цен
                if price_data:
                    logger.info("Построение графиков цены с отметками сделок...")
                    for tf in tqdm(timeframes_to_plot, desc="Графики цены"):
                        if tf in price_data:
                            plot_price_with_trades(
                                price_data, entries, exits, stop_losses, take_profits, 
                                tf, charts_dir, interactive_dir
                            )
                    
                    # Очищаем память
                    del price_data
                    check_memory_usage()
        else:
            # Последовательное выполнение всех шагов
            logger.info("Последовательная обработка для визуализации...")
            
            # Создаем основные графики
            logger.info("Создание основных графиков...")
            plot_balance_equity(results, charts_dir, interactive_dir)
            plot_drawdown(results, charts_dir, interactive_dir)
            plot_trades_distribution(results, charts_dir, interactive_dir)
            
            # Анализируем результаты по сетапам и месяцам
            logger.info("Анализ результатов по сетапам и месяцам...")
            setup_stats = create_profit_factor_by_setup(results, data_dir, charts_dir, interactive_dir)
            monthly_stats = create_monthly_analysis(results, data_dir, charts_dir, interactive_dir)
            
            # Создаем итоговый отчет
            create_summary_file(results, data_dir, setup_stats, monthly_stats)
            
            # Загружаем данные цен и строим графики с отметками сделок
            logger.info("Загрузка данных цен для построения графиков...")
            
            # Преобразуем даты в datetime, если они представлены в другом формате
            if start_date is None and 'entry_time' in results.columns:
                if isinstance(results['entry_time'].iloc[0], (datetime, pd.Timestamp)):
                    start_date = results['entry_time'].min()
                elif isinstance(results['entry_time'].iloc[0], str):
                    try:
                        start_date = pd.to_datetime(results['entry_time'].min())
                    except:
                        start_date = None
            
            if end_date is None and 'exit_time' in results.columns:
                if isinstance(results['exit_time'].iloc[0], (datetime, pd.Timestamp)):
                    end_date = results['exit_time'].max()
                elif isinstance(results['exit_time'].iloc[0], str):
                    try:
                        end_date = pd.to_datetime(results['exit_time'].max())
                    except:
                        end_date = None
            
            price_data = get_price_data(start_date, end_date, SYMBOL, timeframes_to_plot)
            
            if price_data:
                logger.info("Построение графиков цены с отметками сделок...")
                for tf in tqdm(timeframes_to_plot, desc="Графики цены"):
                    if tf in price_data:
                        plot_price_with_trades(
                            price_data, entries, exits, stop_losses, take_profits, 
                            tf, charts_dir, interactive_dir
                        )
                
                # Очищаем память
                del price_data
                check_memory_usage()
        
        # Освобождаем память
        del results, entries, exits, stop_losses, take_profits
        gc.collect()
        
        total_time = time.time() - global_start_time
        logger.info(f"\n=== Визуализация завершена за {total_time:.2f} секунд! ===\n")
        logger.info(f"Результаты сохранены в директории: {results_dir}")
        logger.info(f"Чтобы просмотреть графики, откройте папку: {charts_dir}")
        logger.info(f"Чтобы просмотреть данные статистики, откройте папку: {data_dir}")
        
        if USE_INTERACTIVE and interactive_dir:
            logger.info(f"Интерактивные графики сохранены в: {interactive_dir}")
        
        print(f"\n=== Визуализация завершена за {total_time:.2f} секунд! ===")
        print(f"Результаты сохранены в директории: {results_dir}")
        print(f"Чтобы просмотреть графики, откройте папку: {charts_dir}")
        print(f"Чтобы просмотреть данные статистики, откройте: {os.path.join(data_dir, f'summary_{SYMBOL}.txt')}")
    
    except Exception as e:
        logger.error(f"Критическая ошибка при визуализации результатов: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Произошла ошибка при визуализации: {str(e)}")
        print("Подробности смотрите в логе.")

if __name__ == "__main__":
    # Можно передать параметры для ограничения объема обрабатываемых данных
    import argparse
    
    parser = argparse.ArgumentParser(description="Визуализация результатов бэктеста")
    parser.add_argument("--max_rows", type=int, default=None, help="Максимальное количество строк для загрузки из CSV")
    parser.add_argument("--timeframes", type=str, default="M5,M15,H1", help="Список таймфреймов через запятую для графиков цены")
    parser.add_argument("--start_date", type=str, default=None, help="Начальная дата для анализа (ГГГГ-ММ-ДД)")
    parser.add_argument("--end_date", type=str, default=None, help="Конечная дата для анализа (ГГГГ-ММ-ДД)")
    parser.add_argument("--parallel", action="store_true", help="Использовать параллельную обработку")
    parser.add_argument("--interactive", action="store_true", help="Создавать интерактивные графики")
    parser.add_argument("--dpi", type=int, default=CHART_DPI, help="Разрешение графиков (DPI)")
    parser.add_argument("--format", type=str, default=CHART_FORMAT, help="Формат сохранения графиков (png, svg, pdf)")
    
    args = parser.parse_args()
    
    # Применяем параметры
    if args.parallel:
        PARALLEL_PROCESSING = True
    
    if args.interactive:
        USE_INTERACTIVE = True
    
    CHART_DPI = args.dpi
    CHART_FORMAT = args.format
    
    timeframes_to_plot = args.timeframes.split(",")
    
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Неверный формат даты: {args.start_date}. Используется дата по умолчанию.")
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print(f"Неверный формат даты: {args.end_date}. Используется дата по умолчанию.")
    
    # Запускаем визуализацию
    visualize_backtest(
        max_rows=args.max_rows,
        timeframes_to_plot=timeframes_to_plot,
        start_date=start_date,
        end_date=end_date
    )