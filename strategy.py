import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from config import SYMBOL, TIMEFRAMES, FVG_CONFIRMATION, STRATEGY_SETTINGS
from data_fetcher import get_historical_data
from mt5_connector import connect_mt5, disconnect_mt5, is_connected

# Настройка логгера
logger = logging.getLogger(__name__)

# Константы для улучшения читаемости и производительности
SWING_LOOKBACK_DEFAULT = 5
ORDER_BLOCK_WINDOW_DEFAULT = 20
BREAKER_BLOCK_WINDOW_DEFAULT = 20
FVG_GAP_THRESHOLD_DEFAULT = 0.0003
LIQUIDITY_GRAB_LOOKBACK_DEFAULT = 10
EQ_TOLERANCE_DEFAULT = 0.0002

# Кэш для промежуточных расчетов
_pattern_cache = {}

def clear_pattern_cache():
    """Очистка кэша паттернов"""
    global _pattern_cache
    _pattern_cache.clear()
    logger.info("Кэш паттернов очищен")

@lru_cache(maxsize=128)
def find_swing_points(high_values, low_values, lookback=SWING_LOOKBACK_DEFAULT):
    """
    Оптимизированный поиск точек разворота (swing high/low) на графике
    
    Параметры:
    high_values (tuple): Кортеж значений High
    low_values (tuple): Кортеж значений Low
    lookback (int): Период для поиска разворотов
    
    Возвращает:
    tuple: (highs, lows) - списки точек разворота
    """
    if not high_values or not low_values:
        return [], []
        
    size = len(high_values)
    if size < 2 * lookback + 1:
        return [], []
    
    highs = []
    lows = []
    
    # Используем векторизованный подход для улучшения производительности
    high_array = np.array(high_values)
    low_array = np.array(low_values)
    
    for i in range(lookback, size - lookback):
        # Проверка на максимум
        left_max = np.max(high_array[i-lookback:i])
        right_max = np.max(high_array[i+1:i+lookback+1])
        
        if high_array[i] > left_max and high_array[i] > right_max:
            highs.append((i, high_array[i]))
        
        # Проверка на минимум
        left_min = np.min(low_array[i-lookback:i])
        right_min = np.min(low_array[i+1:i+lookback+1])
        
        if low_array[i] < left_min and low_array[i] < right_min:
            lows.append((i, low_array[i]))
    
    return highs, lows

def preprocess_dataframe(df):
    """
    Предварительная обработка DataFrame для улучшения производительности
    
    Параметры:
    df (DataFrame): Исходный DataFrame с данными
    
    Возвращает:
    DataFrame: Обработанный DataFrame с дополнительными колонками
    """
    if df is None or len(df) < 10:
        return df
    
    # Создаем копию только если необходимо изменять DataFrame
    if 'body_size' not in df.columns or 'candle_type' not in df.columns:
        df = df.copy()
        
        # Размер тела свечи
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Тип свечи (бычья/медвежья)
        df['candle_type'] = np.where(df['close'] >= df['open'], 1, -1)
        
        # Размер верхней тени
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Размер нижней тени
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Размер всей свечи
        df['candle_size'] = df['high'] - df['low']
        
        # Отношение тела к размеру свечи
        df['body_ratio'] = df['body_size'] / df['candle_size'].replace(0, np.nan)
        df['body_ratio'] = df['body_ratio'].fillna(0)
        
        # Предварительный расчет ключевых индикаторов
        if len(df) >= 20:
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['atr20'] = (
                (df['high'] - df['low']).rolling(window=20).mean() + 
                (abs(df['high'] - df['close'].shift(1))).rolling(window=20).mean() + 
                (abs(df['low'] - df['close'].shift(1))).rolling(window=20).mean()
            ) / 3
        
        # Индикатор импульса (отношение изменения цены к среднему диапазону)
        df['momentum'] = (df['close'] - df['close'].shift(3)) / df['candle_size'].rolling(window=5).mean()
    
    return df

def detect_order_block(df, window=ORDER_BLOCK_WINDOW_DEFAULT, threshold_multiplier=1.5):
    """
    Улучшенный поиск Order Block на основе концепции ICT с параметрами качества
    
    Параметры:
    df (DataFrame): DataFrame с данными
    window (int): Размер окна для анализа
    threshold_multiplier (float): Множитель для определения значимости импульса
    
    Возвращает:
    dict: Информация о найденном Order Block или None
    """
    if df is None or len(df) < window + 2:
        logger.warning("detect_order_block: Недостаточно данных для анализа.")
        return None
    
    # Оптимизация: используем кэширование для одних и тех же данных
    cache_key = f"ob_{hash(tuple(df.iloc[-window:]['close']))}"
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]
    
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    # Берем только данные в рамках окна для анализа
    recent_df = df.iloc[-window:]
    
    # Проверяем тренд последних свечей (используем линейную регрессию для определения направления)
    x = np.arange(5)
    y = recent_df.iloc[-5:]['close'].values
    slope = np.polyfit(x, y, 1)[0] if len(y) == 5 else 0
    recent_trend = "up" if slope > 0 else "down"
    
    # Рассчитываем средний размер свечи для определения значимости импульса
    avg_candle_size = recent_df['candle_size'].mean()
    threshold = avg_candle_size * threshold_multiplier
    
    result = None
    
    # Проверяем на order block для покупки
    if recent_trend == "up":
        # Ищем последнюю медвежью свечу перед импульсом вверх
        for i in range(len(recent_df) - 3, 0, -1):
            # Убедимся, что индекс в допустимом диапазоне
            if i + 3 >= len(recent_df):
                continue
                
            # Медвежья свеча
            if recent_df.iloc[i]['candle_type'] == -1:
                # Проверяем, был ли после нее бычий импульс
                next_3_candles = recent_df.iloc[i+1:i+4]
                price_change = next_3_candles['close'].max() - recent_df.iloc[i]['close']
                
                # Импульс должен быть значительным
                if price_change > threshold:
                    # Проверяем объем тела свечи для качества сигнала
                    body_ratio = recent_df.iloc[i]['body_ratio']
                    
                    # Хороший Order Block должен иметь заметное тело
                    if body_ratio > 0.4:
                        mitigation_price = recent_df.iloc[i]['low']
                        
                        # Проверяем, что после импульса цена не вернулась обратно в зону OB
                        if not ((next_3_candles['low'] < mitigation_price).any()):
                            result = {
                                "type": "buy", 
                                "level": mitigation_price,
                                "strength": min(1.0, price_change / threshold),  # Относительная сила сигнала
                                "index": recent_df.index[i]
                            }
                            break
    
    # Проверяем на order block для продажи
    elif recent_trend == "down":
        # Ищем последнюю бычью свечу перед импульсом вниз
        for i in range(len(recent_df) - 3, 0, -1):
            # Убедимся, что индекс в допустимом диапазоне
            if i + 3 >= len(recent_df):
                continue
                
            # Бычья свеча
            if recent_df.iloc[i]['candle_type'] == 1:
                # Проверяем, был ли после нее медвежий импульс
                next_3_candles = recent_df.iloc[i+1:i+4]
                price_change = recent_df.iloc[i]['close'] - next_3_candles['close'].min()
                
                # Импульс должен быть значительным
                if price_change > threshold:
                    # Проверяем объем тела свечи для качества сигнала
                    body_ratio = recent_df.iloc[i]['body_ratio']
                    
                    # Хороший Order Block должен иметь заметное тело
                    if body_ratio > 0.4:
                        mitigation_price = recent_df.iloc[i]['high']
                        
                        # Проверяем, что после импульса цена не вернулась обратно в зону OB
                        if not ((next_3_candles['high'] > mitigation_price).any()):
                            result = {
                                "type": "sell", 
                                "level": mitigation_price,
                                "strength": min(1.0, price_change / threshold),  # Относительная сила сигнала
                                "index": recent_df.index[i]
                            }
                            break
    
    # Сохраняем результат в кэш
    _pattern_cache[cache_key] = result
    return result

def detect_breaker_block(df, window=BREAKER_BLOCK_WINDOW_DEFAULT):
    """
    Улучшенное определение Breaker Block с оптимизацией производительности
    
    Параметры:
    df (DataFrame): DataFrame с данными
    window (int): Размер окна для анализа
    
    Возвращает:
    dict: Информация о найденном Breaker Block или None
    """
    if df is None or len(df) < window + 2:
        return None
    
    # Оптимизация: используем кэширование для одних и тех же данных
    cache_key = f"bb_{hash(tuple(df.iloc[-window:]['close']))}"
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]
    
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    recent_df = df.iloc[-window:]
    
    # Конвертируем в кортежи для оптимизации
    high_values = tuple(recent_df['high'].values)
    low_values = tuple(recent_df['low'].values)
    
    # Находим swing high/low
    highs, lows = find_swing_points(high_values, low_values, lookback=3)
    
    result = None
    
    if len(highs) >= 2 and len(lows) >= 2:
        # Проверяем бычий Breaker Block (продажа от пробитого уровня поддержки)
        if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:  # Восходящий тренд
            # Проверяем, вернулась ли цена к пробитому уровню
            if (recent_df.iloc[-1]['close'] < highs[-2][1] and 
                recent_df.iloc[-2]['high'] > highs[-2][1]):
                # Дополнительная проверка: пробой должен быть значимым
                breakout_size = highs[-1][1] - highs[-2][1]
                avg_candle_size = recent_df['candle_size'].mean()
                
                if breakout_size > avg_candle_size * 0.7:  # Пробой должен быть больше 70% средней свечи
                    result = {
                        "type": "sell", 
                        "level": highs[-2][1],
                        "strength": min(1.0, breakout_size / (avg_candle_size * 2)),  # Относительная сила сигнала
                        "breakout_size": breakout_size
                    }
        
        # Проверяем медвежий Breaker Block (покупка от пробитого уровня сопротивления)
        elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:  # Нисходящий тренд
            # Проверяем, вернулась ли цена к пробитому уровню
            if (recent_df.iloc[-1]['close'] > lows[-2][1] and 
                recent_df.iloc[-2]['low'] < lows[-2][1]):
                # Дополнительная проверка: пробой должен быть значимым
                breakout_size = lows[-2][1] - lows[-1][1]
                avg_candle_size = recent_df['candle_size'].mean()
                
                if breakout_size > avg_candle_size * 0.7:  # Пробой должен быть больше 70% средней свечи
                    result = {
                        "type": "buy", 
                        "level": lows[-2][1],
                        "strength": min(1.0, breakout_size / (avg_candle_size * 2)),  # Относительная сила сигнала
                        "breakout_size": breakout_size
                    }
    
    # Сохраняем результат в кэш
    _pattern_cache[cache_key] = result
    return result

def detect_fvg(df, gap_threshold=FVG_GAP_THRESHOLD_DEFAULT):
    """
    Оптимизированный поиск Fair Value Gap (FVG) с дополнительными проверками
    
    Параметры:
    df (DataFrame): DataFrame с данными
    gap_threshold (float): Минимальный размер гэпа для FVG
    
    Возвращает:
    dict: Информация о найденном FVG или None
    """
    if df is None or len(df) < 3:
        return None
    
    # Оптимизация: используем кэширование для одних и тех же данных
    cache_key = f"fvg_{hash(tuple(df.iloc[-10:]['close']))}"
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]
    
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    # Адаптивный порог на основе средней волатильности
    if 'atr20' in df.columns and not df['atr20'].isna().all():
        adaptive_threshold = df['atr20'].iloc[-1] * 0.3  # 30% от ATR
        gap_threshold = max(gap_threshold, adaptive_threshold)
    
    result = None
    
    # Ищем бычий FVG (эффективнее с векторизацией)
    for i in range(2, min(10, len(df))):
        # Убедимся, что индекс в допустимом диапазоне
        if len(df) <= i + 1:
            continue
            
        if df.iloc[-i]['low'] > df.iloc[-i-2]['high'] + gap_threshold:
            # FVG размер должен быть значимым
            fvg_size = df.iloc[-i]['low'] - df.iloc[-i-2]['high']
            
            # Проверяем, что FVG еще не заполнен
            not_filled = df.iloc[-i+1:]['low'].min() > df.iloc[-i-2]['high']
            
            if not_filled:
                # Центр FVG для размещения ордера
                level = df.iloc[-i-2]['high'] + fvg_size / 2
                
                # Дополнительная проверка для качества сигнала: 
                # FVG должен формироваться в направлении тренда
                trend_aligned = False
                
                # Проверяем тренд (можно использовать предварительно рассчитанную SMA)
                if 'sma20' in df.columns and not df['sma20'].isna().all():
                    trend_aligned = df.iloc[-i]['close'] > df['sma20'].iloc[-i]
                else:
                    # Простая проверка на тренд, если SMA не рассчитана
                    trend_aligned = df.iloc[-i]['close'] > df.iloc[-i-5]['close'] if i+5 < len(df) else True
                
                if trend_aligned:
                    result = {
                        "type": "buy", 
                        "level": level,
                        "size": fvg_size,
                        "strength": min(1.0, fvg_size / (gap_threshold * 3))  # Относительная сила сигнала
                    }
                    break
    
    # Если бычий FVG не найден, ищем медвежий
    if result is None:
        for i in range(2, min(10, len(df))):
            # Убедимся, что индекс в допустимом диапазоне
            if len(df) <= i + 1:
                continue
                
            if df.iloc[-i]['high'] < df.iloc[-i-2]['low'] - gap_threshold:
                # FVG размер должен быть значимым
                fvg_size = df.iloc[-i-2]['low'] - df.iloc[-i]['high']
                
                # Проверяем, что FVG еще не заполнен
                not_filled = df.iloc[-i+1:]['high'].max() < df.iloc[-i-2]['low']
                
                if not_filled:
                    # Центр FVG для размещения ордера
                    level = df.iloc[-i-2]['low'] - fvg_size / 2
                    
                    # Дополнительная проверка для качества сигнала: 
                    # FVG должен формироваться в направлении тренда
                    trend_aligned = False
                    
                    # Проверяем тренд (можно использовать предварительно рассчитанную SMA)
                    if 'sma20' in df.columns and not df['sma20'].isna().all():
                        trend_aligned = df.iloc[-i]['close'] < df['sma20'].iloc[-i]
                    else:
                        # Простая проверка на тренд, если SMA не рассчитана
                        trend_aligned = df.iloc[-i]['close'] < df.iloc[-i-5]['close'] if i+5 < len(df) else True
                    
                    if trend_aligned:
                        result = {
                            "type": "sell", 
                            "level": level,
                            "size": fvg_size,
                            "strength": min(1.0, fvg_size / (gap_threshold * 3))  # Относительная сила сигнала
                        }
                        break
    
    # Сохраняем результат в кэш
    _pattern_cache[cache_key] = result
    return result

def detect_liquidity_grab(df, lookback=LIQUIDITY_GRAB_LOOKBACK_DEFAULT):
    """
    Оптимизированная функция поиска захвата ликвидности (Liquidity Grab)
    
    Параметры:
    df (DataFrame): DataFrame с данными
    lookback (int): Период для анализа
    
    Возвращает:
    dict: Информация о найденном захвате ликвидности или None
    """
    if df is None or len(df) < lookback + 5:
        return None
    
    # Оптимизация: используем кэширование для одних и тех же данных
    cache_key = f"lg_{hash(tuple(df.iloc[-lookback-5:]['close']))}"
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]
    
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    # Создаем явную копию только нужного участка DataFrame
    recent_df = df.iloc[-lookback-5:].copy()
    
    # Считаем максимумы и минимумы за период (оптимизировано)
    prev_high = recent_df.iloc[:-5]['high'].max()
    prev_low = recent_df.iloc[:-5]['low'].min()
    
    # Получаем последние 5 свечей
    last_5_candles = recent_df.iloc[-5:]
    
    result = None
    
    # Проверяем на захват ликвидности вверху (для продажи)
    if (last_5_candles['high'] > prev_high).any() and last_5_candles.iloc[-1]['close'] < prev_high:
        # Должен быть резкий разворот
        highest_idx = last_5_candles['high'].idxmax()
        highest_pos = last_5_candles.index.get_loc(highest_idx)
        
        # Проверяем, произошел ли разворот достаточно быстро (в течение 3-х свечей)
        if highest_pos <= 2:  # В первых трех свечах из пяти
            # Дополнительные проверки для качества сигнала
            price_change = last_5_candles.iloc[-1]['close'] - prev_high
            
            # Разворот должен быть значительным
            if abs(price_change) > last_5_candles['candle_size'].mean() * 0.5:
                # Объем (если доступен) должен быть выше среднего
                volume_condition = True
                if 'tick_volume' in last_5_candles.columns:
                    avg_volume = last_5_candles['tick_volume'].mean()
                    volume_at_high = last_5_candles.iloc[highest_pos]['tick_volume']
                    volume_condition = volume_at_high > avg_volume
                
                if volume_condition:
                    result = {
                        "type": "sell", 
                        "level": prev_high,
                        "strength": min(1.0, abs(price_change) / (last_5_candles['candle_size'].mean() * 1.5))
                    }
    
    # Если не найден захват сверху, проверяем на захват ликвидности внизу (для покупки)
    if result is None and (last_5_candles['low'] < prev_low).any() and last_5_candles.iloc[-1]['close'] > prev_low:
        # Должен быть резкий разворот
        lowest_idx = last_5_candles['low'].idxmin()
        lowest_pos = last_5_candles.index.get_loc(lowest_idx)
        
        # Проверяем, произошел ли разворот достаточно быстро (в течение 3-х свечей)
        if lowest_pos <= 2:  # В первых трех свечах из пяти
            # Дополнительные проверки для качества сигнала
            price_change = last_5_candles.iloc[-1]['close'] - prev_low
            
            # Разворот должен быть значительным
            if abs(price_change) > last_5_candles['candle_size'].mean() * 0.5:
                # Объем (если доступен) должен быть выше среднего
                volume_condition = True
                if 'tick_volume' in last_5_candles.columns:
                    avg_volume = last_5_candles['tick_volume'].mean()
                    volume_at_low = last_5_candles.iloc[lowest_pos]['tick_volume']
                    volume_condition = volume_at_low > avg_volume
                
                if volume_condition:
                    result = {
                        "type": "buy", 
                        "level": prev_low,
                        "strength": min(1.0, abs(price_change) / (last_5_candles['candle_size'].mean() * 1.5))
                    }
    
    # Сохраняем результат в кэш
    _pattern_cache[cache_key] = result
    return result

def detect_equal_highs_lows(df, lookback=20, tolerance=EQ_TOLERANCE_DEFAULT):
    """
    Оптимизированное определение равных максимумов/минимумов (EQH/EQL)
    
    Параметры:
    df (DataFrame): DataFrame с данными
    lookback (int): Период для анализа
    tolerance (float): Допустимое отклонение для равенства
    
    Возвращает:
    dict: Информация о найденном паттерне или None
    """
    if df is None or len(df) < lookback:
        return None
    
    # Оптимизация: используем кэширование для одних и тех же данных
    cache_key = f"eq_{hash(tuple(df.iloc[-lookback:]['close']))}"
    if cache_key in _pattern_cache:
        return _pattern_cache[cache_key]
    
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    # Создаем копию только необходимой части данных
    recent_df = df.iloc[-lookback:].copy()
    
    # Улучшенный способ определения локальных экстремумов с помощью векторизации
    # Определение локальных максимумов
    high_series = recent_df['high']
    high_is_local_max = (high_series > high_series.shift(1)) & (high_series > high_series.shift(2)) & \
                         (high_series > high_series.shift(-1)) & (high_series > high_series.shift(-2))
    local_highs = high_series[high_is_local_max]
    
    # Определение локальных минимумов
    low_series = recent_df['low']
    low_is_local_min = (low_series < low_series.shift(1)) & (low_series < low_series.shift(2)) & \
                        (low_series < low_series.shift(-1)) & (low_series < low_series.shift(-2))
    local_lows = low_series[low_is_local_min]
    
    result = None
    
    # Ищем EQH (Equal Highs) - для продажи
    if len(local_highs) >= 2:
        # Преобразуем в массив для более быстрых вычислений
        high_values = local_highs.values
        
        # Ищем пары близких максимумов
        for i in range(len(high_values)):
            for j in range(i+1, len(high_values)):
                if abs(high_values[i] - high_values[j]) < tolerance:
                    # Найдены равные максимумы
                    # Проверяем, что текущая цена находится под уровнем
                    if recent_df.iloc[-1]['close'] < high_values[i]:
                        # Дополнительная проверка: цена должна недавно отбиться от уровня
                        recent_test = False
                        for k in range(1, min(5, len(recent_df))):
                            if recent_df.iloc[-k]['high'] > high_values[i] - tolerance and \
                               recent_df.iloc[-k]['high'] < high_values[i] + tolerance:
                                recent_test = True
                                break
                        
                        if recent_test:
                            level = (high_values[i] + high_values[j]) / 2  # Средний уровень
                            result = {
                                "type": "sell", 
                                "level": level,
                                "strength": 0.7  # Фиксированная сила сигнала для этого паттерна
                            }
                            break
            if result:
                break
    
    # Если не найдены EQH, ищем EQL (Equal Lows) - для покупки
    if result is None and len(local_lows) >= 2:
        # Преобразуем в массив для более быстрых вычислений
        low_values = local_lows.values
        
        # Ищем пары близких минимумов
        for i in range(len(low_values)):
            for j in range(i+1, len(low_values)):
                if abs(low_values[i] - low_values[j]) < tolerance:
                    # Найдены равные минимумы
                    # Проверяем, что текущая цена находится над уровнем
                    if recent_df.iloc[-1]['close'] > low_values[i]:
                        # Дополнительная проверка: цена должна недавно отбиться от уровня
                        recent_test = False
                        for k in range(1, min(5, len(recent_df))):
                            if recent_df.iloc[-k]['low'] < low_values[i] + tolerance and \
                               recent_df.iloc[-k]['low'] > low_values[i] - tolerance:
                                recent_test = True
                                break
                        
                        if recent_test:
                            level = (low_values[i] + low_values[j]) / 2  # Средний уровень
                            result = {
                                "type": "buy", 
                                "level": level,
                                "strength": 0.7  # Фиксированная сила сигнала для этого паттерна
                            }
                            break
            if result:
                break
    
    # Сохраняем результат в кэш
    _pattern_cache[cache_key] = result
    return result

def find_optimal_entry(df, signal_type, base_level, atr_multiplier=0.5):
    """
    Улучшенный поиск оптимальной точки входа рядом с базовым уровнем
    
    Параметры:
    df (DataFrame): DataFrame с данными
    signal_type (str): Тип сигнала ('buy' или 'sell')
    base_level (float): Базовый уровень
    atr_multiplier (float): Множитель ATR для определения зоны входа
    
    Возвращает:
    float: Оптимальный уровень входа
    """
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    if signal_type == "buy":
        # Рассчитываем ATR, если его еще нет
        if 'atr20' in df.columns and not df['atr20'].isna().all():
            atr = df['atr20'].iloc[-1]
        else:
            # Простой расчет ATR, если он не был предварительно вычислен
            atr = df['candle_size'].tail(20).mean()
        
        # Для покупки ищем оптимальный вход немного ниже базового уровня
        # Используем уровень поддержки и фибоначчи
        recent_low = df.iloc[-10:]['low'].min()
        recent_high = df.iloc[-20:-10]['high'].max()
        
        # Находим уровень Фибоначчи 0.382 от последнего движения вниз
        fib_level = recent_low + (recent_high - recent_low) * 0.382
        
        # Смотрим на зону поддержки
        support_level = base_level - atr * atr_multiplier
        
        # Определяем оптимальный вход как наибольший из уровней
        return max(support_level, fib_level)
    else:  # sell
        # Рассчитываем ATR, если его еще нет
        if 'atr20' in df.columns and not df['atr20'].isna().all():
            atr = df['atr20'].iloc[-1]
        else:
            # Простой расчет ATR, если он не был предварительно вычислен
            atr = df['candle_size'].tail(20).mean()
        
        # Для продажи ищем оптимальный вход немного выше базового уровня
        # Используем уровень сопротивления и фибоначчи
        recent_high = df.iloc[-10:]['high'].max()
        recent_low = df.iloc[-20:-10]['low'].min()
        
        # Находим уровень Фибоначчи 0.382 от последнего движения вверх
        fib_level = recent_high - (recent_high - recent_low) * 0.382
        
        # Смотрим на зону сопротивления
        resistance_level = base_level + atr * atr_multiplier
        
        # Определяем оптимальный вход как наименьший из уровней
        return min(resistance_level, fib_level)

def calculate_stop_loss(df, signal_type, entry_level, default_pips=30, min_atr_multiplier=1.5):
    """
    Улучшенный расчет уровня стоп-лосса на основе структуры рынка
    
    Параметры:
    df (DataFrame): DataFrame с данными
    signal_type (str): Тип сигнала ('buy' или 'sell')
    entry_level (float): Уровень входа
    default_pips (int): Минимальный стоп-лосс в пипсах
    min_atr_multiplier (float): Минимальный множитель ATR для стоп-лосса
    
    Возвращает:
    float: Размер стоп-лосса в валюте
    """
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    # Минимальный стоп-лосс в валюте
    default_sl = default_pips * 0.0001
    
    # Рассчитываем ATR, если его еще нет
    if 'atr20' in df.columns and not df['atr20'].isna().all():
        atr = df['atr20'].iloc[-1]
    else:
        # Простой расчет ATR, если он не был предварительно вычислен
        atr = df['candle_size'].tail(20).mean()
    
    # Минимальный стоп-лосс на основе ATR
    min_sl_atr = atr * min_atr_multiplier
    
    # Обеспечиваем минимальный размер стоп-лосса
    min_sl = max(default_sl, min_sl_atr)
    
    if signal_type == "buy":
        # Для покупки ищем ближайший локальный минимум ниже точки входа
        # Используем скользящее окно для определения локальных минимумов
        window_size = 5  # Размер окна для поиска локальных минимумов
        
        local_mins = []
        for i in range(window_size, len(df)):
            if all(df.iloc[i-j]['low'] >= df.iloc[i]['low'] for j in range(1, window_size+1) if i-j >= 0) and \
               all(df.iloc[i+j]['low'] >= df.iloc[i]['low'] for j in range(1, window_size+1) if i+j < len(df)):
                local_mins.append(df.iloc[i]['low'])
        
        # Фильтруем минимумы ниже точки входа
        valid_mins = [low for low in local_mins if low < entry_level]
        
        if valid_mins:
            # Находим ближайший минимум
            closest_min = max(valid_mins)
            
            # Добавляем буфер для надежности
            sl_level = closest_min - 0.0002
            sl_distance = entry_level - sl_level
            
            # Не меньше минимального стоп-лосса
            return max(sl_distance, min_sl)
        
        # Если не нашли подходящий минимум, используем стандартный подход
        # Ищем минимум за последние 10 свечей
        recent_min = df.iloc[-10:]['low'].min()
        if recent_min < entry_level:
            sl_level = recent_min - 0.0002  # Добавляем буфер
            sl_distance = entry_level - sl_level
            return max(sl_distance, min_sl)
        
        return min_sl
    
    else:  # sell
        # Для продажи ищем ближайший локальный максимум выше точки входа
        # Используем скользящее окно для определения локальных максимумов
        window_size = 5  # Размер окна для поиска локальных максимумов
        
        local_maxs = []
        for i in range(window_size, len(df)):
            if all(df.iloc[i-j]['high'] <= df.iloc[i]['high'] for j in range(1, window_size+1) if i-j >= 0) and \
               all(df.iloc[i+j]['high'] <= df.iloc[i]['high'] for j in range(1, window_size+1) if i+j < len(df)):
                local_maxs.append(df.iloc[i]['high'])
        
        # Фильтруем максимумы выше точки входа
        valid_maxs = [high for high in local_maxs if high > entry_level]
        
        if valid_maxs:
            # Находим ближайший максимум
            closest_max = min(valid_maxs)
            
            # Добавляем буфер для надежности
            sl_level = closest_max + 0.0002
            sl_distance = sl_level - entry_level
            
            # Не меньше минимального стоп-лосса
            return max(sl_distance, min_sl)
        
        # Если не нашли подходящий максимум, используем стандартный подход
        # Ищем максимум за последние 10 свечей
        recent_max = df.iloc[-10:]['high'].max()
        if recent_max > entry_level:
            sl_level = recent_max + 0.0002  # Добавляем буфер
            sl_distance = sl_level - entry_level
            return max(sl_distance, min_sl)
        
        return min_sl

def find_trade_signal(df, min_signal_strength=0.7):
    """
    Улучшенная стратегия ICT/Smart Money для поиска точек входа с оценкой силы сигнала
    
    Параметры:
    df (DataFrame): DataFrame с данными
    min_signal_strength (float): Минимальная сила сигнала (0.0-1.0) для генерации торгового сигнала
    
    Возвращает:
    dict: Информация о сигнале или None
    """
    if df is None or df.empty or len(df) < 30:
        logger.warning("find_trade_signal: Недостаточно данных для анализа.")
        return None
    
    # Предварительная обработка данных
    df = preprocess_dataframe(df)
    
    # Шаг 1: Определение основного тренда с помощью линейной регрессии
    x = np.arange(20)
    y = df.iloc[-20:]['close'].values
    slope, intercept = np.polyfit(x, y, 1)
    trend = "up" if slope > 0 else "down"
    
    # Шаг 2: Проверка различных сетапов в порядке приоритета
    
    # 1. Сначала проверяем Order Block (один из основных компонентов Silver Bullet)
    order_block = detect_order_block(df)
    if order_block and order_block.get('strength', 0) >= min_signal_strength:
        # Проверяем, что сигнал соответствует тренду или это разворотный сетап
        if (trend == "up" and order_block["type"] == "buy") or (trend == "down" and order_block["type"] == "sell"):
            logger.info(f"Найден Order Block: {order_block}")
            
            # Оптимизируем вход
            entry_level = find_optimal_entry(df, order_block["type"], order_block["level"])
            stop_loss = calculate_stop_loss(df, order_block["type"], entry_level)
            
            # Соотношение риск/доходность из настроек или по умолчанию 1:2.5
            risk_reward = STRATEGY_SETTINGS.get("OrderBlock", {}).get("risk_reward_ratio", 2.5)
            take_profit = stop_loss * risk_reward
            
            return {
                "type": order_block["type"],
                "level": entry_level,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "setup": "OrderBlock",
                "strength": order_block.get("strength", 0.8)
            }
    
    # 2. Проверяем FVG (Fair Value Gap) с подтверждением
    fvg = detect_fvg(df)
    if fvg and fvg.get('strength', 0) >= min_signal_strength:
        logger.info(f"Найден FVG: {fvg}")
        
        # FVG должен быть подтвержден дополнительными факторами
        confirmed = False
        
        # Проверяем захват ликвидности как подтверждение
        liquidity_grab = detect_liquidity_grab(df)
        if liquidity_grab and liquidity_grab["type"] == fvg["type"]:
            confirmed = True
        
        # Альтернативно, проверяем подтверждение по объему, если доступен
        elif 'tick_volume' in df.columns:
            # Проверяем, есть ли увеличение объема при формировании FVG
            recent_volume = df.iloc[-3:]['tick_volume'].mean()
            avg_volume = df.iloc[-10:-3]['tick_volume'].mean()
            volume_increase = recent_volume > avg_volume * 1.2  # 20% увеличение
            
            if volume_increase:
                confirmed = True
        
        # Если есть подтверждение и паттерн совпадает с трендом
        if confirmed and ((trend == "up" and fvg["type"] == "buy") or (trend == "down" and fvg["type"] == "sell")):
            entry_level = fvg["level"]
            stop_loss = calculate_stop_loss(df, fvg["type"], entry_level)
            
            # Соотношение риск/доходность из настроек или по умолчанию 1:3
            risk_reward = STRATEGY_SETTINGS.get("FVG", {}).get("risk_reward_ratio", 3.0)
            take_profit = stop_loss * risk_reward
            
            return {
                "type": fvg["type"],
                "level": entry_level,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "setup": "FVG_with_Confirmation",
                "strength": fvg.get("strength", 0.85)
            }
    
    # 3. Проверяем Breaker Block (важный сетап в ICT)
    breaker = detect_breaker_block(df)
    if breaker and breaker.get('strength', 0) >= min_signal_strength:
        logger.info(f"Найден Breaker Block: {breaker}")
        
        entry_level = breaker["level"]
        stop_loss = calculate_stop_loss(df, breaker["type"], entry_level, default_pips=40)
        
        # Соотношение риск/доходность из настроек или по умолчанию 1:3.5
        risk_reward = STRATEGY_SETTINGS.get("BreakerBlock", {}).get("risk_reward_ratio", 3.5)
        take_profit = stop_loss * risk_reward
        
        return {
            "type": breaker["type"],
            "level": entry_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "setup": "BreakerBlock",
            "strength": breaker.get("strength", 0.9)
        }
    
    # 4. Проверяем захват ликвидности (Liquidity Grab) как самостоятельный сетап
    liquidity_grab = detect_liquidity_grab(df)
    if liquidity_grab and liquidity_grab.get('strength', 0) >= min_signal_strength:
        logger.info(f"Найден Liquidity Grab: {liquidity_grab}")
        
        # Для захвата ликвидности требуется подтверждение объемом или другими факторами
        # Проверяем наличие свечи поглощения
        last_candles = df.iloc[-2:]
        confirmed = False
        
        # Проверка на свечу поглощения
        if liquidity_grab["type"] == "buy":
            if (last_candles.iloc[1]['close'] > last_candles.iloc[0]['open'] and 
                last_candles.iloc[1]['open'] < last_candles.iloc[0]['close']):
                confirmed = True
        else:  # sell
            if (last_candles.iloc[1]['close'] < last_candles.iloc[0]['open'] and 
                last_candles.iloc[1]['open'] > last_candles.iloc[0]['close']):
                confirmed = True
        
        # Дополнительная проверка по импульсу
        if 'momentum' in df.columns:
            momentum = df.iloc[-1]['momentum']
            if (liquidity_grab["type"] == "buy" and momentum > 0.5) or (liquidity_grab["type"] == "sell" and momentum < -0.5):
                confirmed = True
                
        if confirmed:
            entry_level = liquidity_grab["level"]
            stop_loss = calculate_stop_loss(df, liquidity_grab["type"], entry_level, default_pips=35)
            
            # Соотношение риск/доходность из настроек или по умолчанию 1:2.8
            risk_reward = STRATEGY_SETTINGS.get("LiquidityGrab", {}).get("risk_reward_ratio", 2.8)
            take_profit = stop_loss * risk_reward
            
            return {
                "type": liquidity_grab["type"],
                "level": entry_level,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "setup": "LiquidityGrab_with_Confirmation",
                "strength": liquidity_grab.get("strength", 0.75)
            }
    
    # 5. Проверяем равные максимумы/минимумы (EQH/EQL)
    eq_levels = detect_equal_highs_lows(df)
    if eq_levels and eq_levels.get('strength', 0) >= min_signal_strength:
        logger.info(f"Найдены Equal Highs/Lows: {eq_levels}")
        
        entry_level = eq_levels["level"]
        stop_loss = calculate_stop_loss(df, eq_levels["type"], entry_level, default_pips=25)
        
        # Соотношение риск/доходность из настроек или по умолчанию 1:2
        risk_reward = STRATEGY_SETTINGS.get("EqualHighLow", {}).get("risk_reward_ratio", 2.0)
        take_profit = stop_loss * risk_reward
        
        return {
            "type": eq_levels["type"],
            "level": entry_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "setup": "EqualHighLow",
            "strength": eq_levels.get("strength", 0.7)
        }
    
    # Если ни один сетап не найден
    return None

if __name__ == "__main__":
    # Настраиваем логирование для консоли при запуске скрипта напрямую
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    if is_connected() or connect_mt5():
        df = get_historical_data(SYMBOL, timeframe="M15")
        
        print("\nАнализ паттернов:")
        
        # Проверяем Order Block
        order_block = detect_order_block(df)
        print(f"Order Block: {order_block}")
        
        # Проверяем Breaker Block
        breaker = detect_breaker_block(df)
        print(f"Breaker Block: {breaker}")
        
        # Проверяем FVG
        fvg = detect_fvg(df)
        print(f"FVG: {fvg}")
        
        # Проверяем захват ликвидности
        liquidity_grab = detect_liquidity_grab(df)
        print(f"Liquidity Grab: {liquidity_grab}")
        
        # Проверяем равные максимумы/минимумы
        eq_levels = detect_equal_highs_lows(df)
        print(f"Equal Highs/Lows: {eq_levels}")
        
        print("\nИтоговый сигнал:")
        signal = find_trade_signal(df)
        if signal:
            print(f"Найден сигнал: {signal}")
            print(f"Тип: {signal['type']}")
            print(f"Цена входа: {signal['level']}")
            print(f"Стоп-лосс: {signal['stop_loss']} ({signal['stop_loss']/0.0001:.0f} пипсов)")
            print(f"Тейк-профит: {signal['take_profit']} ({signal['take_profit']/0.0001:.0f} пипсов)")
            print(f"Сетап: {signal['setup']}")
            print(f"Сила сигнала: {signal.get('strength', 'не определена')}")
        else:
            print("Сигналов нет")
            
        # Очистка кэша
        clear_pattern_cache()
        disconnect_mt5()
    else:
        print("Не удалось подключиться к MT5")