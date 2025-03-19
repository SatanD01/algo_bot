import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import os
from datetime import datetime, timedelta
from config import SYMBOL, TIMEFRAMES, CANDLES_FOR_EACH_TF, BACKTEST_START, BACKTEST_END
from mt5_connector import connect_mt5, disconnect_mt5

# Директория для кэширования данных
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(cache_dir, exist_ok=True)

# Настроим логирование
logger = logging.getLogger(__name__)

# Словарь таймфреймов MT5
MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# Словарь для кэширования данных в памяти
_data_cache = {}

def clear_cache():
    """Очистка кэша данных в памяти"""
    global _data_cache
    _data_cache.clear()
    logger.info("Кэш данных в памяти очищен")

def get_cache_file_path(symbol, timeframe, start_date, end_date):
    """
    Формирует путь к файлу кэша
    
    Параметры:
    symbol (str): Торговый символ
    timeframe (str): Таймфрейм
    start_date (datetime): Начальная дата
    end_date (datetime): Конечная дата
    
    Возвращает:
    str: Путь к файлу кэша
    """
    start_str = start_date.strftime("%Y%m%d") if start_date else "start"
    end_str = end_date.strftime("%Y%m%d") if end_date else "end"
    return os.path.join(cache_dir, f"{symbol}_{timeframe}_{start_str}_{end_str}.pkl")

def is_cache_valid(cache_file, max_age_hours=24):
    """
    Проверяет актуальность кэша
    
    Параметры:
    cache_file (str): Путь к файлу кэша
    max_age_hours (int): Максимальный возраст кэша в часах
    
    Возвращает:
    bool: True, если кэш актуален, иначе False
    """
    if not os.path.exists(cache_file):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    age = datetime.now() - file_time
    
    return age.total_seconds() < max_age_hours * 3600

def get_historical_data(symbol=SYMBOL, timeframe="M5", start_date=None, end_date=None, 
                      num_candles=None, use_cache=True, force_reload=False):
    """
    Получение исторических данных за заданный период или количество свечей
    с использованием кэширования и разбиением на части для больших периодов.
    
    Параметры:
    symbol (str): Торговый символ (например, "EURUSD")
    timeframe (str): Таймфрейм ("M1", "M5", "M15", "H1", "H4", "D1")
    start_date (datetime): Начальная дата для данных
    end_date (datetime): Конечная дата для данных
    num_candles (int): Количество свечей (используется, если start_date и end_date не указаны)
    use_cache (bool): Использовать ли кэширование данных
    force_reload (bool): Принудительная загрузка данных даже при наличии кэша
    
    Возвращает:
    DataFrame: Данные в формате pandas DataFrame или None в случае ошибки
    """
    # Генерируем ключ для кэша
    cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}_{num_candles}"
    
    # Проверяем кэш в памяти, если разрешено
    if use_cache and not force_reload and cache_key in _data_cache:
        logger.debug(f"Использую кэш в памяти для {symbol} ({timeframe})")
        return _data_cache[cache_key].copy()
    
    # Формируем имя файла кэша
    cache_file = get_cache_file_path(symbol, timeframe, start_date, end_date)
    
    # Проверяем кэш в файловой системе, если разрешено
    if use_cache and not force_reload and is_cache_valid(cache_file):
        try:
            logger.info(f"Загрузка данных из кэша: {cache_file}")
            df = pd.read_pickle(cache_file)
            
            # Сохраняем в кэш памяти для ускорения последующих запросов
            _data_cache[cache_key] = df.copy()
            
            return df
        except Exception as e:
            logger.warning(f"Ошибка при чтении кэша: {str(e)}")
            # Если не удалось прочитать кэш, загружаем данные заново
    
    # Проверяем, что MT5 подключен
    mt5_connected = False
    try:
        if not mt5.terminal_info():
            logger.info("MT5 не подключен. Попытка подключения...")
            mt5_connected = connect_mt5()
            if not mt5_connected:
                logger.error("Не удалось подключиться к MT5")
                return None
        else:
            mt5_connected = True
    except Exception as e:
        logger.error(f"Ошибка при проверке состояния MT5: {str(e)}")
        return None
    
    # Получаем код таймфрейма из словаря
    if timeframe not in MT5_TIMEFRAMES:
        logger.error(f"Неизвестный таймфрейм: {timeframe}")
        return None
    
    mt5_timeframe = MT5_TIMEFRAMES[timeframe]
    
    try:
        df = None
        
        if num_candles is not None:
            # Получение данных по количеству свечей
            logger.info(f"Запрос {num_candles} свечей для {symbol} ({timeframe})")
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
        else:
            # Получение данных за период с разбиением на части для больших периодов
            if start_date is None:
                start_date = BACKTEST_START
            if end_date is None:
                end_date = BACKTEST_END
                
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Определяем максимальный размер части в зависимости от таймфрейма
            if timeframe in ["M1", "M5"]:
                max_chunk_days = 7  # Для небольших таймфреймов берем по неделе
            elif timeframe in ["M15", "M30"]:
                max_chunk_days = 14  # Для средних таймфреймов по две недели
            else:
                max_chunk_days = 30  # Для крупных таймфреймов по месяцу
            
            max_chunk_seconds = max_chunk_days * 24 * 60 * 60
            
            logger.info(f"Запрос исторических данных: {symbol} ({timeframe}) с {start_date} по {end_date}")
            
            # Разбиваем большой период на части
            chunks_dfs = []
            current_start = start_timestamp
            
            while current_start < end_timestamp:
                current_end = min(current_start + max_chunk_seconds, end_timestamp)
                
                chunk_start_date = datetime.fromtimestamp(current_start)
                chunk_end_date = datetime.fromtimestamp(current_end)
                
                logger.debug(f"Загрузка части данных с {chunk_start_date} по {chunk_end_date}")
                
                # Делаем несколько попыток загрузки данных
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        chunk_rates = mt5.copy_rates_range(
                            symbol, mt5_timeframe,
                            datetime.fromtimestamp(current_start), 
                            datetime.fromtimestamp(current_end)
                        )
                        
                        if chunk_rates is not None and len(chunk_rates) > 0:
                            chunk_df = pd.DataFrame(chunk_rates)
                            chunks_dfs.append(chunk_df)
                            break
                        else:
                            if attempt < max_attempts - 1:
                                logger.warning(f"Пустой результат для {chunk_start_date}-{chunk_end_date}. Повторная попытка {attempt+2}/{max_attempts}...")
                                time.sleep(1)  # Ждем перед следующей попыткой
                            else:
                                logger.warning(f"Пустой результат для {chunk_start_date}-{chunk_end_date} после {max_attempts} попыток")
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            logger.warning(f"Ошибка при загрузке части данных (попытка {attempt+1}/{max_attempts}): {str(e)}")
                            time.sleep(1)  # Ждем перед следующей попыткой
                        else:
                            logger.error(f"Не удалось загрузить часть данных после {max_attempts} попыток: {str(e)}")
                
                # Переходим к следующей части
                current_start = current_end
            
            # Объединяем все части
            if chunks_dfs:
                df = pd.concat(chunks_dfs, ignore_index=True)
                
                # Удаляем дубликаты, которые могут возникнуть на границах частей
                df = df.drop_duplicates(subset=['time'])
        
        if df is None or df.empty:
            error_code = mt5.last_error()
            logger.error(f"Ошибка получения данных для {symbol} ({timeframe}): {error_code}")
            return None
        
        # Преобразуем данные
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        # Добавляем дополнительные столбцы для улучшения производительности
        # Например, идентификатор бычьей/медвежьей свечи
        df['candle_type'] = np.where(df['close'] >= df['open'], 1, -1)  # 1 для бычьих, -1 для медвежьих
        
        # Предварительно вычисляем некоторые индикаторы для оптимизации
        if len(df) > 20:
            df['sma20'] = df['close'].rolling(window=20).mean()
        if len(df) > 50:
            df['sma50'] = df['close'].rolling(window=50).mean()
        
        # Преобразуем индекс в DatetimeIndex для быстрого доступа
        df = df.set_index('time', drop=False)
        
        logger.info(f"Получено {len(df)} свечей для {symbol} ({timeframe})")
        
        # Сохраняем в кэш файловой системы, если разрешено
        if use_cache:
            try:
                df.to_pickle(cache_file)
                logger.info(f"Данные сохранены в кэш: {cache_file}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить данные в кэш: {str(e)}")
        
        # Сохраняем в кэш памяти
        _data_cache[cache_key] = df.copy()
        
        # Сбрасываем индекс перед возвратом для совместимости с остальным кодом
        return df.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Исключение при получении данных для {symbol} ({timeframe}): {str(e)}")
        return None
    
    finally:
        # Отключаемся от MT5, если мы его подключали в этой функции
        if mt5_connected and not mt5_connected:
            disconnect_mt5()

def get_multi_timeframe_data(symbol=SYMBOL, timeframes=None, start_date=None, end_date=None, use_cache=True):
    """
    Получение данных для нескольких таймфреймов за один запрос
    
    Параметры:
    symbol (str): Торговый символ
    timeframes (list): Список таймфреймов. Если None, используется TIMEFRAMES из конфига
    start_date (datetime): Начальная дата
    end_date (datetime): Конечная дата
    use_cache (bool): Использовать ли кэширование
    
    Возвращает:
    dict: Словарь {таймфрейм: DataFrame} или None в случае ошибки
    """
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    result = {}
    
    # Проверяем подключение к MT5 один раз для всех запросов
    mt5_connected = False
    try:
        if not mt5.terminal_info():
            logger.info("MT5 не подключен. Попытка подключения...")
            mt5_connected = connect_mt5()
            if not mt5_connected:
                logger.error("Не удалось подключиться к MT5")
                return None
        else:
            mt5_connected = True
    except Exception as e:
        logger.error(f"Ошибка при проверке состояния MT5: {str(e)}")
        return None
    
    try:
        for tf in timeframes:
            # Используем num_candles из конфига, если он есть, иначе None
            num_candles = CANDLES_FOR_EACH_TF.get(tf) if start_date is None and end_date is None else None
            
            df = get_historical_data(
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                num_candles=num_candles,
                use_cache=use_cache
            )
            
            if df is not None:
                result[tf] = df
            else:
                logger.warning(f"Не удалось получить данные для {symbol} {tf}")
        
        return result
    
    finally:
        # Отключаемся от MT5, если мы его подключали в этой функции
        if mt5_connected and not mt5_connected:
            disconnect_mt5()

def update_cached_data(symbols=None, timeframes=None, days_back=7):
    """
    Обновление кэшированных данных для указанных символов и таймфреймов
    
    Параметры:
    symbols (list): Список символов. Если None, используется [SYMBOL] из конфига
    timeframes (list): Список таймфреймов. Если None, используется TIMEFRAMES из конфига
    days_back (int): Количество дней для загрузки данных
    
    Возвращает:
    bool: True, если обновление успешно, иначе False
    """
    if symbols is None:
        symbols = [SYMBOL]
    
    if timeframes is None:
        timeframes = TIMEFRAMES
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    success = True
    
    for symbol in symbols:
        for tf in timeframes:
            try:
                logger.info(f"Обновление кэша для {symbol} {tf}...")
                df = get_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    force_reload=True
                )
                
                if df is None:
                    logger.warning(f"Не удалось обновить кэш для {symbol} {tf}")
                    success = False
                else:
                    logger.info(f"Кэш для {symbol} {tf} обновлен успешно")
            except Exception as e:
                logger.error(f"Ошибка при обновлении кэша для {symbol} {tf}: {str(e)}")
                success = False
    
    return success

# Проверка данных перед использованием
if __name__ == "__main__":
    # Настраиваем логирование для консоли при запуске скрипта напрямую
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    if connect_mt5():
        print("Проверка получения данных для разных таймфреймов:")
        
        for tf in TIMEFRAMES:
            # Используем параметр num_candles из конфига
            df = get_historical_data(
                symbol=SYMBOL,
                timeframe=tf,
                num_candles=CANDLES_FOR_EACH_TF.get(tf, 1000)  # По умолчанию 1000 свечей
            )
            
            if df is not None:
                print(f"{tf}: получено {len(df)} свечей")
                print(f"Период: с {df['time'].min()} по {df['time'].max()}")
                print(f"Первые 3 свечи:\n{df.head(3)}\n")
            else:
                print(f"{tf}: Ошибка получения данных\n")
        
        print("Тестирование загрузки всех таймфреймов одним запросом:")
        multi_tf_data = get_multi_timeframe_data()
        if multi_tf_data:
            print(f"Загружено {len(multi_tf_data)} таймфреймов")
            for tf, data in multi_tf_data.items():
                print(f"{tf}: {len(data)} свечей")
        
        print("\nТестирование обновления кэша:")
        update_result = update_cached_data(days_back=1)
        print(f"Результат обновления кэша: {'Успешно' if update_result else 'С ошибками'}")
        
        disconnect_mt5()
    else:
        print("Не удалось подключиться к MT5")