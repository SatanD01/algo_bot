import pandas as pd
import logging
import numpy as np
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from config import SYMBOL, BACKTEST_START, BACKTEST_END, TIMEFRAMES, CANDLES_FOR_EACH_TF, MIN_STOPLOSS_PIPS, INITIAL_BALANCE, RISK_PER_TRADE
from data_fetcher import get_historical_data
from strategy import find_trade_signal
from mt5_connector import connect_mt5, disconnect_mt5

# Директория для результатов
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
os.makedirs(results_dir, exist_ok=True)

# Настроим логирование
from config import LOGS_DIR, LOG_LEVEL, LOG_FILE_FORMAT

# Создаем директорию для логов, если она не существует
os.makedirs(LOGS_DIR, exist_ok=True)

# Получаем текущее время для имени файла
current_time = datetime.now().strftime(LOG_FILE_FORMAT)
log_file_path = os.path.join(LOGS_DIR, current_time)

# Настройка уровня логирования
log_level = getattr(logging, LOG_LEVEL.upper() if hasattr(logging, LOG_LEVEL.upper()) else "INFO")

# Настраиваем логирование
logging.basicConfig(filename=log_file_path, level=log_level, encoding="utf-8",
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

def backtest(verbose=True, save_progress=True, visualize_realtime=False):
    """
    Запуск бэктеста с анализом всех таймфреймов и расчетом прибыли.
    
    Параметры:
    verbose (bool): Если True, будет отображаться прогресс-бар и информация
    save_progress (bool): Если True, будет сохраняться промежуточный результат
    visualize_realtime (bool): Если True, будет отображаться визуализация в реальном времени
    
    Возвращает:
    DataFrame: DataFrame с результатами бэктеста
    """
    start_time = time.time()
    logging.info(f"Начало бэктеста: {BACKTEST_START} - {BACKTEST_END}")
    
    if not connect_mt5():
        logging.error("Ошибка подключения к MT5 перед бэктестом")
        return None

    results = []
    balance = INITIAL_BALANCE
    open_trades = []  # Для отслеживания открытых сделок

    # Для хранения информации о сделках для визуализации
    trade_history = {
        "entries": [],
        "exits": [],
        "stop_losses": [],
        "take_profits": []
    }
    
    logging.info(f"Загрузка исторических данных для таймфреймов: {', '.join(TIMEFRAMES)}")
    
    # Для каждого таймфрейма создаем полный датасет для бэктеста
    timeframe_data = {}
    for tf in TIMEFRAMES:
        if verbose:
            print(f"Загрузка данных для {tf}...")
        df = get_historical_data(SYMBOL, timeframe=tf, start_date=BACKTEST_START, end_date=BACKTEST_END)
        if df is None or len(df) < 20:
            logging.warning(f"Недостаточно данных для {tf}")
            continue
        timeframe_data[tf] = df
        if verbose:
            print(f"Загружено {len(df)} свечей для {tf}")
    
    # Создаем единую временную шкалу для бэктеста с наименьшим таймфреймом
    smallest_tf = min(TIMEFRAMES, key=lambda x: TIMEFRAMES.index(x))
    if smallest_tf in timeframe_data:
        timeline = timeframe_data[smallest_tf]['time'].tolist()
    else:
        logging.error("Не удалось создать временную шкалу для бэктеста")
        disconnect_mt5()
        return None
    
    # Инициализация визуализации в реальном времени, если включена
    visualizer = None
    if visualize_realtime:
        try:
            # Пытаемся импортировать и инициализировать визуализатор
            from backtest_visualizer import start_visualization
            visualizer = start_visualization(timeframe_data, SYMBOL)
            if visualizer:
                print("Визуализация инициализирована")
            else:
                print("Не удалось инициализировать визуализатор")
                visualize_realtime = False
        except Exception as e:
            print(f"Ошибка при инициализации визуализации: {e}")
            visualize_realtime = False
    
    total_bars = len(timeline)
    logging.info(f"Всего временных точек для анализа: {total_bars}")
    
    # Инициализируем прогресс-бар
    if verbose:
        progress_bar = tqdm(total=total_bars, desc="Бэктест", unit="бар")
    
    # Проходим по временной шкале для бэктеста
    last_progress_save = time.time()
    progress_counter = 0
    
    # Начинаем со 100-й свечи, чтобы иметь достаточно исторических данных для анализа
    for current_time_idx in range(100, len(timeline)):
        current_time = timeline[current_time_idx]
        
        # Обновляем визуализацию, если она включена
        if visualize_realtime and visualizer and current_time_idx % 10 == 0:  # Обновляем каждые 10 свечей
            try:
                # Обновляем визуализатор с текущими данными
                visualizer.update_data(current_time, balance, open_trades)
            except Exception as e:
                print(f"Ошибка при обновлении визуализации: {e}")
        
        # Обновляем прогресс
        if verbose:
            progress_bar.update(1)
        
        progress_counter += 1
        
        # Каждые 1000 свечей (или указанное значение) сохраняем промежуточный результат
        if save_progress and progress_counter % 1000 == 0:
            elapsed_time = time.time() - start_time
            processed_percent = (current_time_idx - 100) / (len(timeline) - 100) * 100
            estimated_total = elapsed_time / (processed_percent / 100) if processed_percent > 0 else 0
            estimated_remaining = estimated_total - elapsed_time
            
            logging.info(f"Прогресс бэктеста: {processed_percent:.2f}% ({current_time_idx}/{len(timeline)})")
            logging.info(f"Прошло времени: {timedelta(seconds=int(elapsed_time))}, осталось примерно: {timedelta(seconds=int(estimated_remaining))}")
            
            # Сохраняем промежуточные результаты, если их больше 0
            if len(results) > 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(os.path.join(results_dir, f"backtest_progress_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
        
        # Обновляем открытые сделки - проверяем, достигли ли они стоп-лосса или тейк-профита
        for trade_idx in reversed(range(len(open_trades))):
            trade = open_trades[trade_idx]
            
            # Получаем текущую свечу для проверки
            m5_df = timeframe_data[smallest_tf]
            current_candle_idx = m5_df.index[m5_df['time'] == current_time].tolist()
            
            if not current_candle_idx:
                continue
                
            current_candle_idx = current_candle_idx[0]
            current_candle = m5_df.iloc[current_candle_idx]
            
            # Проверяем достижение стоп-лосса или тейк-профита
            if trade['order'] == "buy":
                # Проверка стоп-лосса
                if current_candle['low'] <= trade['entry_price'] - trade['stop_loss']:
                    # Стоп-лосс сработал
                    trade['result'] = "loss"
                    trade['exit_price'] = trade['entry_price'] - trade['stop_loss']
                    trade['exit_time'] = current_time
                    trade['profit'] = -trade['risk_amount']
                    balance += trade['profit']
                    
                    # Добавляем в историю для визуализации
                    trade_history["exits"].append({
                        "time": trade['exit_time'],
                        "price": trade['exit_price'],
                        "type": "stop_loss",
                        "order": trade['order'],
                        "profit": trade['profit']
                    })
                    
                    # Добавляем результат в лог
                    results.append({
                        "time": trade['entry_time'],
                        "order": trade['order'],
                        "tf": trade['tf'],
                        "result": trade['result'],
                        "profit": trade['profit'],
                        "lot_size": trade['lot_size'],
                        "entry_price": trade['entry_price'],
                        "exit_price": trade['exit_price'],
                        "entry_time": trade['entry_time'],
                        "exit_time": trade['exit_time'],
                        "stop_loss": trade['stop_loss'],
                        "take_profit": trade['take_profit'],
                        "balance": balance,
                        "setup": trade.get('setup', 'Standard')
                    })
                    
                    # Удаляем сделку из открытых
                    open_trades.pop(trade_idx)
                    logging.info(f"SL сработал для BUY на {current_time}: {trade['exit_price']}")
                    continue
                
                # Проверка тейк-профита
                if current_candle['high'] >= trade['entry_price'] + trade['take_profit']:
                    # Тейк-профит сработал
                    trade['result'] = "win"
                    trade['exit_price'] = trade['entry_price'] + trade['take_profit']
                    trade['exit_time'] = current_time
                    trade['profit'] = trade['risk_amount'] * (trade['take_profit'] / trade['stop_loss'])
                    balance += trade['profit']
                    
                    # Добавляем в историю для визуализации
                    trade_history["exits"].append({
                        "time": trade['exit_time'],
                        "price": trade['exit_price'],
                        "type": "take_profit",
                        "order": trade['order'],
                        "profit": trade['profit']
                    })
                    
                    # Добавляем результат в лог
                    results.append({
                        "time": trade['entry_time'],
                        "order": trade['order'],
                        "tf": trade['tf'],
                        "result": trade['result'],
                        "profit": trade['profit'],
                        "lot_size": trade['lot_size'],
                        "entry_price": trade['entry_price'],
                        "exit_price": trade['exit_price'],
                        "entry_time": trade['entry_time'],
                        "exit_time": trade['exit_time'],
                        "stop_loss": trade['stop_loss'],
                        "take_profit": trade['take_profit'],
                        "balance": balance,
                        "setup": trade.get('setup', 'Standard')
                    })
                    
                    # Удаляем сделку из открытых
                    open_trades.pop(trade_idx)
                    logging.info(f"TP сработал для BUY на {current_time}: {trade['exit_price']}")
                    continue
            
            else:  # sell
                # Проверка стоп-лосса
                if current_candle['high'] >= trade['entry_price'] + trade['stop_loss']:
                    # Стоп-лосс сработал
                    trade['result'] = "loss"
                    trade['exit_price'] = trade['entry_price'] + trade['stop_loss']
                    trade['exit_time'] = current_time
                    trade['profit'] = -trade['risk_amount']
                    balance += trade['profit']
                    
                    # Добавляем в историю для визуализации
                    trade_history["exits"].append({
                        "time": trade['exit_time'],
                        "price": trade['exit_price'],
                        "type": "stop_loss",
                        "order": trade['order'],
                        "profit": trade['profit']
                    })
                    
                    # Добавляем результат в лог
                    results.append({
                        "time": trade['entry_time'],
                        "order": trade['order'],
                        "tf": trade['tf'],
                        "result": trade['result'],
                        "profit": trade['profit'],
                        "lot_size": trade['lot_size'],
                        "entry_price": trade['entry_price'],
                        "exit_price": trade['exit_price'],
                        "entry_time": trade['entry_time'],
                        "exit_time": trade['exit_time'],
                        "stop_loss": trade['stop_loss'],
                        "take_profit": trade['take_profit'],
                        "balance": balance,
                        "setup": trade.get('setup', 'Standard')
                    })
                    
                    # Удаляем сделку из открытых
                    open_trades.pop(trade_idx)
                    logging.info(f"SL сработал для SELL на {current_time}: {trade['exit_price']}")
                    continue
                
                # Проверка тейк-профита
                if current_candle['low'] <= trade['entry_price'] - trade['take_profit']:
                    # Тейк-профит сработал
                    trade['result'] = "win"
                    trade['exit_price'] = trade['entry_price'] - trade['take_profit']
                    trade['exit_time'] = current_time
                    trade['profit'] = trade['risk_amount'] * (trade['take_profit'] / trade['stop_loss'])
                    balance += trade['profit']
                    
                    # Добавляем в историю для визуализации
                    trade_history["exits"].append({
                        "time": trade['exit_time'],
                        "price": trade['exit_price'],
                        "type": "take_profit",
                        "order": trade['order'],
                        "profit": trade['profit']
                    })
                    
                    # Добавляем результат в лог
                    results.append({
                        "time": trade['entry_time'],
                        "order": trade['order'],
                        "tf": trade['tf'],
                        "result": trade['result'],
                        "profit": trade['profit'],
                        "lot_size": trade['lot_size'],
                        "entry_price": trade['entry_price'],
                        "exit_price": trade['exit_price'],
                        "entry_time": trade['entry_time'],
                        "exit_time": trade['exit_time'],
                        "stop_loss": trade['stop_loss'],
                        "take_profit": trade['take_profit'],
                        "balance": balance,
                        "setup": trade.get('setup', 'Standard')
                    })
                    
                    # Удаляем сделку из открытых
                    open_trades.pop(trade_idx)
                    logging.info(f"TP сработал для SELL на {current_time}: {trade['exit_price']}")
                    continue
        
        # Проверяем сигналы на всех таймфреймах
        for tf in TIMEFRAMES:
            if tf not in timeframe_data:
                continue
                
            # Получаем данные для текущего таймфрейма до текущего момента времени
            current_df = timeframe_data[tf]
            mask = current_df['time'] <= current_time
            test_df = current_df[mask].copy()
            
            # Пропускаем, если недостаточно данных для анализа
            if len(test_df) < 30:
                continue
                
            # Ищем сигнал на основе данных до текущего момента
            signal = find_trade_signal(test_df)
            
            # Если сигнал найден, проверяем его условия и открываем сделку
            if signal:
                # Проверяем, не находимся ли мы уже в сделке в том же направлении
                already_in_trade = False
                for trade in open_trades:
                    if trade['order'] == signal['type']:
                        already_in_trade = True
                        break
                
                if already_in_trade:
                    logging.info(f"Уже есть открытая сделка в направлении {signal['type']}, пропускаем сигнал")
                    continue
                
                # Проверяем, что сигнал подтверждается и на младшем таймфрейме, если это старший ТФ
                if tf in ["D1", "H4", "H1"]:
                    m15_df = timeframe_data.get("M15")
                    if m15_df is not None:
                        m15_mask = m15_df['time'] <= current_time
                        m15_test_df = m15_df[m15_mask].copy()
                        if len(m15_test_df) >= 30:
                            m15_signal = find_trade_signal(m15_test_df)
                            if not m15_signal or m15_signal['type'] != signal['type']:
                                logging.info(f"Сигнал на {tf} не подтвержден на M15, пропускаем")
                                continue
                
                # Получаем стоп-лосс и тейк-профит из сигнала
                stop_loss = signal.get('stop_loss', 0.0003)  # Дефолтное значение 30 пипсов
                take_profit = signal.get('take_profit', stop_loss * 3)  # Дефолтное соотношение 1:3
                
                # Проверяем минимальный стоп-лосс
                if stop_loss < MIN_STOPLOSS_PIPS * 0.0001:
                    stop_loss = MIN_STOPLOSS_PIPS * 0.0001
                
                # Рассчитываем размер позиции на основе риска
                risk_amount = balance * RISK_PER_TRADE  # Риск в валюте счета
                stop_loss_pips = int(stop_loss / 0.0001)  # Конвертируем в пипсы
                
                # Рассчитываем размер лота (для EUR/USD: 1 пипс при лоте 1.0 = $10)
                lot_size = risk_amount / (stop_loss_pips * 10)
                
                # Округляем лот до 2 знаков после запятой и устанавливаем минимум
                lot_size = round(lot_size, 2)
                if lot_size < 0.01:
                    lot_size = 0.01
                
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
                
                # Создаем запись о новой сделке
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
                logging.info(f"Открыта новая сделка {signal['type']} на {tf} по цене {signal['level']}, лот: {lot_size}, SL: {stop_loss}, TP: {take_profit}")
    
    # Закрываем прогресс-бар
    if verbose:
        progress_bar.close()
    
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
        
        # Добавляем в результаты
        results.append({
            "time": trade['entry_time'],
            "order": trade['order'],
            "tf": trade['tf'],
            "result": result,
            "profit": profit,
            "lot_size": trade['lot_size'],
            "entry_price": trade['entry_price'],
            "exit_price": final_price,
            "entry_time": trade['entry_time'],
            "exit_time": final_time,
            "stop_loss": trade['stop_loss'],
            "take_profit": trade['take_profit'],
            "balance": balance,
            "setup": trade.get('setup', 'Standard')
        })
        
        logging.info(f"Закрыта сделка в конце периода: {trade['order']} на {trade['tf']} с результатом {result}")
    
    # Сохраняем историю сделок для визуализации
    pd.DataFrame(trade_history["entries"]).to_csv(os.path.join(results_dir, f"backtest_entries_{SYMBOL}.csv"), index=False)
    pd.DataFrame(trade_history["exits"]).to_csv(os.path.join(results_dir, f"backtest_exits_{SYMBOL}.csv"), index=False)
    pd.DataFrame(trade_history["stop_losses"]).to_csv(os.path.join(results_dir, f"backtest_sl_{SYMBOL}.csv"), index=False)
    pd.DataFrame(trade_history["take_profits"]).to_csv(os.path.join(results_dir, f"backtest_tp_{SYMBOL}.csv"), index=False)
    
    # Создаем DataFrame из результатов
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        logging.warning("Бэктест завершен, но нет ни одной сделки.")
    else:
        # Сохраняем полные результаты
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f"backtest_results_{SYMBOL}_{timestamp}.csv")
        df_results.to_csv(results_file, index=False)
        
        # Анализ результатов
        order_counts = df_results['order'].value_counts()
        logging.info(f"Бэктест завершен: order\n{order_counts}")
        
        total_trades = len(df_results)
        wins = df_results[df_results["result"] == "win"].shape[0]
        winrate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        logging.info(f"Общий результат: {wins} побед из {total_trades} сделок ({winrate:.2f}%)")
        logging.info(f"Итоговый баланс: {balance:.2f}")
        
        # Анализ по сетапам
        if 'setup' in df_results.columns:
            setup_stats = df_results.groupby('setup')['result'].apply(
                lambda x: f"{(x == 'win').sum()}/{len(x)} ({(x == 'win').sum()/len(x)*100:.2f}%)"
            )
            logging.info(f"Статистика по сетапам:\n{setup_stats}")
        
        # Анализ по таймфреймам
        tf_stats = df_results.groupby('tf')['result'].apply(
            lambda x: f"{(x == 'win').sum()}/{len(x)} ({(x == 'win').sum()/len(x)*100:.2f}%)"
        )
        logging.info(f"Статистика по таймфреймам:\n{tf_stats}")
        
        # Средняя прибыль на сделку
        avg_profit_per_trade = df_results['profit'].mean()
        
        # Расчет максимальной просадки
        df_results['cumulative_profit'] = df_results['profit'].cumsum()
        df_results['peak'] = df_results['cumulative_profit'].cummax()
        df_results['drawdown'] = df_results['cumulative_profit'] - df_results['peak']
        max_drawdown = abs(df_results['drawdown'].min())
        max_drawdown_percent = (max_drawdown / (INITIAL_BALANCE + df_results['peak'].max())) * 100
        
        logging.info(f"Средняя прибыль на сделку: {avg_profit_per_trade:.2f}")
        logging.info(f"Максимальная просадка: {max_drawdown:.2f} ({max_drawdown_percent:.2f}%)")
        
        # Вывод времени выполнения
        total_time = time.time() - start_time
        logging.info(f"Время выполнения бэктеста: {timedelta(seconds=int(total_time))}")
        
        if verbose:
            print(f"\nБэктест завершен за {timedelta(seconds=int(total_time))}")
            print(f"Всего сделок: {total_trades} (Побед: {wins}, Поражений: {total_trades - wins})")
            print(f"Винрейт: {winrate:.2f}%")
            print(f"Итоговый баланс: {balance:.2f}")
            print(f"Результаты сохранены в {results_file}")

    disconnect_mt5()
    return df_results

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    
    results = backtest(verbose=True, save_progress=True)
    
    if results is not None and not results.empty:
        # Строим график баланса
        plt.figure(figsize=(12, 6))
        plt.plot(results['exit_time'], results['balance'])
        plt.title('Динамика баланса')
        plt.xlabel('Время')
        plt.ylabel('Баланс')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"backtest_balance_{SYMBOL}.png"))
        plt.close()
        
        print("Результаты сохранены. Запустите visualization.py для просмотра графиков.")
    else:
        print("Нет данных для анализа")