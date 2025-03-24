import time
import logging
import os
import sys
import traceback
import argparse
from datetime import datetime, timedelta
from config import CHECK_INTERVAL, MODE, SYMBOL, LOG_LEVEL, is_trading_allowed, MAX_POSITIONS, RISK_PER_TRADE, MT5_LOGIN, MT5_SERVER, TRADE_JOURNAL_ENABLED
from mt5_connector import connect_mt5, disconnect_mt5, get_account_info, get_open_positions
from trade_executor import execute_trade
from dotenv import load_dotenv
from backtest import backtest
try:
    from backtest_optimized import optimized_backtest
    OPTIMIZED_BACKTEST_AVAILABLE = True
except ImportError:
    OPTIMIZED_BACKTEST_AVAILABLE = False

# Получаем абсолютный путь к текущей директории
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, '.env')

# Загрузка переменных из .env файла с проверкой его существования
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Загружены настройки из {env_path}")
else:
    print(f"ВНИМАНИЕ: Файл .env не найден в {base_dir}. Используются значения по умолчанию.")

# Директория для логов
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Настройка логгера
logger = logging.getLogger(__name__)

# Имя файла логов с датой
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename = os.path.join(logs_dir, f"{current_date}_bot_log.txt")

# Настройка уровня логирования из конфигурации
log_level = getattr(logging, LOG_LEVEL.upper() if hasattr(logging, LOG_LEVEL.upper()) else "INFO")

# Настраиваем логирование в файл и в консоль
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_backtest(optimized=False):
    """
    Запуск бэктеста и анализ результатов
    
    Параметры:
    optimized (bool): Использовать ли оптимизированный бэктест
    """
    logging.info("=" * 50)
    logging.info(f"Запуск {'оптимизированного ' if optimized else ''}бэктеста для {SYMBOL}")
    logging.info("=" * 50)
    
    start_time = time.time()
    
    try:
        # Запрашиваем параметры для бэктеста
        print("\nВыберите режим бэктеста:")
        print("1. Параллельный бэктест (рекомендуется для больших периодов)")
        print("2. Последовательный бэктест (более точный, но медленнее)")
        
        choice = input("Ваш выбор (1/2): ").strip()
        
        # Опция визуализации в реальном времени
        use_visualizer = input("Запустить визуализатор в реальном времени? (y/n): ").strip().lower() == 'y'
        
        # Запускаем бэктест
        if optimized and OPTIMIZED_BACKTEST_AVAILABLE:
            if choice == "1":
                # Определяем количество процессов
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                recommended_processes = max(1, cpu_count - 1)  # Оставляем одно ядро для ОС
                
                num_processes = input(f"Введите количество процессов [{recommended_processes}]: ").strip()
                num_processes = int(num_processes) if num_processes else recommended_processes
                
                # Запускаем бэктест с параллельной обработкой
                results = optimized_backtest(
                    parallel=True, 
                    num_processes=num_processes,
                    visualize_realtime=use_visualizer
                )
            else:
                # Запускаем последовательный бэктест
                results = optimized_backtest(
                    parallel=False,
                    visualize_realtime=use_visualizer
                )
        else:
            # Используем стандартный бэктест
            results = backtest(
                verbose=True, 
                save_progress=True,
                visualize_realtime=use_visualizer
            )
        
        if results is None or len(results) == 0:
            logging.error("Бэктест завершен без результатов")
            return False
        
        # Вычисляем продолжительность бэктеста
        duration = time.time() - start_time
        duration_str = str(timedelta(seconds=int(duration)))
        
        # Анализируем результаты
        total_trades = len(results)
        wins = results[results["result"] == "win"].shape[0]
        losses = results[results["result"] == "loss"].shape[0]
        winrate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        # Анализ по типу ордера
        buy_trades = results[results["order"] == "buy"].shape[0]
        sell_trades = results[results["order"] == "sell"].shape[0]
        
        # Анализ прибыли
        total_profit = results["profit"].sum()
        avg_profit = results["profit"].mean()
        max_profit = results["profit"].max()
        max_loss = results["profit"].min()
        
        # Вывод результатов
        logging.info(f"Бэктест завершен за {duration_str}")
        logging.info(f"Всего сделок: {total_trades} (Buy: {buy_trades}, Sell: {sell_trades})")
        logging.info(f"Побед: {wins}, Поражений: {losses}, Винрейт: {winrate:.2f}%")
        logging.info(f"Общая прибыль: {total_profit:.2f}, Средняя прибыль на сделку: {avg_profit:.2f}")
        logging.info(f"Максимальная прибыль: {max_profit:.2f}, Максимальный убыток: {max_loss:.2f}")
        
        # Предлагаем визуализировать результаты
        print("\nЗапустить визуализатор для детального анализа результатов? (y/n): ")
        if input().strip().lower() == 'y':
            from backtest_visualizer import visualize_backtest
            visualize_backtest()
        
        return True
    
    except KeyboardInterrupt:
        print("\nБэктест прерван пользователем")
        return False
        
    except Exception as e:
        logging.error(f"Ошибка при выполнении бэктеста: {str(e)}")
        logging.error(traceback.format_exc())
        return False

# В файле main.py убедитесь, что импорт выглядит так:
def run_visualization_menu():
    """Показать меню выбора для визуализации"""
    print("\n=== Меню визуализации бэктеста ===")
    print("1. Обычный режим (статический HTML-отчет)")
    print("2. Интерактивный режим (режим реального времени)")
    print("3. Тёмная тема")
    print("4. Без загрузки ценовых графиков (быстрее)")
    print("5. Выход")
    
    choice = input("\nВыберите режим (1-5): ").strip()
    
    try:
        if choice == "1":
            # Обычный режим - используем функцию из visualization.py
            from visualization import visualize_backtest
            visualize_backtest()
        elif choice == "2":
            # Интерактивный режим - с загрузкой данных
            try:
                from backtest_visualizer import create_realtime_visualizer
                from visualization import find_latest_backtest_files, load_and_optimize_df
                from data_fetcher import get_historical_data
                from mt5_connector import connect_mt5, disconnect_mt5
                from datetime import timedelta
                import pandas as pd
                
                # Находим файлы бэктеста
                print("Поиск файлов бэктеста...")
                result_file, entries_file, exits_file, sl_file, tp_file = find_latest_backtest_files()
                
                if result_file:
                    # Загружаем данные
                    print("Загрузка данных бэктеста...")
                    results = load_and_optimize_df(result_file)
                    entries = load_and_optimize_df(entries_file)
                    exits = load_and_optimize_df(exits_file)
                    stop_losses = load_and_optimize_df(sl_file)
                    take_profits = load_and_optimize_df(tp_file)
                    
                    # Извлекаем основной символ
                    from config import SYMBOL
                    
                    # Определяем временной диапазон
                    start_date = None
                    end_date = None
                    
                    if results is not None and not results.empty:
                        if 'entry_time' in results.columns:
                            start_date = pd.to_datetime(results['entry_time'].min()) - timedelta(days=3)
                        if 'exit_time' in results.columns:
                            end_date = pd.to_datetime(results['exit_time'].max()) + timedelta(days=3)
                    
                    # Загружаем ценовые данные
                    print("Загрузка данных цен...")
                    price_data = {}
                    
                    if connect_mt5():
                        try:
                            # Загружаем данные для основных таймфреймов
                            for tf in ["M5", "M15", "H1"]:
                                print(f"Загрузка данных для {tf}...")
                                df = get_historical_data(
                                    symbol=SYMBOL,
                                    timeframe=tf, 
                                    start_date=start_date,
                                    end_date=end_date,
                                    use_cache=True
                                )
                                if df is not None and not df.empty:
                                    price_data[tf] = df
                                    print(f"Загружено {len(df)} свечей для {tf}")
                        finally:
                            disconnect_mt5()
                    
                    # Запускаем интерактивный визуализатор
                    print("Запуск интерактивной визуализации...")
                    create_realtime_visualizer(
                        results=results,
                        entries=entries,
                        exits=exits,
                        stop_losses=stop_losses,
                        take_profits=take_profits,
                        timeframe_data=price_data
                    )
                    print("Запущена интерактивная визуализация. Для выхода нажмите Ctrl+C.")
                    
                    # Ждем, чтобы пользователь мог взаимодействовать с визуализацией
                    try:
                        import time
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("Визуализация остановлена.")
                else:
                    print("Не найдены файлы с результатами бэктеста.")
            except ImportError as e:
                print(f"Интерактивный режим недоступен: {e}")
                print("Используется стандартная визуализация.")
                from visualization import visualize_backtest
                visualize_backtest()
            except Exception as e:
                print(f"Ошибка при запуске интерактивной визуализации: {e}")
                import traceback
                traceback.print_exc()
        elif choice == "3":
            # Тёмная тема
            from visualization import visualize_backtest
            visualize_backtest(theme="dark")
        elif choice == "4":
            # Без загрузки ценовых графиков
            from visualization import visualize_backtest
            visualize_backtest(with_prices=False)
        elif choice == "5":
            print("Выход из меню визуализации")
            return
        else:
            print("Некорректный выбор")
    except ImportError as e:
        print(f"Не удалось импортировать модуль визуализации: {e}")
        print("Убедитесь, что у вас установлены необходимые библиотеки:")
        print("pip install plotly pandas numpy")
        print("Для интерактивного режима также: pip install dash dash-bootstrap-components")
    except Exception as e:
        print(f"Ошибка при запуске визуализации: {e}")
        logging.error(f"Ошибка при выполнении действия: {str(e)}")
        logging.error(traceback.format_exc())

def run_live_trading():
    """Запуск режима реальной торговли"""
    if not connect_mt5():
        logging.error("Ошибка подключения к MT5")
        return False
    
    logging.info("=" * 50)
    logging.info(f"Бот запущен в режиме реальной торговли для {SYMBOL}")
    logging.info("=" * 50)
    
    # Импортируем все необходимые модули
    from trade_executor import execute_trade, get_trade_journal
    from mt5_connector import get_account_info, get_open_positions, get_symbol_info
    import traceback
    
    # Инициализируем расширенный журнал сделок, если он доступен
    try:
        from trade_journal import TradeJournal
        from config import TRADE_JOURNAL_ENABLED, TRADE_JOURNAL_AUTO_REPORT, TRADE_JOURNAL_DAYS_SUMMARY
        
        if TRADE_JOURNAL_ENABLED:
            journal = get_trade_journal(SYMBOL)
            if journal is not None:
                # Генерируем отчет о предыдущей торговле, если включено автосоздание отчетов
                if TRADE_JOURNAL_AUTO_REPORT:
                    report_file = journal.generate_performance_report()
                    if report_file:
                        logger.info(f"Создан отчет о производительности: {report_file}")
                
                # Выводим краткую статистику за указанное количество дней
                journal.print_summary(days=TRADE_JOURNAL_DAYS_SUMMARY)
    except ImportError:
        logger.info("Расширенный журнал сделок недоступен")
    except Exception as e:
        logger.warning(f"Ошибка при инициализации расширенного журнала сделок: {str(e)}")
    
    # Инициализируем риск-менеджер, если он доступен
    risk_manager = None
    try:
        from risk_manager import get_risk_manager
        from config import RISK_MANAGER_ENABLED, RISK_POSITION_SIZING_METHOD
        
        # Получаем информацию о счете для инициализации риск-менеджера
        account_info = get_account_info()
        if account_info and RISK_MANAGER_ENABLED:
            risk_manager = get_risk_manager(
                account_balance=account_info["balance"],
                position_sizing_method=RISK_POSITION_SIZING_METHOD
            )
            logger.info("Риск-менеджер инициализирован")
    except ImportError:
        logger.info("Модуль риск-менеджера недоступен")
    except Exception as e:
        logger.warning(f"Ошибка при инициализации риск-менеджера: {str(e)}")
    
    # Инициализируем менеджер позиций для трейлинг-стопов, если он доступен
    position_manager = None
    try:
        from position_manager import get_position_manager
        
        # Инициализируем с настройками из конфига (если они доступны)
        try:
            from config import (TRAILING_ACTIVATION, BREAKEVEN_ACTIVATION, 
                              TRAILING_STEP, PARTIAL_CLOSE_PCT, USE_AUTO_CLOSE)
            
            position_manager = get_position_manager(
                symbol=SYMBOL,
                trailing_activation=TRAILING_ACTIVATION if 'TRAILING_ACTIVATION' in locals() else 0.5,
                breakeven_activation=BREAKEVEN_ACTIVATION if 'BREAKEVEN_ACTIVATION' in locals() else 0.3,
                trailing_step=TRAILING_STEP if 'TRAILING_STEP' in locals() else 0.1,
                partial_close_pct=PARTIAL_CLOSE_PCT if 'PARTIAL_CLOSE_PCT' in locals() else 0.5,
                use_auto_close=USE_AUTO_CLOSE if 'USE_AUTO_CLOSE' in locals() else True
            )
        except ImportError:
            position_manager = get_position_manager(symbol=SYMBOL)
        
        logger.info("Менеджер позиций инициализирован")
    except ImportError:
        logger.info("Модуль менеджера позиций недоступен")
    except Exception as e:
        logger.warning(f"Ошибка при инициализации менеджера позиций: {str(e)}")
    
    # Получаем информацию о счете
    account_info = get_account_info()
    if account_info:
        logging.info(f"Счет: {account_info['login']}, Баланс: {account_info['balance']} {account_info['currency']}")
    
    # Получаем открытые позиции
    positions = get_open_positions()
    if positions:
        logging.info(f"Открыто позиций: {len(positions)}")
        for pos in positions:
            logging.info(f"  {pos['symbol']} {pos['type']} {pos['volume']} лот, "
                        f"прибыль: {pos['profit']}, SL: {pos['sl']}, TP: {pos['tp']}")
    else:
        logging.info("Нет открытых позиций")
    
    # Создаем словарь для отслеживания открытых позиций по символам
    open_positions_tracking = {}
    
    # Заполняем словарь начальными данными
    if positions:
        for pos in positions:
            symbol = pos['symbol']
            if symbol not in open_positions_tracking:
                open_positions_tracking[symbol] = []
            open_positions_tracking[symbol].append({'ticket': pos['ticket'], 'type': pos['type']})
    
    # Время последнего обновления трейлинг-стопов
    last_trailing_update = time.time()
    # Интервал обновления трейлинг-стопов (в секундах)
    trailing_update_interval = 300  # 5 минут
    
    try:
        last_check_time = time.time()
        cycle_count = 0
        
        while True:
            # Проверяем, разрешена ли торговля сейчас
            if is_trading_allowed():
                # Проверяем MT5 и переподключаемся при необходимости
                if cycle_count % 10 == 0:  # Каждые 10 циклов
                    if not connect_mt5():
                        logging.error("Ошибка подключения к MT5, повтор через 30 секунд")
                        time.sleep(30)
                        continue
                    
                    # После переподключения обновляем информацию об открытых позициях
                    current_positions = get_open_positions()
                    
                    # Создаем новый словарь текущих позиций
                    current_positions_dict = {}
                    if current_positions:
                        for pos in current_positions:
                            symbol = pos['symbol']
                            if symbol not in current_positions_dict:
                                current_positions_dict[symbol] = []
                            current_positions_dict[symbol].append({'ticket': pos['ticket'], 'type': pos['type']})
                    
                    # Обновляем словарь отслеживания
                    open_positions_tracking = current_positions_dict
                
                # Проверяем необходимость обновления трейлинг-стопов
                current_time = time.time()
                if current_time - last_trailing_update >= trailing_update_interval:
                    # Обновляем информацию о трейлинг-стопах, если есть открытые позиции
                    if position_manager and any(open_positions_tracking.values()):
                        try:
                            # Обрабатываем открытые позиции
                            result = position_manager.process_open_positions()
                            
                            # Логируем результат обработки позиций
                            if 'actions' in result and any(result['actions'].values()):
                                logger.info(f"Обработка позиций: {result['message']}")
                                
                                # Детальное логирование действий
                                for action_type, actions in result['actions'].items():
                                    if actions:
                                        for action in actions:
                                            if action_type == "trailing_stop_modified":
                                                logger.info(f"Трейлинг-стоп: тикет {action['ticket']}, "
                                                           f"новый SL: {action['new_sl']:.5f}, "
                                                           f"прибыль: {action['profit_pips']:.1f} пипсов")
                                            elif action_type == "moved_to_breakeven":
                                                logger.info(f"Безубыток: тикет {action['ticket']}, "
                                                           f"новый SL: {action['new_sl']:.5f}")
                                            elif action_type == "partially_closed":
                                                logger.info(f"Частичное закрытие: тикет {action['ticket']}, "
                                                           f"объем: {action['volume_closed']} лот, "
                                                           f"прибыль: {action['profit']:.2f}")
                                            elif action_type == "auto_closed":
                                                logger.info(f"Автозакрытие: тикет {action['ticket']}, "
                                                           f"причина: {action['reason']}, "
                                                           f"прибыль: {action['profit']:.2f}")
                        except Exception as e:
                            logger.error(f"Ошибка при обработке открытых позиций: {str(e)}")
                    
                    # Обновляем время последнего обновления трейлинг-стопов
                    last_trailing_update = current_time
                
                # Получаем текущие открытые позиции для текущего символа
                current_positions = get_open_positions(symbol=SYMBOL)
                
                # ВАЖНОЕ ИЗМЕНЕНИЕ: Проверяем наличие открытых позиций для символа
                symbol_has_open_position = False
                for pos in current_positions:
                    if pos['symbol'] == SYMBOL:
                        symbol_has_open_position = True
                        break
                
                # Проверяем, есть ли уже открытые позиции для текущего символа
                # и допустимо ли открытие новых позиций
                if len(current_positions) < MAX_POSITIONS and not symbol_has_open_position:
                    # Проверяем, разрешена ли торговля через риск-менеджер, если он доступен
                    can_trade = True
                    if risk_manager:
                        can_trade, reason = risk_manager.can_trade()
                        if not can_trade:
                            logger.warning(f"Торговля остановлена риск-менеджером: {reason}")
                    
                    if can_trade:
                        # Выполняем торговую логику
                        # Если есть сигнал, он будет проверен на соответствие с текущими открытыми позициями
                        signal = execute_trade()
                        
                        # Если был открыт новый ордер, обновляем информацию в tracking
                        if signal:
                            symbol = signal['symbol']
                            if symbol not in open_positions_tracking:
                                open_positions_tracking[symbol] = []
                            open_positions_tracking[symbol].append({'ticket': signal['ticket'], 'type': signal['type']})
                            
                            # Обновляем информацию в риск-менеджере, если новая сделка
                            if risk_manager:
                                # Получаем обновленный баланс
                                account_info = get_account_info()
                                if account_info:
                                    risk_manager.update_balance(account_info["balance"])
                else:
                    if symbol_has_open_position:
                        logging.info(f"Уже есть открытая позиция для {SYMBOL}, ждем ее закрытия перед открытием новой")
                    else:
                        logging.info(f"Достигнуто максимальное количество позиций: {len(current_positions)}/{MAX_POSITIONS}")
                
                # Логируем активность каждый час
                current_time = time.time()
                if current_time - last_check_time >= 3600:  # 1 час
                    positions = get_open_positions()
                    logging.info(f"Бот активен. Открыто позиций: {len(positions)}")
                    
                    # Обновляем баланс в риск-менеджере
                    if risk_manager:
                        account_info = get_account_info()
                        if account_info:
                            risk_manager.update_balance(account_info["balance"])
                            logging.info(f"Обновлен баланс риск-менеджера: {account_info['balance']}")
                    
                    last_check_time = current_time
            else:
                if cycle_count % 30 == 0:  # Логируем только периодически
                    logging.info("Торговля не разрешена в текущее время")
            
            # Увеличиваем счетчик циклов
            cycle_count += 1
            
            # Ждем до следующей проверки
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        logging.info("Бот остановлен пользователем")
    except Exception as e:
        # Расширенное логирование ошибок с трассировкой стека
        logging.error(f"Критическая ошибка в основном цикле: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        # Гарантированное отключение от MT5
        disconnect_mt5()
        logging.info("Соединение с MT5 закрыто")
        
        # Создаем финальный отчет по торговле, если включен журнал сделок
        try:
            from trade_journal import TradeJournal
            from trade_executor import get_trade_journal
            from config import TRADE_JOURNAL_ENABLED, TRADE_JOURNAL_AUTO_REPORT
            
            if TRADE_JOURNAL_ENABLED and TRADE_JOURNAL_AUTO_REPORT:
                journal = get_trade_journal(SYMBOL, init_if_none=False)
                if journal is not None:
                    # Генерируем финальный отчет
                    report_file = journal.generate_performance_report()
                    if report_file:
                        logging.info(f"Создан финальный отчет о торговле: {report_file}")
        except Exception as e:
            logging.warning(f"Не удалось создать финальный отчет: {str(e)}")
        
        # Экспортируем статистику риск-менеджера
        try:
            if risk_manager:
                stats_file = risk_manager.export_statistics(format="json")
                logging.info(f"Статистика риск-менеджера экспортирована в {stats_file}")
        except Exception as e:
            logging.warning(f"Ошибка при экспорте статистики риск-менеджера: {str(e)}")
    
    return True

def run_trade_journal_menu():
    """Меню для работы с расширенным журналом сделок"""
    try:
        from trade_journal import TradeJournal
        
        # Инициализируем журнал сделок
        journal = TradeJournal(SYMBOL)
        
        while True:
            print("\n=== Меню журнала сделок ===")
            print("1. Показать краткую статистику")
            print("2. Создать подробный отчет")
            print("3. Экспортировать данные в CSV")
            print("4. Экспортировать данные в формат MT5")
            print("5. Построить график накопленной прибыли")
            print("6. Создать график выигрышей/проигрышей")
            print("7. Вернуться в главное меню")
            
            choice = input("Выберите опцию (1-7): ").strip()
            
            if choice == "1":
                days = input("За сколько последних дней показать статистику? (Enter для всего периода): ").strip()
                if days and days.isdigit():
                    journal.print_summary(days=int(days))
                else:
                    journal.print_summary()
            
            elif choice == "2":
                print("Создание отчета...")
                report_file = journal.generate_performance_report()
                if report_file:
                    print(f"Отчет создан и сохранен: {report_file}")
                    
                    # Спрашиваем, открыть ли отчет в браузере
                    if input("Открыть отчет в браузере? (y/n): ").strip().lower() == 'y':
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(report_file)}")
                else:
                    print("Не удалось создать отчет")
            
            elif choice == "3":
                print("Экспорт данных в CSV...")
                export_file = journal.export_to_csv()
                if export_file:
                    print(f"Данные экспортированы в: {export_file}")
                else:
                    print("Не удалось экспортировать данные")
            
            elif choice == "4":
                print("Экспорт данных в формат MT5...")
                export_file = journal.export_to_mt5_format()
                if export_file:
                    print(f"Данные экспортированы в формат MT5: {export_file}")
                else:
                    print("Не удалось экспортировать данные")
            
            elif choice == "5":
                print("Построение графика накопленной прибыли...")
                chart_file = journal.plot_cumulative_profit(show=False)
                if chart_file:
                    print(f"График сохранен: {chart_file}")
                else:
                    print("Не удалось построить график")
            
            elif choice == "6":
                print("Построение графика выигрышей/проигрышей...")
                chart_file = journal.plot_wins_vs_losses(show=False)
                if chart_file:
                    print(f"График сохранен: {chart_file}")
                else:
                    print("Не удалось построить график")
            
            elif choice == "7":
                print("Возврат в главное меню")
                break
            
            else:
                print("Некорректный выбор. Пожалуйста, выберите 1-7.")
    
    except ImportError:
        print("Расширенный журнал сделок недоступен. Установите необходимые библиотеки:")
        print("pip install pandas numpy matplotlib")
    
    except Exception as e:
        print(f"Ошибка при работе с журналом сделок: {e}")
        logging.error(f"Ошибка при работе с журналом сделок: {str(e)}")
        logging.error(traceback.format_exc())

def run_risk_manager_menu():
    """Меню для работы с риск-менеджером"""
    try:
        from risk_manager import get_risk_manager, reset_risk_manager
        from mt5_connector import connect_mt5, disconnect_mt5, get_account_info, get_symbol_info
        
        # Проверяем наличие журнала сделок для инициализации риск-менеджера с историей
        try:
            from trade_journal import TradeJournal
            from trade_executor import get_trade_journal
            from config import SYMBOL
            
            journal = get_trade_journal(SYMBOL, init_if_none=False)
            if journal is not None:
                # Получаем историю сделок для статистики риск-менеджера
                closed_trades = journal.get_closed_trades()
                if not closed_trades.empty:
                    print(f"Найдено {len(closed_trades)} закрытых сделок в журнале")
        except Exception as e:
            print(f"Не удалось загрузить историю сделок: {e}")
        
        # Получаем информацию о счете для инициализации риск-менеджера
        account_info = None
        if connect_mt5():
            try:
                account_info = get_account_info()
                if account_info:
                    print(f"Информация о счете получена: Баланс {account_info['balance']} {account_info['currency']}")
            except Exception as e:
                print(f"Ошибка при получении информации о счете: {e}")
            finally:
                disconnect_mt5()
        
        # Инициализируем риск-менеджер
        account_balance = account_info["balance"] if account_info else 10000
        risk_manager = get_risk_manager(account_balance=account_balance)
        
        # Основной цикл меню
        while True:
            print("\n=== Меню управления рисками ===")
            print("1. Показать текущую статистику")
            print("2. Настроить параметры риск-менеджмента")
            print("3. Рассчитать рекомендуемый размер позиции")
            print("4. Экспортировать статистику")
            print("5. Генерировать отчет по управлению рисками")
            print("6. Сбросить риск-менеджер")
            print("7. Вернуться в главное меню")
            
            choice = input("Выберите опцию (1-7): ").strip()
            
            if choice == "1":
                # Показать текущую статистику
                stats = risk_manager.get_statistics_summary()
                
                print("\n=== Текущая статистика ===")
                print(f"Баланс счета: {stats['account_balance']:.2f}")
                print(f"Пиковый баланс: {stats['peak_balance']:.2f}")
                print(f"Текущая просадка: {stats['current_drawdown']:.2f}%")
                print(f"Всего сделок: {stats['total_trades']}")
                print(f"Винрейт: {stats['win_rate']:.2f}%")
                print(f"Соотношение выигрыш/проигрыш: {stats['win_loss_ratio']:.2f}")
                print(f"Используемый риск на сделку: {stats['risk_per_trade']:.2f}%")
                print(f"Использовано дневного риска: {stats['daily_risk_used']:.2f}%")
                
                print("\n=== Рекомендации ===")
                print(f"Рекомендуемый риск на сделку: {stats.get('recommended_risk', 'Н/Д')}%")
                print(f"Рекомендуемое соотношение R:R: {stats.get('recommended_rr', 'Н/Д')}")
                print(f"Рекомендуемый подход: {stats.get('recommended_approach', 'Н/Д')}")
                print(f"Рекомендуемый метод расчета позиции: {stats.get('recommended_position_method', 'Н/Д')}")
            
            elif choice == "2":
                # Настройка параметров
                print("\n=== Настройка параметров риск-менеджмента ===")
                
                # Риск на сделку
                risk_input = input(f"Риск на сделку (%) [{risk_manager.risk_per_trade*100:.2f}]: ").strip()
                if risk_input:
                    try:
                        risk_percentage = float(risk_input) / 100.0
                        risk_manager.risk_per_trade = max(0.001, min(0.1, risk_percentage))  # Ограничиваем от 0.1% до 10%
                    except ValueError:
                        print("Некорректное значение. Используется предыдущее значение.")
                
                # Макс. дневной риск
                max_daily_risk_input = input(f"Максимальный дневной риск (%) [{risk_manager.max_daily_risk*100:.2f}]: ").strip()
                if max_daily_risk_input:
                    try:
                        max_daily_risk = float(max_daily_risk_input) / 100.0
                        risk_manager.max_daily_risk = max(0.01, min(0.2, max_daily_risk))  # Ограничиваем от 1% до 20%
                    except ValueError:
                        print("Некорректное значение. Используется предыдущее значение.")
                
                # Метод расчета позиции
                print("\nМетоды расчета размера позиции:")
                print("1. fixed_percent - Фиксированный процент (стандартный)")
                print("2. kelly - Формула Келли (адаптивный на основе винрейта)")
                print("3. optimal_f - Оптимальное F (для максимизации роста)")
                print("4. martingale - Мартингейл (увеличение после убытка) - РИСКОВАННО!")
                print("5. anti_martingale - Анти-мартингейл (увеличение после выигрыша)")
                
                method_input = input(f"Выберите метод [текущий: {risk_manager.position_sizing_method}]: ").strip()
                if method_input:
                    method_map = {
                        "1": "fixed_percent",
                        "2": "kelly",
                        "3": "optimal_f",
                        "4": "martingale",
                        "5": "anti_martingale"
                    }
                    
                    if method_input in method_map:
                        new_method = method_map[method_input]
                        
                        # Предупреждение для мартингейла
                        if new_method == "martingale":
                            confirm = input("ВНИМАНИЕ: Мартингейл - высокорисковая стратегия! Подтвердите (y/n): ").lower()
                            if confirm != 'y':
                                print("Метод не изменен.")
                                continue
                        
                        risk_manager.position_sizing_method = new_method
                        print(f"Метод расчета позиции изменен на: {new_method}")
                    else:
                        print("Некорректный выбор метода.")
                
                print("\nПараметры риск-менеджмента обновлены")
            
            elif choice == "3":
                # Расчет рекомендуемого размера позиции
                print("\n=== Расчет размера позиции ===")
                
                # Получение параметров
                symbol_input = input(f"Символ [{SYMBOL}]: ").strip() or SYMBOL
                
                stop_loss_input = input("Размер стоп-лосса (в пипсах): ").strip()
                if not stop_loss_input or not stop_loss_input.isdigit():
                    print("Необходимо указать стоп-лосс в пипсах.")
                    continue
                
                stop_loss_pips = int(stop_loss_input)
                
                # Получаем информацию о символе
                if connect_mt5():
                    try:
                        symbol_info = get_symbol_info(symbol_input)
                        if not symbol_info:
                            print(f"Не удалось получить информацию о символе {symbol_input}")
                            disconnect_mt5()
                            continue
                        
                        # Расчет размера позиции
                        lot_size, risk_amount, risk_percent = risk_manager.calculate_position_size(
                            stop_loss_pips, 0, symbol_info
                        )
                        
                        print(f"\nРезультаты расчета для {symbol_input}:")
                        print(f"Баланс: {risk_manager.account_balance:.2f}")
                        print(f"Стоп-лосс: {stop_loss_pips} пипсов")
                        print(f"Метод расчета: {risk_manager.position_sizing_method}")
                        print(f"Риск: {risk_percent*100:.2f}% = ${risk_amount:.2f}")
                        print(f"Рекомендуемый размер позиции: {lot_size:.2f} лот")
                        
                        # Рассчитываем тейк-профит
                        recommendations = risk_manager.get_trade_recommendations()
                        rr_ratio = recommendations.get("recommended_r_r_ratio", 2.0)
                        tp_pips = risk_manager.calculate_take_profit(stop_loss_pips, rr_ratio)
                        
                        print(f"Рекомендуемый тейк-профит: {tp_pips} пипсов (R/R = {rr_ratio})")
                        
                    except Exception as e:
                        print(f"Ошибка при расчете размера позиции: {e}")
                    finally:
                        disconnect_mt5()
            
            elif choice == "4":
                # Экспорт статистики
                print("\n=== Экспорт статистики ===")
                print("Выберите формат:")
                print("1. JSON")
                print("2. CSV")
                print("3. HTML")
                
                format_choice = input("Формат (1-3): ").strip()
                
                format_map = {"1": "json", "2": "csv", "3": "html"}
                if format_choice in format_map:
                    export_format = format_map[format_choice]
                    try:
                        file_path = risk_manager.export_statistics(format=export_format)
                        print(f"Статистика экспортирована в файл: {file_path}")
                        
                        # Предлагаем открыть HTML-отчет, если он был сгенерирован
                        if export_format == "html" and os.path.exists(file_path):
                            open_file = input("Открыть отчет в браузере? (y/n): ").strip().lower()
                            if open_file == 'y':
                                import webbrowser
                                webbrowser.open(f"file://{os.path.abspath(file_path)}")
                    except Exception as e:
                        print(f"Ошибка при экспорте статистики: {e}")
                else:
                    print("Некорректный выбор формата.")
            
            elif choice == "5":
                # Генерация отчета по управлению рисками
                print("\n=== Генерация отчета по управлению рисками ===")
                
                try:
                    # Получаем данные для отчета
                    stats = risk_manager.get_statistics_summary()
                    equity_curve_data = risk_manager.get_equity_curve()
                    
                    # Генерируем HTML-отчет
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = os.path.join(risk_manager.data_dir, f"risk_report_{timestamp}.html")
                    
                    # Создаем HTML-содержимое (упрощенная версия)
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Отчет по управлению рисками - {timestamp}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2, h3 {{ color: #333; }}
                            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            .positive {{ color: green; }}
                            .negative {{ color: red; }}
                            .warning {{ color: orange; }}
                            .recommendation {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4285f4; margin-bottom: 20px; }}
                        </style>
                    </head>
                    <body>
                        <h1>Отчет по управлению рисками</h1>
                        <p>Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        
                        <h2>Текущее состояние</h2>
                        <table>
                            <tr><th>Показатель</th><th>Значение</th></tr>
                            <tr><td>Баланс счета</td><td>{stats['account_balance']:.2f}</td></tr>
                            <tr><td>Пиковый баланс</td><td>{stats['peak_balance']:.2f}</td></tr>
                            <tr><td>Текущая просадка</td><td class="{'negative' if stats['current_drawdown'] > 10 else 'warning' if stats['current_drawdown'] > 5 else ''}">{stats['current_drawdown']:.2f}%</td></tr>
                        </table>
                        
                        <h2>Статистика торговли</h2>
                        <table>
                            <tr><th>Показатель</th><th>Значение</th></tr>
                            <tr><td>Всего сделок</td><td>{stats['total_trades']}</td></tr>
                            <tr><td>Выигрышные сделки</td><td>{stats['winning_trades']}</td></tr>
                            <tr><td>Проигрышные сделки</td><td>{stats['losing_trades']}</td></tr>
                            <tr><td>Винрейт</td><td class="{'positive' if stats['win_rate'] > 50 else 'warning' if stats['win_rate'] > 40 else 'negative'}">{stats['win_rate']:.2f}%</td></tr>
                            <tr><td>Соотношение выигрыш/проигрыш</td><td class="{'positive' if stats['win_loss_ratio'] > 1 else 'negative'}">{stats['win_loss_ratio']:.2f}</td></tr>
                            <tr><td>Серия выигрышей</td><td>{stats['consecutive_wins']}</td></tr>
                            <tr><td>Серия проигрышей</td><td>{stats['consecutive_losses']}</td></tr>
                        </table>
                        
                        <h2>Параметры риска</h2>
                        <table>
                            <tr><th>Показатель</th><th>Значение</th></tr>
                            <tr><td>Риск на сделку</td><td>{stats['risk_per_trade']:.2f}%</td></tr>
                            <tr><td>Метод расчета позиции</td><td>{stats['position_sizing_method']}</td></tr>
                            <tr><td>Использованный дневной риск</td><td>{stats['daily_risk_used']:.2f}%</td></tr>
                            <tr><td>Лимит дневного риска</td><td>{stats['daily_risk_limit']:.2f}%</td></tr>
                        </table>
                        
                        <h2>Рекомендации по управлению рисками</h2>
                        <div class="recommendation">
                            <h3>Ключевые рекомендации</h3>
                            <p><strong>Риск на сделку:</strong> {stats.get('recommended_risk', '1.0')}%</p>
                            <p><strong>Соотношение риск/доходность:</strong> {stats.get('recommended_rr', '2.0')}</p>
                            <p><strong>Рекомендуемый подход:</strong> {stats.get('recommended_approach', 'Нет данных')}</p>
                            <p><strong>Метод расчета позиции:</strong> {stats.get('recommended_position_method', 'fixed_percent')}</p>
                        </div>
                    </body>
                    </html>
                    """
                    
                    # Сохраняем отчет
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    print(f"Отчет сгенерирован и сохранен в: {report_file}")
                    
                    # Предлагаем открыть отчет
                    open_report = input("Открыть отчет в браузере? (y/n): ").strip().lower()
                    if open_report == 'y':
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(report_file)}")
                
                except Exception as e:
                    print(f"Ошибка при генерации отчета: {e}")
            
            elif choice == "6":
                # Сброс риск-менеджера
                confirm = input("Вы уверены, что хотите сбросить риск-менеджер? Вся статистика будет потеряна. (y/n): ").strip().lower()
                if confirm == 'y':
                    reset_risk_manager()
                    print("Риск-менеджер сброшен. Перезапустите меню для инициализации нового экземпляра.")
                    break
                else:
                    print("Операция отменена.")
            
            elif choice == "7":
                print("Возврат в главное меню")
                break
            
            else:
                print("Некорректный выбор. Пожалуйста, выберите 1-7.")
    
    except ImportError:
        print("Модуль риск-менеджера недоступен. Пожалуйста, убедитесь, что файл risk_manager.py находится в директории проекта.")
    
    except Exception as e:
        logging.error(f"Ошибка в меню риск-менеджера: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Произошла ошибка: {e}")

def main():
    """Основной метод запуска бота"""
    try:
        # Выводим текущие значения для отладки
        mode_env = os.environ.get('MODE', 'не задано')
        mode_config = MODE
        print(f"Значение MODE из системы: {mode_env}")
        print(f"Значение MODE из config.py: {mode_config}")
        
        # Проверка режима работы из конфига
        print(f"\nТЕКУЩИЙ РЕЖИМ РАБОТЫ: {MODE}")
        if MODE.lower() not in ["backtest", "live"]:
            print(f"ОШИБКА: Неизвестный режим работы '{MODE}'. Доступны только 'backtest' или 'live'.")
            return False
            
        # Добавляем явное меню независимо от режима
        if MODE.lower() == "backtest":
            # Предлагаем выбрать между стандартным и оптимизированным бэктестом
            if OPTIMIZED_BACKTEST_AVAILABLE:
                while True:
                    print("\nВыберите действие:")
                    print("1. Стандартный бэктест")
                    print("2. Оптимизированный бэктест (быстрее)")
                    print("3. Визуализация результатов")
                    print("4. Управление рисками и капиталом")
                    print("5. Выход")
                    
                    try:
                        choice = input("Ваш выбор (1-5): ").strip()
                        
                        if choice == "1":
                            run_backtest(optimized=False)
                        elif choice == "2":
                            run_backtest(optimized=True)
                        elif choice == "3":
                            run_visualization_menu()
                        elif choice == "4":
                            run_risk_manager_menu()
                        elif choice == "5":
                            print("Выход из программы")
                            break
                        else:
                            print("Некорректный выбор. Пожалуйста, выберите 1-5.")
                    except Exception as e:
                        logging.error(f"Ошибка при выполнении действия: {str(e)}")
                        logging.error(traceback.format_exc())
                        print(f"Произошла ошибка: {str(e)}")
            else:
                # Если оптимизированный бэктест недоступен, просто запускаем стандартный
                run_backtest(optimized=False)
                
        elif MODE.lower() == "live":
            while True:
                print("\nВыберите действие:")
                print("1. Запустить торговлю")
                print("2. Журнал сделок и статистика")
                print("3. Управление рисками и капиталом")
                print("4. Выход")
                
                try:
                    choice = input("Ваш выбор (1-4): ").strip()
                    
                    if choice == "1":
                        run_live_trading()
                    elif choice == "2":
                        run_trade_journal_menu()
                    elif choice == "3":
                        run_risk_manager_menu()
                    elif choice == "4":
                        print("Выход из программы")
                        break
                    else:
                        print("Некорректный выбор. Пожалуйста, выберите 1-4.")
                except Exception as e:
                    logging.error(f"Ошибка при выполнении действия: {str(e)}")
                    logging.error(traceback.format_exc())
                    print(f"Произошла ошибка: {str(e)}")
        else:
            logging.error(f"Неизвестный режим работы: {MODE}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Критическая ошибка при запуске: {str(e)}")
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AlgoTrade Bot')
    parser.add_argument('--mode', choices=['live', 'backtest'], 
                        help='Режим работы бота (live или backtest)')
    args = parser.parse_args()
    
    # Если указан аргумент --mode, переопределяем настройку из конфига
    if args.mode:
        os.environ['MODE'] = args.mode
        print(f"Режим работы установлен через командную строку: {args.mode}")
    
    main()