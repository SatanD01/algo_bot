import time
import logging
import os
import sys
import traceback
import argparse
from datetime import datetime, timedelta
from config import CHECK_INTERVAL, MODE, SYMBOL, LOG_LEVEL, is_trading_allowed, MAX_POSITIONS
from mt5_connector import connect_mt5, disconnect_mt5, get_account_info, get_open_positions
from trade_executor import execute_trade
from backtest import backtest
try:
    from backtest_optimized import optimized_backtest
    OPTIMIZED_BACKTEST_AVAILABLE = True
except ImportError:
    OPTIMIZED_BACKTEST_AVAILABLE = False

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
    
    # Инициализируем расширенный журнал сделок, если он доступен
    try:
        from trade_journal import TradeJournal
        from trade_executor import get_trade_journal
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
                
                # Получаем текущие открытые позиции для проверки перед выполнением торговой логики
                current_positions = get_open_positions(symbol=SYMBOL)
                
                # Проверяем, есть ли уже открытые позиции для текущего символа
                # и допустимо ли открытие новых позиций
                if len(current_positions) < MAX_POSITIONS:
                    # Извлекаем типы текущих открытых позиций для символа
                    current_types = [pos['type'] for pos in current_positions]
                    
                    # Выполняем торговую логику
                    # Если есть сигнал, он будет проверен на соответствие с текущими открытыми позициями
                    signal = execute_trade()
                    
                    # Если был открыт новый ордер, обновляем информацию в tracking
                    if signal:
                        symbol = signal['symbol']
                        if symbol not in open_positions_tracking:
                            open_positions_tracking[symbol] = []
                        open_positions_tracking[symbol].append({'ticket': signal['ticket'], 'type': signal['type']})
                else:
                    logging.info(f"Достигнуто максимальное количество позиций для {SYMBOL}: {len(current_positions)}/{MAX_POSITIONS}")
                
                # Логируем активность каждый час
                current_time = time.time()
                if current_time - last_check_time >= 3600:  # 1 час
                    positions = get_open_positions()
                    logging.info(f"Бот активен. Открыто позиций: {len(positions)}")
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

def main():
    """Основной метод запуска бота"""
    try:
        # Выводим информацию о запуске
        logging.info(f"Запуск AlgoTrade бота (Версия 1.0, Режим: {MODE})")

        # Инициализируем Telegram-нотификатор, если включен
        try:
            from telegram_notifier import initialize_telegram_notifier
            from config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            
            if TELEGRAM_ENABLED:
                notifier = initialize_telegram_notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                if notifier.enabled:
                    logging.info("Telegram-уведомления включены")
                else:
                    logging.warning("Telegram-уведомления отключены: не указан токен бота или ID чата")
        except Exception as e:
            logging.warning(f"Не удалось инициализировать Telegram-нотификатор: {str(e)}")
        
        # Запускаем соответствующий режим
        if MODE.lower() == "backtest":
            # Предлагаем выбрать между стандартным и оптимизированным бэктестом
            if OPTIMIZED_BACKTEST_AVAILABLE:
                while True:
                    print("\nВыберите действие:")
                    print("1. Стандартный бэктест")
                    print("2. Оптимизированный бэктест (быстрее)")
                    print("3. Визуализация результатов")
                    print("4. Журнал сделок и статистика")
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
                            run_trade_journal_menu()
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
                print("3. Выход")
                
                try:
                    choice = input("Ваш выбор (1-3): ").strip()
                    
                    if choice == "1":
                        run_live_trading()
                    elif choice == "2":
                        run_trade_journal_menu()
                    elif choice == "3":
                        print("Выход из программы")
                        break
                    else:
                        print("Некорректный выбор. Пожалуйста, выберите 1-3.")
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
    main()