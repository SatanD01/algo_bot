import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import time
import logging
import argparse
import json
import webbrowser
import threading
import gc

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Глобальные настройки
SYMBOL = None
THEME = "white"  # white или dark
AUTO_OPEN = True  # Автоматически открывать браузер
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")

# Создаем директорию для результатов, если она не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def load_data(file_path, optimize=True):
    """
    Загружает и оптимизирует данные из CSV файла
    
    Параметры:
    file_path (str): Путь к файлу CSV
    optimize (bool): Оптимизировать ли данные для экономии памяти
    
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
        
        # Загружаем данные
        df = pd.read_csv(file_path, dtype=dtype_dict if optimize else None)
        
        # Преобразуем временные колонки
        date_columns = [col for col in df.columns if 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
        
        # Оптимизируем численные столбцы
        if optimize:
            for col in df.columns:
                # Преобразуем int64 в int32 или int16, если возможно
                if pd.api.types.is_integer_dtype(df[col]):
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                # Преобразуем float64 в float32, если возможно
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = df[col].astype(np.float32)
        
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла {file_path}: {str(e)}")
        logger.exception(e)
        return None

def load_price_data(symbol, timeframe, start_date, end_date):
    """
    Загружает данные цен из MT5
    
    Параметры:
    symbol (str): Символ
    timeframe (str): Таймфрейм (M5, M15, H1 и т.д.)
    start_date (datetime): Начальная дата
    end_date (datetime): Конечная дата
    
    Возвращает:
    DataFrame: Загруженные данные цен или None в случае ошибки
    """
    try:
        # Импортируем функцию для загрузки данных
        from data_fetcher import get_historical_data
        from mt5_connector import connect_mt5, disconnect_mt5
        
        if not connect_mt5():
            logger.error("Не удалось подключиться к MT5")
            return None
        
        logger.info(f"Загрузка данных для {symbol} ({timeframe}) с {start_date} по {end_date}")
        
        try:
            df = get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None:
                logger.info(f"Загружено {len(df)} свечей для {timeframe}")
            else:
                logger.error(f"Не удалось загрузить данные для {timeframe}")
            
            return df
        finally:
            disconnect_mt5()
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных цен: {str(e)}")
        logger.exception(e)
        return None

def create_balance_chart(results, theme=THEME):
    """
    Создает график баланса
    
    Параметры:
    results (DataFrame): Данные результатов бэктеста
    theme (str): Тема оформления ('white' или 'dark')
    
    Возвращает:
    go.Figure: Объект графика Plotly
    """
    try:
        # Создаем фигуру с двумя осями Y
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Вычисляем начальный баланс
        initial_balance = results['balance'].iloc[0] - results['profit'].iloc[0]
        
        # Добавляем линию баланса
        fig.add_trace(
            go.Scatter(
                x=results['exit_time'],
                y=results['balance'],
                mode='lines',
                name='Баланс',
                line=dict(color='blue', width=2, shape='spline', smoothing=0.3)
            )
        )
        
        # Добавляем накопленную прибыль
        results['cumulative_profit'] = results['profit'].cumsum() + initial_balance
        fig.add_trace(
            go.Scatter(
                x=results['exit_time'],
                y=results['cumulative_profit'],
                mode='lines',
                name='Накопленная прибыль',
                line=dict(color='orange', width=1.5, dash='dash', shape='spline', smoothing=0.3)
            )
        )
        
        # Добавляем точки выигрышей и проигрышей
        wins = results[results['result'] == 'win']
        losses = results[results['result'] == 'loss']
        
        if not wins.empty:
            fig.add_trace(
                go.Scatter(
                    x=wins['exit_time'],
                    y=wins['balance'],
                    mode='markers',
                    name='Выигрыши',
                    marker=dict(color='green', size=8, symbol='circle')
                )
            )
        
        if not losses.empty:
            fig.add_trace(
                go.Scatter(
                    x=losses['exit_time'],
                    y=losses['balance'],
                    mode='markers',
                    name='Проигрыши',
                    marker=dict(color='red', size=8, symbol='x')
                )
            )
        
        # Добавляем просадку на втором Y
        results['drawdown'] = (results['cumulative_profit'].cummax() - results['cumulative_profit']) / results['cumulative_profit'].cummax() * 100
        
        fig.add_trace(
            go.Scatter(
                x=results['exit_time'],
                y=results['drawdown'],
                mode='lines',
                name='Просадка (%)',
                line=dict(color='red', width=1.5),
                visible='legendonly'  # Скрыто по умолчанию
            ),
            secondary_y=True
        )
        
        # Добавляем аннотацию с итоговой статистикой
        total_trades = len(results)
        wins_count = len(wins)
        winrate = (wins_count / total_trades * 100) if total_trades > 0 else 0
        final_balance = results['balance'].iloc[-1] if not results.empty else 0
        profit_percent = ((final_balance / initial_balance) - 1) * 100
        max_drawdown = results['drawdown'].max()
        
        stats_text = (
            f"Всего сделок: {total_trades}<br>"
            f"Выигрышей: {wins_count} ({winrate:.2f}%)<br>"
            f"Начальный баланс: {initial_balance:.2f}<br>"
            f"Конечный баланс: {final_balance:.2f}<br>"
            f"Прибыль: {profit_percent:.2f}%<br>"
            f"Макс. просадка: {max_drawdown:.2f}%"
        )
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)" if theme == "white" else "rgba(50, 50, 50, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="left"
        )
        
        # Настраиваем макет
        fig.update_layout(
            title=f'Динамика баланса',
            xaxis_title='Время',
            yaxis_title='Баланс',
            yaxis2_title='Просадка (%)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template=f"plotly_{theme}"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Ошибка при создании графика баланса: {str(e)}")
        logger.exception(e)
        return None

def create_trades_distribution(results, theme=THEME):
    """
    Создает графики распределения сделок
    
    Параметры:
    results (DataFrame): Данные результатов бэктеста
    theme (str): Тема оформления ('white' или 'dark')
    
    Возвращает:
    go.Figure: Объект графика Plotly
    """
    try:
        # Создаем фигуру с 6 подграфиками
        fig = make_subplots(
            rows=2, 
            cols=3,
            subplot_titles=(
                'Распределение по типу ордера', 
                'Распределение по результату',
                'Распределение по таймфреймам',
                'Распределение прибыли',
                'Распределение по сетапам',
                'Винрейт по сетапам (%)'
            ),
            specs=[
                [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # 1. Распределение по типу ордера (Buy/Sell)
        if 'order' in results.columns:
            order_counts = results['order'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=order_counts.index,
                    values=order_counts.values,
                    hole=.3,
                    marker_colors=['royalblue', 'tomato']
                ),
                row=1, col=1
            )
        
        # 2. Распределение по результату (Win/Loss)
        if 'result' in results.columns:
            result_counts = results['result'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=result_counts.index,
                    values=result_counts.values,
                    hole=.3,
                    marker_colors=['limegreen', 'crimson']
                ),
                row=1, col=2
            )
        
        # 3. Распределение по таймфреймам
        if 'tf' in results.columns:
            tf_counts = results['tf'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=tf_counts.index,
                    values=tf_counts.values,
                   hole=.3
               ),
               row=1, col=3
           )
       
        # 4. Распределение прибыли
        if 'profit' in results.columns:
            profit_data = results['profit'].copy()
            # Ограничиваем выбросы для лучшей визуализации
            q1, q3 = np.percentile(profit_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            profit_data = profit_data[(profit_data >= lower_bound) & (profit_data <= upper_bound)]
            
            fig.add_trace(
                go.Histogram(
                    x=profit_data,
                    nbinsx=20,
                    marker_color='seagreen'
                ),
                row=2, col=1
            )
        
        # 5. Распределение по сетапам
        if 'setup' in results.columns:
            setup_counts = results['setup'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=setup_counts.index,
                    y=setup_counts.values,
                    marker_color='slateblue'
                ),
                row=2, col=2
            )
        
        # 6. Винрейт по сетапам
        if 'setup' in results.columns and 'result' in results.columns:
            # Рассчитываем винрейт для каждого сетапа
            setup_stats = results.groupby('setup').apply(
                lambda x: pd.Series({
                    'total': len(x),
                    'wins': (x['result'] == 'win').sum(),
                    'winrate': 100 * (x['result'] == 'win').sum() / len(x) if len(x) > 0 else 0
                })
            ).reset_index()
            
            setup_stats = setup_stats.sort_values('winrate', ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=setup_stats['setup'],
                    y=setup_stats['winrate'],
                    marker_color='darkorange',
                    text=setup_stats['winrate'].round(1).astype(str) + '%',
                    textposition='auto'
                ),
                row=2, col=3
            )
            
            # Добавляем линию среднего винрейта
            avg_winrate = 100 * results['result'].value_counts(normalize=True).get('win', 0)
            
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=avg_winrate,
                x1=len(setup_stats)-0.5,
                y1=avg_winrate,
                line=dict(color="red", width=2, dash="dash"),
                row=2, col=3
            )
            
            fig.add_annotation(
                x=len(setup_stats)-1,
                y=avg_winrate,
                text=f"Средний: {avg_winrate:.1f}%",
                showarrow=False,
                font=dict(color="red"),
                row=2, col=3
            )
        
        # Настраиваем макет
        fig.update_layout(
            height=800,
            title='Распределение сделок',
            showlegend=True,
            template=f"plotly_{theme}"
        )
        
        # Обновляем оси для лучшей видимости
        fig.update_xaxes(tickangle=45, row=2, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=3)
        
        return fig
    
    except Exception as e:
        logger.error(f"Ошибка при создании графиков распределения сделок: {str(e)}")
        logger.exception(e)
        return None

def create_monthly_analysis(results, theme=THEME):
    """
    Создает графики анализа по месяцам
    
    Параметры:
    results (DataFrame): Данные результатов бэктеста
    theme (str): Тема оформления ('white' или 'dark')
    
    Возвращает:
    go.Figure: Объект графика Plotly
    """
    try:
        # Проверяем наличие необходимых столбцов
        if 'entry_time' not in results.columns:
            logger.warning("В данных отсутствует столбец 'entry_time', анализ по месяцам невозможен")
            return None
        
        # Создаем копию данных для обработки
        df = results.copy()
        
        # Добавляем столбцы для месяца и года
        df['year'] = df['entry_time'].dt.year
        df['month'] = df['entry_time'].dt.month
        
        # Группируем по году и месяцу
        monthly_stats = df.groupby(['year', 'month']).agg(
            trades=('result', 'count'),
            wins=('result', lambda x: sum(x == 'win')),
            losses=('result', lambda x: sum(x == 'loss')),
            winrate=('result', lambda x: 100 * sum(x == 'win') / len(x) if len(x) > 0 else 0),
            profit=('profit', 'sum'),
            gross_profit=('profit', lambda x: x[x > 0].sum() if any(x > 0) else 0),
            gross_loss=('profit', lambda x: abs(x[x < 0].sum()) if any(x < 0) else 0)
        ).reset_index()
        
        # Добавляем профит-фактор
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
        
        # Создаем фигуру с 3 подграфиками
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
        avg_winrate = df['result'].value_counts(normalize=True).get('win', 0) * 100
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
            height=800,
            title='Месячный анализ',
            showlegend=False,
            template=f"plotly_{theme}"
        )
        
        # Обновляем оси X для всех подграфиков
        for i in range(1, 4):
            fig.update_xaxes(tickangle=45, row=i, col=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"Ошибка при создании анализа по месяцам: {str(e)}")
        logger.exception(e)
        return None

def create_price_chart(price_data, entries, exits, stop_losses, take_profits, current_time=None, theme=THEME, autoscale=True):
    """
    Создает свечной график с отметками сделок
    
    Параметры:
    price_data (DataFrame): Данные цен
    entries (DataFrame): Данные о входах
    exits (DataFrame): Данные о выходах
    stop_losses (DataFrame): Данные о стоп-лоссах
    take_profits (DataFrame): Данные о тейк-профитах
    current_time (datetime, optional): Текущее время для отображения
    theme (str): Тема оформления ('white' или 'dark')
    autoscale (bool): Автоматически масштабировать ось Y
    
    Возвращает:
    go.Figure: Объект графика Plotly
    """
    try:
        if price_data is None or price_data.empty:
            logger.warning("Нет данных для построения свечного графика")
            return None
        
        # Определяем видимый диапазон дат
        if current_time is None:
            # Если current_time не указано (режим "Показать все сделки"),
            # показываем весь диапазон данных
            visible_range = [price_data['time'].min(), price_data['time'].max()]
            
            # В режиме "Показать все сделки" используем все данные без фильтрации по времени
            entries_in_range = entries
            exits_in_range = exits
            sl_in_range = stop_losses
            tp_in_range = take_profits
        else:
            # Показываем данные вокруг текущего времени (пошаговый режим)
            # Убедимся, что current_time - объект datetime
            if not isinstance(current_time, (datetime, pd.Timestamp)):
                try:
                    current_time = pd.to_datetime(current_time)
                except:
                    logger.warning(f"Не удалось преобразовать current_time в datetime: {current_time}")
                    visible_range = [price_data['time'].min(), price_data['time'].max()]
                    
            # Находим индекс ближайшей свечи к текущему времени
            try:
                price_data_times = price_data['time'].copy()
                # Убедимся, что все данные в формате datetime
                if not pd.api.types.is_datetime64_any_dtype(price_data_times):
                    price_data_times = pd.to_datetime(price_data_times)
                
                idx = price_data_times.searchsorted(current_time)
                idx = min(idx, len(price_data) - 1)
                
                # Определяем видимый диапазон как 100 свечей вокруг текущего времени
                start_idx = max(0, idx - 50)
                end_idx = min(len(price_data) - 1, idx + 50)
                
                visible_range = [price_data['time'].iloc[start_idx], price_data['time'].iloc[end_idx]]
            except Exception as e:
                logger.warning(f"Ошибка при определении видимого диапазона: {e}")
                visible_range = [price_data['time'].min(), price_data['time'].max()]
            
            # В пошаговом режиме фильтруем данные по текущему времени
            if entries is not None and not entries.empty:
                entries_in_range = entries[entries['time'] <= current_time]
            else:
                entries_in_range = entries
                
            if exits is not None and not exits.empty:
                exits_in_range = exits[exits['time'] <= current_time]
            else:
                exits_in_range = exits
            
            if stop_losses is not None and not stop_losses.empty:
                sl_in_range = stop_losses[stop_losses['time'] <= current_time]
            else:
                sl_in_range = stop_losses
                
            if take_profits is not None and not take_profits.empty:
                tp_in_range = take_profits[take_profits['time'] <= current_time]
            else:
                tp_in_range = take_profits
        
        # Создаем свечной график
        fig = go.Figure()
        
        fig.add_trace(
            go.Candlestick(
                x=price_data['time'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Цена',
                increasing_line_color='green',
                decreasing_line_color='red'
            )
        )
        
        # Добавляем вертикальную линию для текущего времени, только если указано current_time
        if current_time is not None:
            # Вместо использования add_vline, используем add_shape для создания линии
            # с явным форматированием временной метки
            fig.add_shape(
                type="line",
                x0=current_time,
                y0=0,
                x1=current_time,
                y1=1,
                line=dict(color="gold", width=2, dash="dash"),
                xref="x",
                yref="paper"
            )
            
            # Добавляем аннотацию отдельно
            fig.add_annotation(
                x=current_time,
                y=1,
                text="Текущее время",
                showarrow=False,
                yshift=10,
                xshift=10,
                bgcolor="gold",
                opacity=0.7,
                bordercolor="black",
                borderwidth=1
            )
        
        # Дополнительно фильтруем данные по видимому диапазону для отображения на графике
        # Добавляем отметки для входов
        if entries_in_range is not None and not entries_in_range.empty:
            # Фильтруем по видимому диапазону
            entries_filtered = entries_in_range[(entries_in_range['time'] >= visible_range[0]) & 
                                                (entries_in_range['time'] <= visible_range[1])]
            
            # Группируем по типу ордера
            buy_entries = entries_filtered[entries_filtered['order'] == 'buy']
            sell_entries = entries_filtered[entries_filtered['order'] == 'sell']
            
            if not buy_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_entries['time'],
                        y=buy_entries['price'],
                        mode='markers',
                        name='Buy Entry',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        ),
                        text=[f"Buy: {setup}" for setup in buy_entries.get('setup', 'Standard')],
                        hoverinfo='text'
                    )
                )
            
            if not sell_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_entries['time'],
                        y=sell_entries['price'],
                        mode='markers',
                        name='Sell Entry',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='red',
                            line=dict(width=1, color='darkred')
                        ),
                        text=[f"Sell: {setup}" for setup in sell_entries.get('setup', 'Standard')],
                        hoverinfo='text'
                    )
                )
        
        # Добавляем отметки для выходов
        if exits_in_range is not None and not exits_in_range.empty:
            # Фильтруем по видимому диапазону
            exits_filtered = exits_in_range[(exits_in_range['time'] >= visible_range[0]) & 
                                           (exits_in_range['time'] <= visible_range[1])]
            
            # Группируем по типу выхода
            tp_exits = exits_filtered[exits_filtered['type'] == 'take_profit']
            sl_exits = exits_filtered[exits_filtered['type'] == 'stop_loss']
            other_exits = exits_filtered[~exits_filtered['type'].isin(['take_profit', 'stop_loss'])]
            
            if not tp_exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=tp_exits['time'],
                        y=tp_exits['price'],
                        mode='markers',
                        name='Take Profit',
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color='lime',
                            line=dict(width=1, color='green')
                        ),
                        text=[f"TP: {profit:.2f}" for profit in tp_exits.get('profit', 0)],
                        hoverinfo='text'
                    )
                )
            
            if not sl_exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sl_exits['time'],
                        y=sl_exits['price'],
                        mode='markers',
                        name='Stop Loss',
                        marker=dict(
                            symbol='x',
                            size=8,
                            color='red',
                            line=dict(width=1, color='darkred')
                        ),
                        text=[f"SL: {profit:.2f}" for profit in sl_exits.get('profit', 0)],
                        hoverinfo='text'
                    )
                )
            
            if not other_exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=other_exits['time'],
                        y=other_exits['price'],
                        mode='markers',
                        name='Other Exit',
                        marker=dict(
                            symbol='square',
                            size=8,
                            color='blue',
                            line=dict(width=1, color='darkblue')
                        ),
                        text=[f"Exit: {profit:.2f}" for profit in other_exits.get('profit', 0)],
                        hoverinfo='text'
                    )
                )
        
        # Добавляем линии для стоп-лоссов и тейк-профитов активных сделок
        if entries_in_range is not None and not entries_in_range.empty and sl_in_range is not None and not sl_in_range.empty:
            # Фильтруем стоп-лоссы по видимому диапазону
            sl_filtered = sl_in_range[(sl_in_range['time'] >= visible_range[0]) & 
                                     (sl_in_range['time'] <= visible_range[1])]
            
            for _, sl in sl_filtered.iterrows():
                # Находим соответствующий вход
                entry_time = sl['time']
                
                # Добавляем горизонтальную линию для стоп-лосса
                fig.add_shape(
                    type="line",
                    x0=entry_time,
                    y0=sl['price'],
                    x1=visible_range[1],
                    y1=sl['price'],
                    line=dict(
                        color="red",
                        width=1,
                        dash="dash"
                    )
                )
                
                # Добавляем метку "SL" у линии
                fig.add_annotation(
                    x=entry_time,
                    y=sl['price'],
                    text="SL",
                    showarrow=False,
                    font=dict(color="red", size=10),
                    bgcolor="white" if theme == "white" else "black",
                    bordercolor="red",
                    borderwidth=1,
                    xanchor="center"
                )
        
        if entries_in_range is not None and not entries_in_range.empty and tp_in_range is not None and not tp_in_range.empty:
            # Фильтруем тейк-профиты по видимому диапазону
            tp_filtered = tp_in_range[(tp_in_range['time'] >= visible_range[0]) & 
                                     (tp_in_range['time'] <= visible_range[1])]
            
            for _, tp in tp_filtered.iterrows():
                # Находим соответствующий вход
                entry_time = tp['time']
                
                # Добавляем горизонтальную линию для тейк-профита
                fig.add_shape(
                    type="line",
                    x0=entry_time,
                    y0=tp['price'],
                    x1=visible_range[1],
                    y1=tp['price'],
                    line=dict(
                        color="green",
                        width=1,
                        dash="dash"
                    )
                )
                
                # Добавляем метку "TP" у линии
                fig.add_annotation(
                    x=entry_time,
                    y=tp['price'],
                    text="TP",
                    showarrow=False,
                    font=dict(color="green", size=10),
                    bgcolor="white" if theme == "white" else "black",
                    bordercolor="green",
                    borderwidth=1,
                    xanchor="center"
                )
        
        # Настраиваем макет
        symbol = price_data.get('symbol', ['SYMBOL'])[0] if 'symbol' in price_data.columns else "SYMBOL"
        timeframe = "Unknown"
        if 'time' in price_data.columns:
            # Попытка определить таймфрейм по интервалу между свечами
            if len(price_data) >= 2:
                time_diff = (price_data['time'].iloc[1] - price_data['time'].iloc[0]).total_seconds()
                if time_diff == 300:  # 5 минут
                    timeframe = "M5"
                elif time_diff == 900:  # 15 минут
                    timeframe = "M15"
                elif time_diff == 3600:  # 1 час
                    timeframe = "H1"
                elif time_diff == 14400:  # 4 часа
                    timeframe = "H4"
                elif time_diff == 86400:  # 1 день
                    timeframe = "D1"
        
        # Заголовок графика зависит от режима просмотра
        title_text = f'График {symbol} ({timeframe})'
        if current_time is None:
            title_text += ' - Все сделки'
        
        # Настраиваем макет
        fig.update_layout(
            title=title_text,
            xaxis_title='Время',
            yaxis_title='Цена',
            xaxis_rangeslider_visible=False,
            template=f"plotly_{theme}",
            height=700,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Устанавливаем видимый диапазон по X
        fig.update_xaxes(range=[visible_range[0], visible_range[1]])

        # Автоматически масштабируем ось Y, если требуется
        if autoscale:
            # Находим минимум и максимум цены в видимом диапазоне
            if price_data is not None and not price_data.empty:
                visible_data = price_data[(price_data['time'] >= visible_range[0]) & (price_data['time'] <= visible_range[1])]
                
                if not visible_data.empty:
                    min_price = visible_data['low'].min()
                    max_price = visible_data['high'].max()
                    
                    # Добавляем небольшой отступ для лучшего отображения
                    padding = (max_price - min_price) * 0.05
                    fig.update_yaxes(range=[min_price - padding, max_price + padding])

        return fig
    
    except Exception as e:
        logger.error(f"Ошибка при создании свечного графика: {str(e)}")
        logger.exception(e)
        return None

def create_setup_analysis(results, theme=THEME):
    """
    Создает график анализа сетапов
    
    Параметры:
    results (DataFrame): Данные результатов бэктеста
    theme (str): Тема оформления ('white' или 'dark')
    
    Возвращает:
    go.Figure: Объект графика Plotly
    """
    try:
        if 'setup' not in results.columns:
            logger.warning("В данных отсутствует информация о сетапах")
            return None
        
        # Рассчитываем статистику по сетапам
        setup_stats = results.groupby('setup').agg(
            trades=('result', 'count'),
            wins=('result', lambda x: sum(x == 'win')),
            losses=('result', lambda x: sum(x == 'loss')),
            winrate=('result', lambda x: 100 * sum(x == 'win') / len(x) if len(x) > 0 else 0),
            gross_profit=('profit', lambda x: x[x > 0].sum() if any(x > 0) else 0),
            gross_loss=('profit', lambda x: abs(x[x < 0].sum()) if any(x < 0) else 0),
            avg_win=('profit', lambda x: x[x > 0].mean() if any(x > 0) else 0),
            avg_loss=('profit', lambda x: x[x < 0].mean() if any(x < 0) else 0),
            profit=('profit', 'sum')
        ).reset_index()
        
        # Добавляем профит-фактор
        setup_stats['profit_factor'] = setup_stats.apply(
            lambda row: row['gross_profit'] / row['gross_loss'] 
                       if row['gross_loss'] > 0 else float('inf'), 
            axis=1
        )
        
        # Добавляем ожидание (expectancy)
        setup_stats['expectancy'] = setup_stats.apply(
            lambda row: (row['winrate']/100 * row['avg_win'] + (1-row['winrate']/100) * row['avg_loss']),
            axis=1
        )
        
        # Сортируем по профит-фактору
        setup_stats = setup_stats.sort_values('profit_factor', ascending=False)
        
        # Создаем фигуру с 4 подграфиками
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
        # Ограничиваем бесконечные значения для отображения
        pf_values = []
        for pf in setup_stats['profit_factor']:
            if pf == float('inf'):
                pf_values.append(10)  # Ограничиваем бесконечность для отображения
            else:
                pf_values.append(min(pf, 10))  # Ограничиваем максимальное значение для лучшего отображения
        
        fig.add_trace(
            go.Bar(
                x=setup_stats['setup'],
                y=pf_values,
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
        avg_winrate = results['result'].value_counts(normalize=True).get('win', 0) * 100
        
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=avg_winrate,
            x1=len(setup_stats['setup'])-0.5,
            y1=avg_winrate,
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=2
        )
        
        fig.add_annotation(
            x=len(setup_stats['setup'])-1,
            y=avg_winrate,
            text=f"Средний: {avg_winrate:.1f}%",
            showarrow=False,
            font=dict(color="red"),
            row=2, col=2
        )
        
        # Настраиваем макет
        fig.update_layout(
            title='Анализ сетапов',
            height=800,
            showlegend=True,
            template=f"plotly_{theme}"
        )
        
        # Обновляем оси X для всех подграфиков
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        return fig
    
    except Exception as e:
        logger.error(f"Ошибка при создании анализа сетапов: {str(e)}")
        logger.exception(e)
        return None

def create_summary_html(results, entries, exits, stop_losses, take_profits, price_data=None):
    """
    Создает HTML-файл с итоговыми результатами и графиками
    
    Параметры:
    results (DataFrame): Данные результатов бэктеста
    entries (DataFrame): Данные о входах
    exits (DataFrame): Данные о выходах
    stop_losses (DataFrame): Данные о стоп-лоссах
    take_profits (DataFrame): Данные о тейк-профитах
    price_data (dict, optional): Словарь с данными цен по таймфреймам
    
    Возвращает:
    str: Путь к созданному HTML-файлу или None в случае ошибки
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        symbol = SYMBOL or "SYMBOL"
        output_file = os.path.join(OUTPUT_DIR, f"backtest_results_{symbol}_{timestamp}.html")
        
        logger.info(f"Создание HTML-отчета: {output_file}")
        
        # Создаем различные графики
        balance_chart = create_balance_chart(results)
        trades_chart = create_trades_distribution(results)
        monthly_chart = create_monthly_analysis(results)
        setup_chart = create_setup_analysis(results)
        
        # Конвертируем графики в HTML
        charts_html = []
        
        if balance_chart:
            charts_html.append(f"<div class='chart-container'>{balance_chart.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        if trades_chart:
            charts_html.append(f"<div class='chart-container'>{trades_chart.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        if monthly_chart:
            charts_html.append(f"<div class='chart-container'>{monthly_chart.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        if setup_chart:
            charts_html.append(f"<div class='chart-container'>{setup_chart.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        # Добавляем графики цен, если доступны
        if price_data:
            for timeframe, df in price_data.items():
                price_chart = create_price_chart(df, entries, exits, stop_losses, take_profits)
                if price_chart:
                    charts_html.append(f"<div class='chart-container'><h3>График {symbol} ({timeframe})</h3>{price_chart.to_html(full_html=False, include_plotlyjs=False)}</div>")
        
        # Создаем сводную статистику
        if results is not None and not results.empty:
            # Извлекаем базовую статистику
            total_trades = len(results)
            wins = results[results['result'] == 'win'].shape[0]
            losses = total_trades - wins
            winrate = (wins / total_trades) * 100 if total_trades > 0 else 0
            
            # Расчет прибыли и убытков
            gross_profit = results.loc[results['profit'] > 0, 'profit'].sum() if not results.loc[results['profit'] > 0].empty else 0
            gross_loss = abs(results.loc[results['profit'] < 0, 'profit'].sum()) if not results.loc[results['profit'] < 0].empty else 0
            net_profit = gross_profit - gross_loss
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Расчет профита на сделку
            avg_profit = results['profit'].mean()
            avg_win = results.loc[results['result'] == 'win', 'profit'].mean() if not results.loc[results['result'] == 'win'].empty else 0
            avg_loss = results.loc[results['result'] == 'loss', 'profit'].mean() if not results.loc[results['result'] == 'loss'].empty else 0
            
            # Расчет максимальной просадки
            if 'balance' in results.columns:
                results['cumulative_profit'] = results['profit'].cumsum() + results['balance'].iloc[0]
                results['running_max'] = results['cumulative_profit'].cummax()
                results['drawdown'] = (results['running_max'] - results['cumulative_profit']) / results['running_max'] * 100
                results['drawdown_abs'] = results['running_max'] - results['cumulative_profit']
                max_drawdown = results['drawdown'].max()
                max_drawdown_abs = results['drawdown_abs'].max()
            else:
                max_drawdown = "N/A"
                max_drawdown_abs = "N/A"
            
            # Расчет начального и конечного баланса
            if 'balance' in results.columns:
                initial_balance = results['balance'].iloc[0] - results['profit'].iloc[0]
                final_balance = results['balance'].iloc[-1]
                profit_percent = ((final_balance / initial_balance) - 1) * 100
            else:
                initial_balance = "N/A"
                final_balance = "N/A"
                profit_percent = "N/A"
            
            # Форматирование для бесконечного значения
            profit_factor_str = "∞" if profit_factor == float('inf') else f"{profit_factor:.2f}"
            
            # Создаем HTML-таблицу со статистикой
            stats_html = f"""
            <div class="stats-container">
                <h2>Сводная статистика</h2>
                <table>
                    <tr>
                        <th>Показатель</th>
                        <th>Значение</th>
                    </tr>
                    <tr>
                        <td>Всего сделок</td>
                        <td>{total_trades}</td>
                    </tr>
                    <tr>
                        <td>Выигрышных сделок</td>
                        <td>{wins} ({winrate:.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Проигрышных сделок</td>
                        <td>{losses} ({100-winrate:.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Начальный баланс</td>
                        <td>{initial_balance if isinstance(initial_balance, str) else f"{initial_balance:.2f}"}</td>
                    </tr>
                    <tr>
                        <td>Конечный баланс</td>
                        <td>{final_balance if isinstance(final_balance, str) else f"{final_balance:.2f}"}</td>
                    </tr>
                    <tr>
                        <td>Прибыль</td>
                        <td>{profit_percent if isinstance(profit_percent, str) else f"{profit_percent:.2f}%"}</td>
                    </tr>
                    <tr>
                        <td>Профит-фактор</td>
                        <td>{profit_factor_str}</td>
                    </tr>
                    <tr>
                        <td>Средняя прибыль на сделку</td>
                        <td>{avg_profit:.2f}</td>
                    </tr>
                    <tr>
                        <td>Средний выигрыш</td>
                        <td>{avg_win:.2f}</td>
                    </tr>
                    <tr>
                        <td>Средний проигрыш</td>
                        <td>{avg_loss:.2f}</td>
                    </tr>
                    <tr>
                        <td>Максимальная просадка</td>
                        <td>{max_drawdown if isinstance(max_drawdown, str) else f"{max_drawdown:.2f}%"}</td>
                    </tr>
                </table>
            </div>
            """
        else:
            stats_html = "<div class='stats-container'><h2>Нет данных для анализа</h2></div>"
        
        # Создаем полный HTML-документ
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Результаты бэктеста {symbol}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                h1, h2, h3 {{
                    color: #333;
                }}
                
                .chart-container {{
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 15px;
                }}
                
                .stats-container {{
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 15px;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                
                th {{
                    background-color: #f2f2f2;
                }}
                
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Результаты бэктеста {symbol}</h1>
                <p>Отчет создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                {stats_html}
                
                {''.join(charts_html)}
            </div>
        </body>
        </html>
        """
        
        # Сохраняем HTML-файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML-отчет создан: {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Ошибка при создании HTML-отчета: {str(e)}")
        logger.exception(e)
        return None

def open_html_report(file_path):
    """
    Открывает HTML-отчет в браузере
    
    Параметры:
    file_path (str): Путь к HTML-файлу
    """
    try:
        if os.path.exists(file_path):
            # Преобразуем путь в URL-формат
            file_url = f"file://{os.path.abspath(file_path)}"
            
            # Открываем файл в браузере по умолчанию
            webbrowser.open(file_url)
            logger.info(f"Отчет открыт в браузере: {file_url}")
        else:
            logger.error(f"Файл не найден: {file_path}")
    
    except Exception as e:
        logger.error(f"Ошибка при открытии отчета: {str(e)}")
        logger.exception(e)

def visualize_backtest(symbol=None, timeframes=None, with_prices=True, debug=False):
    """
    Основная функция для визуализации результатов бэктеста
    
    Параметры:
    symbol (str, optional): Символ
    timeframes (list, optional): Список таймфреймов для графиков цены
    with_prices (bool): Включать ли графики цен
    debug (bool): Включить режим отладки
    
    Возвращает:
    bool: True в случае успеха, False в случае ошибки
    """
    try:
        global SYMBOL
        SYMBOL = symbol
        
        # Устанавливаем уровень логирования
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info("=== Запуск визуализации бэктеста ===")
        
        # 1. Находим файлы с результатами бэктеста
        result_file, entries_file, exits_file, sl_file, tp_file = find_latest_backtest_files(symbol)
        
        if result_file is None:
            logger.error("Не удалось найти результаты бэктеста. Визуализация прервана.")
            return False
        
        # Получаем символ из имени файла результатов, если не был указан
        if not symbol:
            parts = os.path.basename(result_file).split('_')
            if len(parts) >= 3:
                SYMBOL = parts[2]
                logger.info(f"Определен символ: {SYMBOL}")
        
        # 2. Загружаем данные
        logger.info("Загрузка данных бэктеста...")
        results = load_data(result_file)
        entries = load_data(entries_file)
        exits = load_data(exits_file)
        stop_losses = load_data(sl_file)
        take_profits = load_data(tp_file)
        
        if results is None or results.empty:
            logger.error("Не удалось загрузить данные бэктеста. Визуализация прервана.")
            return False
        
        logger.info(f"Загружено {len(results)} записей результатов бэктеста.")
        
        # 3. Загружаем данные цен, если требуется
        price_data = {}
        if with_prices:
            if timeframes is None:
                # Используем основные таймфреймы по умолчанию
                timeframes = ["M5", "M15", "H1"]
            
            logger.info(f"Загрузка данных цен для {timeframes}...")
            
            # Определяем диапазон дат из результатов
            if 'entry_time' in results.columns and 'exit_time' in results.columns:
                start_date = results['entry_time'].min() - timedelta(days=5)
                end_date = results['exit_time'].max() + timedelta(days=5)
                
                for tf in timeframes:
                    df = load_price_data(SYMBOL, tf, start_date, end_date)
                    if df is not None and not df.empty:
                        price_data[tf] = df
                        logger.info(f"Загружено {len(df)} свечей для {tf}")
            else:
                logger.warning("Не удалось определить диапазон дат для загрузки данных цен.")
        
        # 4. Создаем HTML-отчет
        logger.info("Создание HTML-отчета...")
        html_file = create_summary_html(results, entries, exits, stop_losses, take_profits, price_data)
        
        if html_file is None:
            logger.error("Не удалось создать HTML-отчет.")
            return False
        
        # 5. Открываем отчет в браузере, если требуется
        if AUTO_OPEN:
            open_html_report(html_file)
        
        logger.info("=== Визуализация завершена ===")
        logger.info(f"Отчет сохранен в файле: {html_file}")
        print(f"\nВизуализация завершена. Отчет сохранен в файле: {html_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Критическая ошибка при визуализации результатов: {str(e)}")
        logger.exception(e)
        print(f"Произошла ошибка при визуализации: {str(e)}")
        print("Подробности смотрите в логе.")
        return False

def create_realtime_visualizer(results=None, entries=None, exits=None, stop_losses=None, take_profits=None, timeframe_data=None):
    """
    Создает интерактивный визуализатор для отображения хода бэктеста в реальном времени
    
    Параметры:
    results (DataFrame, optional): Исходные данные результатов
    entries (DataFrame, optional): Исходные данные о входах
    exits (DataFrame, optional): Исходные данные о выходах
    stop_losses (DataFrame, optional): Исходные данные о стоп-лоссах
    take_profits (DataFrame, optional): Исходные данные о тейк-профитах
    timeframe_data (dict, optional): Исходные данные цен по таймфреймам
    
    Возвращает:
    object: Объект визуализатора
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        import dash_bootstrap_components as dbc
        
        # Проверяем и преобразуем данные во временных колонках
        def ensure_datetime_columns(df, time_columns):
            if df is not None and not df.empty:
                for col in time_columns:
                    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])
            return df
        
        # Инициализируем глобальные данные с правильной проверкой на None и преобразованием типов
        global_data = {
            'results': ensure_datetime_columns(results, ['entry_time', 'exit_time']) if results is not None and not results.empty else pd.DataFrame(),
            'entries': ensure_datetime_columns(entries, ['time']) if entries is not None and not entries.empty else pd.DataFrame(),
            'exits': ensure_datetime_columns(exits, ['time']) if exits is not None and not exits.empty else pd.DataFrame(),
            'stop_losses': ensure_datetime_columns(stop_losses, ['time']) if stop_losses is not None and not stop_losses.empty else pd.DataFrame(),
            'take_profits': ensure_datetime_columns(take_profits, ['time']) if take_profits is not None and not take_profits.empty else pd.DataFrame(),
            'timeframe_data': {},
            'current_time': None,
            'current_timeframe': None,
            'balance': 10000  # Начальный баланс по умолчанию
        }
        
        # Преобразуем временные колонки в данных цен
        if timeframe_data is not None:
            for tf, df in timeframe_data.items():
                if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df = df.copy()
                    df['time'] = pd.to_datetime(df['time'])
                global_data['timeframe_data'][tf] = df
        
        # Определяем текущий таймфрейм, если данные доступны
        if global_data['timeframe_data']:
            global_data['current_timeframe'] = list(global_data['timeframe_data'].keys())[0]
        
        # Создаем приложение Dash
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.title = f"Визуализатор бэктеста {SYMBOL}"
        
        # Определяем layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(f"Визуализатор бэктеста {SYMBOL}", className="text-center mb-4"),
                    html.Div(id="current-time-display", className="text-center mb-3"),
                    html.Div(id="balance-display", className="text-center h4 mb-4"),
                    
                    # Контролы для управления воспроизведением
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Button("Play", id="play-button", n_clicks=0, className="btn btn-success me-2"),
                                    html.Button("Pause", id="pause-button", n_clicks=0, className="btn btn-warning me-2"),
                                    html.Button("Step", id="step-button", n_clicks=0, className="btn btn-info me-2"),
                                ], width=12, className="d-flex justify-content-center mb-3"),
                                
                                dbc.Col([
                                    html.Label("Speed:"),
                                    dcc.Slider(
                                        id="speed-slider",
                                        min=1,
                                        max=50,
                                        step=1,
                                        value=10,
                                        marks={1: '1x', 10: '10x', 20: '20x', 30: '30x', 40: '40x', 50: '50x'},
                                    ),
                                ], width=12),
                                
                                dbc.Col([
                                    html.Div([
                                        html.Label("Timeframe:"),
                                        dcc.Dropdown(
                                            id="timeframe-dropdown",
                                            options=[
                                                {"label": tf, "value": tf} 
                                                for tf in global_data['timeframe_data'].keys()
                                            ] if global_data['timeframe_data'] else [],
                                            value=global_data['current_timeframe'],
                                            clearable=False,
                                            className="mb-3",
                                        ),
                                    ]),
                                ], width=12),
                            ]),
                        ]),
                    ], className="mb-4"),
                    

                    # Переключатель режима отображения
                    dbc.Card([
                        dbc.CardHeader("Режим отображения"),
                        dbc.CardBody([
                            dbc.RadioItems(
                                id="display-mode",
                                options=[
                                    {"label": "Пошаговый просмотр", "value": "step-by-step"},
                                    {"label": "Показать все сделки", "value": "show-all"},
                                ],
                                value="step-by-step",
                                inline=True,
                                className="mb-2"
                            ),
                            html.Button(
                                "Обновить график", 
                                id="refresh-chart-button", 
                                className="btn btn-primary"
                            )
                        ]),
                    ], className="mb-4"),
                    
                    # График цены
                    dbc.Card([
                        dbc.CardHeader("График цены"),
                        dbc.CardBody([
                            dcc.Graph(id="price-chart", style={"height": "60vh"}),
                        ]),
                    ], className="mb-4"),
                    
                    # График баланса
                    dbc.Card([
                        dbc.CardHeader("График баланса"),
                        dbc.CardBody([
                            dcc.Graph(id="balance-chart", style={"height": "40vh"}),
                        ]),
                    ], className="mb-4"),
                    
                    # Таблица текущих сделок
                    dbc.Card([
                        dbc.CardHeader("Текущие сделки"),
                        dbc.CardBody([
                            html.Div(id="active-trades-table"),
                        ]),
                    ], className="mb-4"),
                    
                    # Скрытые хранилища для данных
                    dcc.Store(id="global-data-store"),
                    dcc.Store(id="playback-state", data={"is_playing": False, "speed": 10, "current_index": 0}),
                    dcc.Interval(id="playback-interval", interval=1000, n_intervals=0, disabled=True),
                ], width=12),
            ]),
        ], fluid=True)
        
        # Колбэк для обновления данных
        @app.callback(
            [Output("price-chart", "figure"),
            Output("balance-chart", "figure"),
            Output("active-trades-table", "children"),
            Output("current-time-display", "children"),
            Output("balance-display", "children")],
            [Input("playback-interval", "n_intervals"),
            Input("step-button", "n_clicks"),
            Input("timeframe-dropdown", "value"),
            Input("display-mode", "value"),     # Добавляем новый Input для режима отображения
            Input("refresh-chart-button", "n_clicks")],  # Добавляем кнопку обновления
            [State("playback-state", "data"),
            State("global-data-store", "data")],
        )
        def update_charts(n_intervals, step_clicks, selected_timeframe, display_mode, refresh_clicks, playback_state, stored_data):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
            
            # Обновляем текущий таймфрейм, если он изменился
            if selected_timeframe and global_data['current_timeframe'] != selected_timeframe:
                global_data['current_timeframe'] = selected_timeframe
            
            # Определяем временную шкалу на основе данных таймфрейма
            timeline = []
            if global_data['timeframe_data'] and global_data['current_timeframe'] in global_data['timeframe_data']:
                timeline = global_data['timeframe_data'][global_data['current_timeframe']]['time'].tolist()
            
            # Если временная шкала пуста, используем шкалу из результатов
            if not timeline and not global_data['results'].empty and 'entry_time' in global_data['results'].columns:
                # Убедимся, что данные в правильном формате
                if not pd.api.types.is_datetime64_any_dtype(global_data['results']['entry_time']):
                    global_data['results']['entry_time'] = pd.to_datetime(global_data['results']['entry_time'])
                if not pd.api.types.is_datetime64_any_dtype(global_data['results']['exit_time']):
                    global_data['results']['exit_time'] = pd.to_datetime(global_data['results']['exit_time'])
                    
                timeline = sorted(global_data['results']['entry_time'].tolist() + global_data['results']['exit_time'].tolist())
            
            # Если все еще нет временной шкалы, возвращаем пустые графики
            if not timeline:
                return go.Figure(), go.Figure(), "Нет данных", "Текущее время: -", "Баланс: -"
            
            # Логика зависит от выбранного режима отображения
            if display_mode == "show-all":
                # В режиме "Показать все сделки" устанавливаем текущее время на конец периода
                current_time = timeline[-1]
                current_index = len(timeline) - 1
                # Обновляем баланс до последнего значения
                if not global_data['results'].empty and 'balance' in global_data['results'].columns:
                    global_data['balance'] = global_data['results']['balance'].iloc[-1]
            else:
                # В режиме пошагового просмотра используем обычную логику
                current_index = playback_state.get('current_index', 0) if playback_state else 0
                
                # Если был нажат кнопка Step, увеличиваем индекс
                if trigger == "step-button.n_clicks" and step_clicks > 0:
                    current_index += 1
                
                # Проверяем границы
                if current_index >= len(timeline):
                    current_index = len(timeline) - 1
                
                current_time = timeline[current_index]
            
            global_data['current_time'] = current_time
            
            # Обновляем баланс на основе завершенных сделок
            if not global_data['results'].empty and 'exit_time' in global_data['results'].columns:
                # Убедимся, что данные в правильном формате для сравнения
                if not pd.api.types.is_datetime64_any_dtype(global_data['results']['exit_time']):
                    global_data['results']['exit_time'] = pd.to_datetime(global_data['results']['exit_time'])
                    
                completed_trades = global_data['results'][global_data['results']['exit_time'] <= current_time]
                
                if not completed_trades.empty and 'balance' in completed_trades.columns:
                    global_data['balance'] = completed_trades['balance'].iloc[-1]
            
            # Создаем график цены
            if display_mode == "show-all":
                # В режиме "Показать все сделки" не передаем current_time, чтобы показать весь диапазон
                price_chart = create_price_chart(
                    global_data['timeframe_data'].get(global_data['current_timeframe']),
                    global_data['entries'],
                    global_data['exits'],
                    global_data['stop_losses'],
                    global_data['take_profits'],
                    None,  # Не указываем current_time
                    autoscale=True
                )
            else:
                # В обычном режиме пошагового просмотра
                price_chart = create_price_chart(
                    global_data['timeframe_data'].get(global_data['current_timeframe']),
                    global_data['entries'],
                    global_data['exits'],
                    global_data['stop_losses'],
                    global_data['take_profits'],
                    current_time,
                    autoscale=True
                )
            
            # Если нет графика цены, создаем пустой график
            if price_chart is None:
                price_chart = go.Figure()
                price_chart.update_layout(title="Нет данных для отображения")
            
            # Создаем график баланса
            balance_chart = None
            if not global_data['results'].empty:
                # Преобразуем типы данных, если нужно
                if not pd.api.types.is_datetime64_any_dtype(global_data['results']['exit_time']):
                    global_data['results']['exit_time'] = pd.to_datetime(global_data['results']['exit_time'])
                    
                completed_trades = global_data['results'][global_data['results']['exit_time'] <= current_time]
                if not completed_trades.empty:
                    balance_chart = create_balance_chart(completed_trades)
            
            # Если нет графика баланса, создаем пустой график
            if balance_chart is None:
                balance_chart = go.Figure()
                balance_chart.update_layout(title="Нет данных о балансе")
            
            # Создаем таблицу текущих сделок
            active_trades_html = "Нет активных сделок"
            
            # Находим текущие активные сделки
            active_trades = []
            if not global_data['entries'].empty:
                # Убедимся, что данные в правильном формате
                if not pd.api.types.is_datetime64_any_dtype(global_data['entries']['time']):
                    global_data['entries']['time'] = pd.to_datetime(global_data['entries']['time'])
                
                entries_until_now = global_data['entries'][global_data['entries']['time'] <= current_time]
                
                if not global_data['exits'].empty:
                    # Убедимся, что данные в правильном формате
                    if not pd.api.types.is_datetime64_any_dtype(global_data['exits']['time']):
                        global_data['exits']['time'] = pd.to_datetime(global_data['exits']['time'])
                        
                    exits_until_now = global_data['exits'][global_data['exits']['time'] <= current_time]
                    
                    # Группируем входы по времени и типу ордера
                    entry_keys = set()
                    for _, entry in entries_until_now.iterrows():
                        key = (entry['time'], entry['order'])
                        entry_keys.add(key)
                    
                    # Группируем выходы по времени и типу ордера
                    exit_keys = set()
                    for _, exit_data in exits_until_now.iterrows():
                        # Находим соответствующий вход для выхода
                        for entry_time, entry_order in entry_keys:
                            if exit_data['order'] == entry_order and exit_data['time'] > entry_time:
                                exit_keys.add((entry_time, entry_order))
                                break
                    
                    # Активные сделки - это те, которые есть во входах, но нет в выходах
                    active_entries = entry_keys - exit_keys
                    
                    # Создаем список активных сделок с информацией о SL/TP
                    for entry_time, entry_order in active_entries:
                        entry_data = entries_until_now[(entries_until_now['time'] == entry_time) & (entries_until_now['order'] == entry_order)].iloc[0]
                        
                        # Находим SL и TP для этой сделки
                        sl_data = None
                        tp_data = None
                        
                        if not global_data['stop_losses'].empty:
                            # Убедимся, что данные в правильном формате
                            if not pd.api.types.is_datetime64_any_dtype(global_data['stop_losses']['time']):
                                global_data['stop_losses']['time'] = pd.to_datetime(global_data['stop_losses']['time'])
                                
                            sl_matches = global_data['stop_losses'][(global_data['stop_losses']['time'] == entry_time) & (global_data['stop_losses']['order'] == entry_order)]
                            if not sl_matches.empty:
                                sl_data = sl_matches.iloc[0]
                        
                        if not global_data['take_profits'].empty:
                            # Убедимся, что данные в правильном формате
                            if not pd.api.types.is_datetime64_any_dtype(global_data['take_profits']['time']):
                                global_data['take_profits']['time'] = pd.to_datetime(global_data['take_profits']['time'])
                                
                            tp_matches = global_data['take_profits'][(global_data['take_profits']['time'] == entry_time) & (global_data['take_profits']['order'] == entry_order)]
                            if not tp_matches.empty:
                                tp_data = tp_matches.iloc[0]
                        
                        # Добавляем сделку в список активных
                        active_trades.append({
                            'entry_time': entry_time,
                            'order': entry_order,
                            'price': entry_data['price'],
                            'sl': sl_data['price'] if sl_data is not None else None,
                            'tp': tp_data['price'] if tp_data is not None else None,
                            'setup': entry_data.get('setup', 'Standard')
                        })
            
            # Создаем HTML-таблицу активных сделок
            if active_trades:
                active_trades_html = html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Время входа"),
                            html.Th("Тип"),
                            html.Th("Цена"),
                            html.Th("SL"),
                            html.Th("TP"),
                            html.Th("Сетап")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(trade['entry_time'].strftime('%Y-%m-%d %H:%M') if isinstance(trade['entry_time'], (datetime, pd.Timestamp)) else str(trade['entry_time'])),
                            html.Td(trade['order'].upper()),
                            html.Td(f"{trade['price']:.5f}"),
                            html.Td(f"{trade['sl']:.5f}" if trade['sl'] is not None else "-"),
                            html.Td(f"{trade['tp']:.5f}" if trade['tp'] is not None else "-"),
                            html.Td(trade['setup'])
                        ]) for trade in active_trades
                    ])
                ], className="table table-striped table-bordered")
            
            # Проверка типа current_time перед использованием strftime
            current_time_str = ""
            if isinstance(current_time, (datetime, pd.Timestamp)):
                current_time_str = current_time.strftime('%Y-%m-%d %H:%M')
            else:
                # Если current_time не datetime, пробуем преобразовать или просто отображаем как строку
                try:
                    if isinstance(current_time, str):
                        current_time_str = current_time
                    else:
                        current_time_str = str(current_time)
                except:
                    current_time_str = "Unknown time"
            
            return (
                price_chart, 
                balance_chart, 
                active_trades_html, 
                f"Текущее время: {current_time_str}", 
                f"Баланс: {global_data['balance']:.2f}"
            )
        
        # Колбэк для управления воспроизведением
        @app.callback(
            [Output("playback-interval", "disabled"),
            Output("playback-interval", "interval"),
            Output("playback-state", "data")],
            [Input("play-button", "n_clicks"),
            Input("pause-button", "n_clicks"),
            Input("speed-slider", "value")],
            [State("playback-state", "data")]
        )
        def control_playback(play_clicks, pause_clicks, speed, playback_state):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
            
            if playback_state is None:
                playback_state = {"is_playing": False, "speed": 10, "current_index": 0}
            
            # Управление воспроизведением
            if trigger == "play-button.n_clicks" and play_clicks > 0:
                playback_state["is_playing"] = True
            elif trigger == "pause-button.n_clicks" and pause_clicks > 0:
                playback_state["is_playing"] = False
            
            # Установка скорости
            if trigger == "speed-slider.value":
                playback_state["speed"] = speed
            
            # Определяем интервал обновления
            if speed > 30:
                interval = 100  # Очень быстро: обновляем реже, но с большим шагом
            elif speed > 20:
                interval = 200
            elif speed > 10:
                interval = 300
            else:
                interval = 500  # Медленно: частые обновления с маленьким шагом
            
            return (
                not playback_state["is_playing"],  # Интервал включен, когда is_playing=True
                interval,
                playback_state
            )
        
        # Колбэк для обновления индекса при воспроизведении
        @app.callback(
            Output("playback-state", "data", allow_duplicate=True),
            [Input("playback-interval", "n_intervals")],
            [State("playback-state", "data")],
            prevent_initial_call=True
        )
        def update_index(n_intervals, playback_state):
            if not playback_state["is_playing"]:
                return playback_state
            
            # Определяем временную шкалу
            timeline = []
            if global_data['timeframe_data'] and global_data['current_timeframe'] in global_data['timeframe_data']:
                timeline = global_data['timeframe_data'][global_data['current_timeframe']]['time'].tolist()
            
            # Если временная шкала пуста, используем шкалу из результатов
            if not timeline and not global_data['results'].empty and 'entry_time' in global_data['results'].columns:
                # Убедимся, что данные в правильном формате
                if not pd.api.types.is_datetime64_any_dtype(global_data['results']['entry_time']):
                    global_data['results']['entry_time'] = pd.to_datetime(global_data['results']['entry_time'])
                if not pd.api.types.is_datetime64_any_dtype(global_data['results']['exit_time']):
                    global_data['results']['exit_time'] = pd.to_datetime(global_data['results']['exit_time'])
                    
                timeline = sorted(global_data['results']['entry_time'].tolist() + global_data['results']['exit_time'].tolist())
            
            # Если все еще нет временной шкалы, возвращаем текущее состояние
            if not timeline:
                return playback_state
            
            # Определяем шаг в зависимости от скорости
            step = 1
            if playback_state["speed"] > 30:
                step = 5  # Очень быстро
            elif playback_state["speed"] > 20:
                step = 3  # Быстро
            elif playback_state["speed"] > 10:
                step = 2  # Средне
            
            # Обновляем индекс с учетом границ
            current_index = playback_state["current_index"] + step
            if current_index >= len(timeline):
                current_index = len(timeline) - 1
                playback_state["is_playing"] = False  # Останавливаем, когда достигнут конец
            
            playback_state["current_index"] = current_index
            return playback_state
        
        # Запускаем сервер Dash в отдельном потоке
        def run_server():
            try:
                app.run(debug=False, port=8050)
            except Exception as e:
                logger.error(f"Ошибка при запуске Dash сервера: {str(e)}")
                print(f"Ошибка при запуске визуализации: {str(e)}")
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Открываем браузер
        webbrowser.open('http://localhost:8050/')
        
        # Возвращаем объект приложения
        return app
    
    except ImportError as e:
        logger.error(f"Не удалось импортировать необходимые библиотеки для интерактивного визуализатора: {e}")
        logger.info("Устанавливаем необходимые библиотеки: pip install dash dash-bootstrap-components")
        print("Не удалось создать интерактивный визуализатор. Установите необходимые библиотеки:")
        print("pip install dash dash-bootstrap-components")
        return None
    except Exception as e:
        logger.error(f"Ошибка при создании интерактивного визуализатора: {str(e)}")
        logger.exception(e)
        return None

def update_realtime_data(visualizer, data):
    """
    Обновляет данные в реальном времени в визуализаторе
    
    Параметры:
    visualizer (object): Объект визуализатора
    data (dict): Новые данные
    """
    try:
        # Добавляем новые данные в глобальный словарь
        if visualizer is not None:
            # Обновляем текущее время
            if 'current_time' in data:
                visualizer.global_data['current_time'] = data['current_time']
            
            # Обновляем баланс
            if 'balance' in data:
                visualizer.global_data['balance'] = data['balance']
            
            # Обновляем сделки
            if 'action' in data and data['action'] == 'update_trades':
                # Обновляем вход
                if 'entry' in data:
                    entry = data['entry']
                    visualizer.global_data['entries'] = pd.concat([
                        visualizer.global_data['entries'],
                        pd.DataFrame([entry])
                    ], ignore_index=True)
                
                # Обновляем выход
                if 'exit' in data:
                    exit_data = data['exit']
                    visualizer.global_data['exits'] = pd.concat([
                        visualizer.global_data['exits'],
                        pd.DataFrame([exit_data])
                    ], ignore_index=True)
                
                # Обновляем стоп-лосс
                if 'sl' in data:
                    sl_data = data['sl']
                    visualizer.global_data['stop_losses'] = pd.concat([
                        visualizer.global_data['stop_losses'],
                        pd.DataFrame([sl_data])
                    ], ignore_index=True)
                
                # Обновляем тейк-профит
                if 'tp' in data:
                    tp_data = data['tp']
                    visualizer.global_data['take_profits'] = pd.concat([
                        visualizer.global_data['take_profits'],
                        pd.DataFrame([tp_data])
                    ], ignore_index=True)
            
            # Обновляем статистику
            if 'action' in data and data['action'] == 'update_statistics':
                if 'balance' in data:
                    visualizer.global_data['balance'] = data['balance']
                
                # Добавляем результат сделки
                if 'trade_result' in data and 'profit' in data:
                    new_result = {
                        'result': data['trade_result'],
                        'profit': data['profit'],
                        'balance': data['balance'],
                        'exit_time': visualizer.global_data['current_time']
                    }
                    
                    # Добавляем результат в историю
                    visualizer.global_data['results'] = pd.concat([
                        visualizer.global_data['results'],
                        pd.DataFrame([new_result])
                    ], ignore_index=True)
    
    except Exception as e:
        logger.error(f"Ошибка при обновлении данных в реальном времени: {str(e)}")
        logger.exception(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация результатов бэктеста")
    parser.add_argument("--symbol", type=str, help="Символ")
    parser.add_argument("--timeframes", type=str, default="M5,M15,H1", help="Список таймфреймов через запятую")
    parser.add_argument("--no-prices", action="store_true", help="Не включать графики цен")
    parser.add_argument("--theme", type=str, default="white", choices=["white", "dark"], help="Тема оформления")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер")
    parser.add_argument("--debug", action="store_true", help="Включить режим отладки")
    parser.add_argument("--realtime", action="store_true", help="Интерактивная визуализация в реальном времени")
    
    args = parser.parse_args()
    
    # Применяем параметры
    SYMBOL = args.symbol
    THEME = args.theme
    AUTO_OPEN = not args.no_browser
    
    if args.realtime:
        # Запускаем интерактивный визуализатор
        try:
            # Загружаем самые последние данные бэктеста
            result_file, entries_file, exits_file, sl_file, tp_file = find_latest_backtest_files(args.symbol)
            
            if result_file is not None:
                # Загружаем данные
                results = load_data(result_file)
                entries = load_data(entries_file)
                exits = load_data(exits_file)
                stop_losses = load_data(sl_file)
                take_profits = load_data(tp_file)
                
                # Загружаем данные цен, если нужно
                timeframe_data = {}
                if not args.no_prices:
                    timeframes = args.timeframes.split(",")
                    
                    # Определяем диапазон дат из результатов
                    start_date = None
                    end_date = None
                    
                    if results is not None and not results.empty and 'entry_time' in results.columns and 'exit_time' in results.columns:
                        start_date = results['entry_time'].min() - timedelta(days=5)
                        end_date = results['exit_time'].max() + timedelta(days=5)
                    
                    if start_date is not None and end_date is not None:
                        for tf in timeframes:
                            df = load_price_data(SYMBOL or "EURUSD", tf, start_date, end_date)
                            if df is not None and not df.empty:
                                timeframe_data[tf] = df
                
                # Создаем интерактивный визуализатор
                visualizer = create_realtime_visualizer(
                    results=results,
                    entries=entries,
                    exits=exits,
                    stop_losses=stop_losses,
                    take_profits=take_profits,
                    timeframe_data=timeframe_data
                )
                
                if visualizer is not None:
                    print("Интерактивный визуализатор запущен. Нажмите Ctrl+C для завершения.")
                    try:
                        # Оставляем основной поток работающим, чтобы сервер Dash не завершался
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("Визуализатор остановлен.")
                else:
                    print("Не удалось запустить интерактивный визуализатор.")
            else:
                print("Не найдены файлы с результатами бэктеста.")
            
        except Exception as e:
            print(f"Ошибка при запуске интерактивного визуализатора: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    else:
        # Запускаем обычную визуализацию
        timeframes = args.timeframes.split(",") if args.timeframes else None
        visualize_backtest(
            symbol=args.symbol,
            timeframes=timeframes,
            with_prices=not args.no_prices,
            debug=args.debug
        )