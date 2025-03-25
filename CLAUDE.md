# AlgoTrade Bot Guidelines

## Commands
- **Run main script**: `python main.py`
- **Run backtest**: `python main.py --mode backtest`
- **Run live trading**: `python main.py --mode live`
- **Run optimized backtest**: `python backtest_optimized.py`
- **Visualize results**: `python visualization.py`

## Code Style
- **Imports**: Standard library, then third-party, then local imports
- **Formatting**: Use 4 spaces for indentation
- **Types**: Use type hints where possible
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error handling**: Use try/except blocks with specific exceptions
- **Documentation**: Docstrings should follow the format seen in strategy.py functions
- **Caching**: Use @lru_cache for expensive computational functions
- **Configuration**: All constants should be defined in config.py
- **Logging**: Use logging module instead of print statements

## Performance
- Prefer pandas/numpy vectorized operations over loops when processing data
- Use caching mechanisms for computational-heavy functions