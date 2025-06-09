import backtrader as bt
import backtrader.analyzers as bta
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np


class ImprovedMaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 8),          # Optimized from 10
        ('slow_period', 21),         # Optimized from 20
        ('signal_period', 9),        # MACD signal line
        ('rsi_period', 14),
        ('rsi_oversold', 30),        # RSI oversold threshold
        ('rsi_overbought', 70),      # RSI overbought threshold
        ('atr_period', 14),          # ATR period for dynamic stops
        ('stop_atr_mult', 2.0),      # ATR multiplier for stop loss
        ('trail_atr_mult', 1.5),     # ATR multiplier for trailing stop
        ('min_volume_mult', 1.2),    # Minimum volume multiplier
        ('lookback_volume', 20),     # Volume lookback period
    )

    def __init__(self):
        # Moving averages
        self.ma_fast = bt.ind.EMA(period=self.params.fast_period)  # EMA instead of SMA
        self.ma_slow = bt.ind.EMA(period=self.params.slow_period)
        
        # MACD for trend confirmation
        self.macd = bt.ind.MACD(period_me1=12, period_me2=26, period_signal=self.params.signal_period)
        
        # RSI for momentum confirmation
        self.rsi = bt.ind.RSI(period=self.params.rsi_period)
        
        # ATR for dynamic position sizing and stops
        self.atr = bt.ind.ATR(period=self.params.atr_period)
        
        # Volume indicators
        self.volume_sma = bt.ind.SMA(self.data.volume, period=self.params.lookback_volume)
        
        # Bollinger Bands for mean reversion detection
        self.bb = bt.ind.BollingerBands(period=20, devfactor=2.0)
        
        # Crossover signals
        self.ma_crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)
        self.macd_crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        
        # Trade tracking
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.trail_stop = None
        self.trade_count = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                # Set initial stop loss using ATR
                self.stop_price = self.entry_price - (self.atr[0] * self.params.stop_atr_mult)
                self.trail_stop = self.stop_price
                self.trade_count += 1
                print(f'BUY #{self.trade_count}: Price: {order.executed.price:.2f}, Stop: {self.stop_price:.2f}')
            else:
                profit = order.executed.price - self.entry_price if self.entry_price else 0
                print(f'SELL: Price: {order.executed.price:.2f}, P&L: {profit:.2f}')
                self.entry_price = None
                self.stop_price = None
                self.trail_stop = None
        
        self.order = None

    def get_position_size(self):
        """Dynamic position sizing based on volatility (ATR)"""
        if self.atr[0] == 0:
            return 0.02  # Default 2% risk
        
        # Risk 1% of portfolio per trade
        portfolio_value = self.broker.getvalue()
        risk_amount = portfolio_value * 0.01
        
        # Calculate position size based on ATR stop
        stop_distance = self.atr[0] * self.params.stop_atr_mult
        if stop_distance > 0:
            shares = risk_amount / stop_distance
            position_value = shares * self.data.close[0]
            position_percent = position_value / portfolio_value
            return min(position_percent, 0.1)  # Cap at 10% of portfolio
        
        return 0.02

    def is_trending_market(self):
        """Check if market is in a trending state"""
        # Use MACD histogram slope
        if len(self.macd.histo) < 3:
            return False
        
        # Check if MACD histogram is consistently above/below zero
        histogram_trend = (self.macd.histo[0] > 0 and self.macd.histo[-1] > 0 and 
                          self.macd.histo[-2] > 0)
        
        # Check price distance from moving averages
        price_above_fast = self.data.close[0] > self.ma_fast[0]
        fast_above_slow = self.ma_fast[0] > self.ma_slow[0]
        
        return histogram_trend and price_above_fast and fast_above_slow

    def has_volume_confirmation(self):
        """Check if current volume supports the move"""
        if len(self.data.volume) == 0 or self.volume_sma[0] == 0:
            return True  # Default to True if no volume data
        
        return self.data.volume[0] > (self.volume_sma[0] * self.params.min_volume_mult)

    def is_not_overbought(self):
        """Check multiple overbought conditions"""
        # RSI check
        rsi_ok = self.rsi[0] < self.params.rsi_overbought
        
        # Bollinger Bands check - not touching upper band
        bb_ok = self.data.close[0] < self.bb.top[0]
        
        return rsi_ok and bb_ok

    def should_exit_position(self):
        """Comprehensive exit logic"""
        current_price = self.data.close[0]
        
        # Update trailing stop
        if self.trail_stop and current_price > self.entry_price:
            new_trail = current_price - (self.atr[0] * self.params.trail_atr_mult)
            self.trail_stop = max(self.trail_stop, new_trail)
        
        # Exit conditions
        # 1. Trailing stop hit
        if self.trail_stop and current_price <= self.trail_stop:
            print(f'TRAILING STOP at {current_price:.2f}')
            return True
        
        # 2. MA crossover (trend reversal)
        if self.ma_crossover < 0:
            print(f'MA CROSSOVER EXIT at {current_price:.2f}')
            return True
        
        # 3. MACD bearish divergence
        if self.macd_crossover < 0 and self.macd.macd[0] > 0:
            print(f'MACD BEARISH at {current_price:.2f}')
            return True
        
        # 4. RSI extremely overbought
        if self.rsi[0] > 80:
            print(f'RSI OVERBOUGHT EXIT at {current_price:.2f}')
            return True
        
        return False

    def next(self):
        # Cancel pending orders
        if self.order:
            return

        if not self.position:
            # Entry conditions - all must be true
            conditions = [
                self.ma_crossover > 0,                    # MA bullish crossover
                self.macd_crossover > 0,                  # MACD bullish crossover
                self.rsi[0] > self.params.rsi_oversold,   # RSI not oversold
                self.is_not_overbought(),                 # Not overbought
                self.has_volume_confirmation(),           # Volume confirmation
                self.is_trending_market(),                # Trending market
            ]
            
            if all(conditions):
                # Dynamic position sizing
                size_percent = self.get_position_size()
                self.order = self.buy(size=None)  # Use default sizer
                
        else:
            # Exit logic
            if self.should_exit_position():
                self.order = self.close()


def download_data(symbol='BTC-USD', period='2y'):
    """Download data from Yahoo Finance with improved error handling"""
    try:
        # Download data
        data = yf.download(symbol, period=period, auto_adjust=False, prepost=True, threads=True)
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Standardize column names
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Data quality checks
        data = data.dropna()
        
        # Remove outliers (optional)
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                Q1 = data[col].quantile(0.01)
                Q3 = data[col].quantile(0.99)
                data = data[(data[col] >= Q1) & (data[col] <= Q3)]
        
        print(f"Downloaded {len(data)} rows of {symbol} data")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
        
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None


def optimize_parameters(symbol='BTC-USD', period='2y'):
    """Simple parameter optimization"""
    best_sharpe = -999
    best_params = None
    
    # Parameter ranges to test
    fast_periods = [5, 8, 10, 12]
    slow_periods = [18, 21, 26, 30]
    
    print("Optimizing parameters...")
    
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
                
            cerebro = bt.Cerebro()
            
            # Download data
            df = download_data(symbol, period)
            if df is None:
                continue
            
            # Add data
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,
                open='open', high='high', low='low', close='close',
                volume='volume', openinterest=None
            )
            cerebro.adddata(data)
            
            # Add strategy with test parameters
            cerebro.addstrategy(ImprovedMaCrossStrategy, 
                              fast_period=fast, slow_period=slow)
            
            # Set up cerebro
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addsizer(bt.sizers.PercentSizer, percents=5)
            cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.02)
            
            # Run
            try:
                results = cerebro.run()
                sharpe = results[0].analyzers.sharpe.get_analysis().get('sharperatio', -999)
                
                if sharpe and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (fast, slow)
                    
                print(f"Fast: {fast}, Slow: {slow}, Sharpe: {sharpe:.3f}")
                
            except:
                continue
    
    print(f"\nBest parameters: Fast={best_params[0]}, Slow={best_params[1]}, Sharpe={best_sharpe:.3f}")
    return best_params


def run_backtest(symbol='BTC-USD', period='2y', optimize=False):
    """Enhanced backtest with optimization option"""
    
    if optimize:
        best_params = optimize_parameters(symbol, period)
        if best_params:
            fast_period, slow_period = best_params
        else:
            fast_period, slow_period = 8, 21
    else:
        fast_period, slow_period = 8, 21
    
    cerebro = bt.Cerebro()
    
    # Download data
    print(f"Downloading {symbol} data...")
    df = download_data(symbol, period)
    
    if df is None or df.empty:
        print("Failed to download data")
        return None, None
    
    # Add data to cerebro
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open', high='high', low='low', close='close',
        volume='volume', openinterest=None
    )
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(ImprovedMaCrossStrategy, 
                       fast_period=fast_period, 
                       slow_period=slow_period)
    
    # Broker settings
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # Position sizing - reduced to 5% per trade for better risk management
    cerebro.addsizer(bt.sizers.PercentSizer, percents=5)
    
    # Add analyzers
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bta.Returns, _name="returns")
    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bta.SQN, _name="sqn")  # System Quality Number
    cerebro.addanalyzer(bta.Calmar, _name="calmar")  # Calmar ratio
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    print(f"Using MA periods: Fast={fast_period}, Slow={slow_period}")
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value / 10000) - 1) * 100
    
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Print detailed analysis
    print_analysis(strategy)
    
    return cerebro, strategy


def print_analysis(strategy):
    """Print comprehensive analysis results"""
    print("\n" + "="*50)
    print("         PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Get analyzer results
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    returns_analysis = strategy.analyzers.returns.get_analysis()
    trades_analysis = strategy.analyzers.trades.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    sqn_analysis = strategy.analyzers.sqn.get_analysis()
    calmar_analysis = strategy.analyzers.calmar.get_analysis()
    
    # Risk-adjusted returns
    sharpe_ratio = sharpe_analysis.get('sharperatio', 'N/A')
    sqn = sqn_analysis.get('sqn', 'N/A')
    calmar = calmar_analysis.get('calmar', 'N/A')
    
    print(f"Sharpe Ratio:        {sharpe_ratio}")
    print(f"System Quality No.:  {sqn}")
    print(f"Calmar Ratio:        {calmar}")
    
    # Returns
    total_return = returns_analysis.get('rtot', 0) * 100
    average_return = returns_analysis.get('ravg', 0) * 100
    print(f"Total Return:        {total_return:.2f}%")
    print(f"Average Return:      {average_return:.4f}%")
    
    # Trade statistics
    total_trades = trades_analysis.get('total', {}).get('total', 0)
    won_trades = trades_analysis.get('won', {}).get('total', 0)
    lost_trades = trades_analysis.get('lost', {}).get('total', 0)
    
    if total_trades > 0:
        win_rate = (won_trades / total_trades) * 100
        print(f"Total Trades:        {total_trades}")
        print(f"Win Rate:            {win_rate:.2f}%")
        
        if won_trades > 0:
            avg_win = trades_analysis.get('won', {}).get('pnl', {}).get('average', 0)
            max_win = trades_analysis.get('won', {}).get('pnl', {}).get('max', 0)
            print(f"Average Win:         ${avg_win:.2f}")
            print(f"Max Win:             ${max_win:.2f}")
        
        if lost_trades > 0:
            avg_loss = trades_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
            max_loss = trades_analysis.get('lost', {}).get('pnl', {}).get('max', 0)
            print(f"Average Loss:        ${avg_loss:.2f}")
            print(f"Max Loss:            ${max_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * won_trades / (avg_loss * lost_trades))
                print(f"Profit Factor:       {profit_factor:.2f}")
    
    # Drawdown
    max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0) * 100
    max_dd_len = drawdown_analysis.get('max', {}).get('len', 0)
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    print(f"Max DD Length:       {max_dd_len} periods")


def setup_plot_style():
    """Enhanced plotting style"""
    plt.style.use('dark_background')
    
    plt.rcParams["figure.figsize"] = (16, 12)
    plt.rcParams['lines.linewidth'] = 1.2
    
    SIZE = 9
    plt.rcParams['axes.labelsize'] = SIZE
    plt.rcParams['ytick.labelsize'] = SIZE
    plt.rcParams['xtick.labelsize'] = SIZE
    plt.rcParams["font.size"] = SIZE
    
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['grid.alpha'] = 0.3
    
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams['legend.fontsize'] = SIZE
    plt.rcParams['legend.title_fontsize'] = SIZE + 1


if __name__ == "__main__":
    # Run the improved backtest
    print("Running Improved MA Cross Strategy Backtest")
    print("-" * 50)
    
    # Option to optimize parameters
    optimize_params = input("Optimize parameters? (y/n): ").lower() == 'y'
    
    cerebro, strategy = run_backtest('BTC-USD', '2y', optimize=optimize_params)
    
    if cerebro and strategy:
        # Setup and show plot
        setup_plot_style()
        try:
            cerebro.plot(style='candle', barup='#00ff88', bardown='#ff4444', 
                        volume=False, figsize=(16, 12))
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
            print("Backtest completed successfully, but plotting failed.")
    else:
        print("Backtest failed to run.")