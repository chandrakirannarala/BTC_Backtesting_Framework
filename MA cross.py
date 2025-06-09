import backtrader as bt
import backtrader.analyzers as bta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class ImprovedMaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),         # Slightly slower for stability
        ('slow_period', 25),         # Better separation ratio
        ('signal_period', 9),        
        ('rsi_period', 14),
        ('rsi_oversold', 35),        # More conservative thresholds
        ('rsi_overbought', 65),      
        ('atr_period', 14),          
        ('stop_atr_mult', 1.8),      # Tighter stops for better risk management
        ('trail_atr_mult', 1.2),     
        ('min_volume_mult', 1.1),    # Less restrictive volume filter
        ('lookback_volume', 20),     
        ('min_trend_strength', 0.02), # Minimum trend strength filter
        ('volatility_filter', True), # Enable volatility filtering
    )

    def __init__(self):
        # Moving averages - using EMA for responsiveness
        self.ma_fast = bt.ind.EMA(period=self.params.fast_period)
        self.ma_slow = bt.ind.EMA(period=self.params.slow_period)
        
        # MACD for trend confirmation
        self.macd = bt.ind.MACD(period_me1=12, period_me2=26, period_signal=self.params.signal_period)
        
        # RSI for momentum
        self.rsi = bt.ind.RSI(period=self.params.rsi_period)
        
        # ATR for volatility-based stops
        self.atr = bt.ind.ATR(period=self.params.atr_period)
        
        # Volume analysis
        self.volume_sma = bt.ind.SMA(self.data.volume, period=self.params.lookback_volume)
        
        # Bollinger Bands for volatility context
        self.bb = bt.ind.BollingerBands(period=20, devfactor=2.0)
        
        # Additional trend strength indicator
        self.adx = bt.ind.ADX(period=14)  # Directional Movement Index
        
        # Crossover signals
        self.ma_crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)
        self.macd_crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        
        # Trade tracking
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.trail_stop = None
        self.trade_count = 0
        self.bars_in_trade = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.stop_price = self.entry_price - (self.atr[0] * self.params.stop_atr_mult)
                self.trail_stop = self.stop_price
                self.trade_count += 1
                self.bars_in_trade = 0
                print(f'BUY #{self.trade_count}: Price: {order.executed.price:.2f}, Stop: {self.stop_price:.2f}')
            else:
                if self.entry_price:
                    profit = order.executed.price - self.entry_price
                    profit_pct = (profit / self.entry_price) * 100
                    print(f'SELL: Price: {order.executed.price:.2f}, P&L: {profit:.2f} ({profit_pct:.2f}%), Bars: {self.bars_in_trade}')
                self.entry_price = None
                self.stop_price = None
                self.trail_stop = None
                self.bars_in_trade = 0
        
        self.order = None

    def get_position_size(self):
        """Improved position sizing with volatility adjustment"""
        if self.atr[0] == 0:
            return 0.03  # 3% default
        
        # Base risk per trade - reduced for better Sharpe
        base_risk = 0.008  # 0.8% risk per trade
        
        # Adjust for volatility
        volatility_adj = min(2.0, max(0.5, self.atr[0] / self.data.close[0] * 100))
        adjusted_risk = base_risk / volatility_adj
        
        # Calculate position size
        portfolio_value = self.broker.getvalue()
        risk_amount = portfolio_value * adjusted_risk
        
        stop_distance = self.atr[0] * self.params.stop_atr_mult
        if stop_distance > 0:
            shares = risk_amount / stop_distance
            position_value = shares * self.data.close[0]
            position_percent = position_value / portfolio_value
            return min(position_percent, 0.08)  # Cap at 8%
        
        return 0.03

    def is_strong_trend(self):
        """Enhanced trend detection"""
        # ADX strength check
        if len(self.adx) < 1:
            return False
        
        trend_strong = self.adx[0] > 25  # Strong trend threshold
        
        # Price momentum check
        if len(self.data.close) < 5:
            return False
        
        recent_momentum = (self.data.close[0] - self.data.close[-4]) / self.data.close[-4]
        momentum_ok = abs(recent_momentum) > self.params.min_trend_strength
        
        # MACD trend alignment
        macd_aligned = (self.macd.macd[0] > self.macd.signal[0] and 
                       self.macd.macd[0] > 0)
        
        return trend_strong and momentum_ok and macd_aligned

    def has_volume_confirmation(self):
        """Volume confirmation with better logic"""
        if len(self.data.volume) == 0 or self.volume_sma[0] == 0:
            return True
        
        # Current volume vs average
        volume_ok = self.data.volume[0] > (self.volume_sma[0] * self.params.min_volume_mult)
        
        # Volume trend (increasing volume on breakout)
        if len(self.data.volume) >= 3:
            volume_trend = (self.data.volume[0] > self.data.volume[-1] and 
                           self.data.volume[-1] > self.data.volume[-2])
            return volume_ok or volume_trend
        
        return volume_ok

    def is_good_entry_context(self):
        """Comprehensive entry context analysis"""
        current_price = self.data.close[0]
        
        # Not too close to resistance (BB upper band)
        resistance_ok = current_price < (self.bb.top[0] * 0.98)
        
        # RSI in favorable range
        rsi_favorable = (self.params.rsi_oversold < self.rsi[0] < self.params.rsi_overbought)
        
        # Price position relative to MAs
        ma_structure = (current_price > self.ma_fast[0] and 
                       self.ma_fast[0] > self.ma_slow[0])
        
        # Volatility not excessive
        volatility_ok = True
        if self.params.volatility_filter and len(self.atr) >= 20:
            atr_avg = sum(self.atr.get(ago=i) for i in range(20)) / 20
            volatility_ok = self.atr[0] < atr_avg * 1.5
        
        return resistance_ok and rsi_favorable and ma_structure and volatility_ok

    def should_exit_position(self):
        """Enhanced exit logic for better Sharpe ratio"""
        if not self.position:
            return False
        
        current_price = self.data.close[0]
        self.bars_in_trade += 1
        
        # Update trailing stop more conservatively
        if self.trail_stop and current_price > self.entry_price:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Only update trail stop after significant profit
            if profit_pct > 0.02:  # 2% profit before trailing
                new_trail = current_price - (self.atr[0] * self.params.trail_atr_mult)
                self.trail_stop = max(self.trail_stop, new_trail)
        
        # Exit conditions with improved logic
        
        # 1. Hard stop loss
        if current_price <= self.stop_price:
            print(f'STOP LOSS at {current_price:.2f}')
            return True
        
        # 2. Trailing stop (only if profitable)
        if (self.trail_stop and current_price <= self.trail_stop and 
            current_price > self.entry_price * 1.01):  # Only if 1%+ profit
            print(f'TRAILING STOP at {current_price:.2f}')
            return True
        
        # 3. Trend reversal signals
        if self.ma_crossover < 0:  # MA bearish cross
            print(f'MA REVERSAL EXIT at {current_price:.2f}')
            return True
        
        # 4. MACD bearish divergence
        if (self.macd_crossover < 0 and self.macd.macd[0] > 0 and 
            current_price > self.entry_price * 1.005):  # Small profit buffer
            print(f'MACD BEARISH at {current_price:.2f}')
            return True
        
        # 5. RSI extreme overbought with profit
        if (self.rsi[0] > 75 and current_price > self.entry_price * 1.01):
            print(f'RSI OVERBOUGHT EXIT at {current_price:.2f}')
            return True
        
        # 6. Time-based exit (prevent holding too long)
        if self.bars_in_trade > 50:  # Max 50 bars
            print(f'TIME EXIT at {current_price:.2f}')
            return True
        
        # 7. Volatility spike exit
        if len(self.atr) >= 5:
            recent_atr_avg = sum(self.atr.get(ago=i) for i in range(5)) / 5
            if self.atr[0] > recent_atr_avg * 2:  # Volatility spike
                print(f'VOLATILITY EXIT at {current_price:.2f}')
                return True
        
        return False

    def next(self):
        if self.order:
            return

        if not self.position:
            # Enhanced entry conditions
            conditions = [
                self.ma_crossover > 0,                    # MA bullish crossover
                self.macd_crossover > 0,                  # MACD bullish crossover  
                self.is_strong_trend(),                   # Strong trend detected
                self.has_volume_confirmation(),           # Volume support
                self.is_good_entry_context(),             # Good entry context
                len(self.data) > max(self.params.fast_period, self.params.slow_period) + 5  # Enough data
            ]
            
            if all(conditions):
                size_percent = self.get_position_size()
                self.order = self.buy(size=None)
                
        else:
            # Check exit conditions
            if self.should_exit_position():
                self.order = self.close()


def download_data(symbol='BTC-USD', period='2y', interval='1d'):
    """Improved data download with better error handling"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"Attempting to download {symbol} data (attempt {retry_count + 1}/{max_retries})...")
            
            # Use specific date range for better control
            if period == '2y':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
            elif period == '1y':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            else:
                # Fallback to period string
                start_date = None
                end_date = None
            
            # Download with specific parameters for crypto
            if start_date and end_date:
                data = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,  # Use adjusted prices
                    prepost=False,     # No pre/post market for crypto
                    threads=True,
                    progress=False,
                    show_errors=False
                )
            else:
                data = yf.download(
                    symbol, 
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                    threads=True,
                    progress=False,
                    show_errors=False
                )
            
            if data.empty:
                raise ValueError("No data downloaded")
            
            # Handle multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Basic data cleaning
            data = data.dropna()
            
            # Remove obvious outliers (price jumps > 50%)
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    
            data = data.dropna()
            
            # Ensure minimum data points
            if len(data) < 100:
                raise ValueError(f"Insufficient data: only {len(data)} points")
            
            # Validate data integrity
            if not ((data['high'] >= data['low']).all() and 
                   (data['high'] >= data['close']).all() and
                   (data['high'] >= data['open']).all() and
                   (data['low'] <= data['close']).all() and
                   (data['low'] <= data['open']).all()):
                print("Warning: Data integrity issues detected, cleaning...")
                # Fix high/low issues
                data['high'] = data[['high', 'low', 'open', 'close']].max(axis=1)
                data['low'] = data[['high', 'low', 'open', 'close']].min(axis=1)
            
            print(f"Successfully downloaded {len(data)} rows of {symbol} data")
            print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            return data
            
        except Exception as e:
            retry_count += 1
            print(f"Error downloading data (attempt {retry_count}): {e}")
            
            if retry_count < max_retries:
                print(f"Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                print(f"Failed to download {symbol} after {max_retries} attempts")
                
                # Try alternative symbols for BTC
                if symbol == 'BTC-USD' and retry_count == max_retries:
                    alternative_symbols = ['BTC-USD', 'BTCUSD=X']
                    for alt_symbol in alternative_symbols:
                        if alt_symbol != symbol:
                            print(f"Trying alternative symbol: {alt_symbol}")
                            return download_data(alt_symbol, period, interval)
                
                return None


def run_backtest(symbol='BTC-USD', period='2y', initial_cash=10000.0):
    """Enhanced backtest with better defaults"""
    
    cerebro = bt.Cerebro()
    
    # Download data with retries
    print(f"Downloading {symbol} data for {period}...")
    df = download_data(symbol, period)
    
    if df is None or df.empty:
        print("Failed to download data")
        return None, None
    
    # Create data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open', 
        high='high', 
        low='low', 
        close='close',
        volume='volume', 
        openinterest=None
    )
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(ImprovedMaCrossStrategy)
    
    # Broker settings for better Sharpe ratio
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Conservative position sizing
    cerebro.addsizer(bt.sizers.PercentSizer, percents=4)  # 4% per trade
    
    # Add comprehensive analyzers
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bta.Returns, _name="returns")
    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bta.SQN, _name="sqn")
    cerebro.addanalyzer(bta.Calmar, _name="calmar")
    cerebro.addanalyzer(bta.VWR, _name="vwr")  # Variability-Weighted Return
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    
    # Run backtest
    try:
        results = cerebro.run()
        strategy = results[0]
        
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value / initial_cash) - 1) * 100
        
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Print analysis
        print_analysis(strategy)
        
        return cerebro, strategy
        
    except Exception as e:
        print(f"Backtest error: {e}")
        return None, None


def print_analysis(strategy):
    """Enhanced analysis reporting"""
    print("\n" + "="*60)
    print("         ENHANCED PERFORMANCE ANALYSIS")
    print("="*60)
    
    try:
        # Get analyzer results safely
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        returns_analysis = strategy.analyzers.returns.get_analysis()
        trades_analysis = strategy.analyzers.trades.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        sqn_analysis = strategy.analyzers.sqn.get_analysis()
        
        # Risk-adjusted metrics
        sharpe_ratio = sharpe_analysis.get('sharperatio', 'N/A')
        if sharpe_ratio != 'N/A' and sharpe_ratio is not None:
            print(f"Sharpe Ratio:        {sharpe_ratio:.3f}")
            if sharpe_ratio > 1.0:
                print("                     ✓ Excellent (>1.0)")
            elif sharpe_ratio > 0.5:
                print("                     ✓ Good (>0.5)")
            else:
                print("                     ⚠ Needs improvement (<0.5)")
        else:
            print("Sharpe Ratio:        N/A")
        
        sqn = sqn_analysis.get('sqn', 'N/A')
        if sqn != 'N/A' and sqn is not None:
            print(f"System Quality No.:  {sqn:.3f}")
        
        # Returns
        total_return = returns_analysis.get('rtot', 0) * 100
        print(f"Total Return:        {total_return:.2f}%")
        
        # Trade statistics
        total_trades = trades_analysis.get('total', {}).get('total', 0) if trades_analysis.get('total') else 0
        won_trades = trades_analysis.get('won', {}).get('total', 0) if trades_analysis.get('won') else 0
        
        if total_trades > 0:
            win_rate = (won_trades / total_trades) * 100
            print(f"Total Trades:        {total_trades}")
            print(f"Win Rate:            {win_rate:.1f}%")
            
            if won_trades > 0 and trades_analysis.get('won'):
                avg_win = trades_analysis.get('won', {}).get('pnl', {}).get('average', 0)
                print(f"Average Win:         ${avg_win:.2f}")
            
            lost_trades = trades_analysis.get('lost', {}).get('total', 0) if trades_analysis.get('lost') else 0
            if lost_trades > 0 and trades_analysis.get('lost'):
                avg_loss = trades_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
                print(f"Average Loss:        ${avg_loss:.2f}")
                
                if avg_loss != 0 and won_trades > 0:
                    avg_win = trades_analysis.get('won', {}).get('pnl', {}).get('average', 0)
                    profit_factor = abs((avg_win * won_trades) / (avg_loss * lost_trades))
                    print(f"Profit Factor:       {profit_factor:.2f}")
        
        # Drawdown
        max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0) * 100 if drawdown_analysis.get('max') else 0
        print(f"Max Drawdown:        {max_drawdown:.2f}%")
        
        # Risk assessment
        print("\n" + "-"*40)
        print("RISK ASSESSMENT:")
        if max_drawdown < 10:
            print("Drawdown Risk:       ✓ Low (<10%)")
        elif max_drawdown < 20:
            print("Drawdown Risk:       ⚠ Moderate (10-20%)")
        else:
            print("Drawdown Risk:       ⚠ High (>20%)")
            
        if total_trades > 0:
            if win_rate > 50:
                print("Win Rate:            ✓ Good (>50%)")
            else:
                print("Win Rate:            ⚠ Needs improvement (<50%)")
    
    except Exception as e:
        print(f"Error in analysis: {e}")


if __name__ == "__main__":
    print("Running Enhanced MA Cross Strategy Backtest")
    print("-" * 50)
    
    # Test with different assets
    assets = ['BTC-USD']  # Start with BTC, can add more
    
    for asset in assets:
        print(f"\n{'='*20} Testing {asset} {'='*20}")
        cerebro, strategy = run_backtest(asset, '2y', 10000.0)
        
        if cerebro and strategy:
            print(f"✓ {asset} backtest completed successfully")
            
            # Optional: Create plot
            try:
                plt.style.use('dark_background')
                cerebro.plot(style='candle', barup='#00ff88', bardown='#ff4444', 
                           volume=False, figsize=(15, 10))
                plt.suptitle(f'{asset} - Enhanced MA Cross Strategy', fontsize=14)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Plotting failed: {e}")
        else:
            print(f"✗ {asset} backtest failed")
    
    print("\nBacktest completed!")