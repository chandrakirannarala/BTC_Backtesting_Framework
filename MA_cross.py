import backtrader as bt
import backtrader.analyzers as bta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import deque

warnings.filterwarnings('ignore')


class BalancedStrategy(bt.Strategy):
    params = (
        # Core MA parameters - balanced for signals
        ('fast_period', 8),
        ('slow_period', 21),
        ('signal_period', 9),
        
        # RSI parameters - relaxed but not extreme
        ('rsi_period', 14),
        ('rsi_oversold', 40),        # More relaxed
        ('rsi_overbought', 60),      # More relaxed
        
        # ATR and risk management - moderate
        ('atr_period', 14),
        ('stop_atr_mult', 1.8),      # Reasonable stops
        ('trail_atr_mult', 1.2),     # Moderate trailing
        
        # Position sizing - moderate risk
        ('base_risk', 0.01),         # 1% risk per trade
        ('max_position', 0.05),      # Max 5% position size
        ('volatility_target', 0.15),
        
        # Exit optimization
        ('profit_target_mult', 2.0), # Reasonable profit target
        ('max_bars_hold', 25),       # Moderate holding period
    )

    def __init__(self):
        # Core indicators
        self.ma_fast = bt.ind.EMA(period=self.params.fast_period)
        self.ma_slow = bt.ind.EMA(period=self.params.slow_period)
        
        # MACD
        self.macd = bt.ind.MACD(period_me1=12, period_me2=26, period_signal=self.params.signal_period)
        
        # RSI
        self.rsi = bt.ind.RSI(period=self.params.rsi_period)
        
        # ATR for volatility
        self.atr = bt.ind.ATR(period=self.params.atr_period)
        
        # Volume
        self.volume_sma = bt.ind.SMA(self.data.volume, period=20)
        self.volume_ratio = self.data.volume / self.volume_sma
        
        # Crossover signals
        self.ma_crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)
        self.macd_crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        
        # Trade tracking
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.profit_target = None
        self.trail_stop = None
        self.trade_count = 0
        self.bars_in_trade = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Performance tracking
        self.trade_returns = []
        self.daily_returns = []
        self.portfolio_values = []
        
        # Debug tracking
        self.entry_attempts = 0
        self.entry_failures = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.stop_price = self.entry_price - (self.atr[0] * self.params.stop_atr_mult)
                self.profit_target = self.entry_price + (self.atr[0] * self.params.profit_target_mult)
                self.trail_stop = self.stop_price
                self.trade_count += 1
                self.bars_in_trade = 0
                
                print(f'BUY #{self.trade_count}: Price: {order.executed.price:.2f}, '
                      f'Stop: {self.stop_price:.2f}, Target: {self.profit_target:.2f}')
            else:
                if self.entry_price:
                    profit = order.executed.price - self.entry_price
                    profit_pct = (profit / self.entry_price) * 100
                    self.trade_returns.append(profit_pct)
                    
                    if profit < 0:
                        self.consecutive_losses += 1
                        self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                    else:
                        self.consecutive_losses = 0
                    
                    print(f'SELL: Price: {order.executed.price:.2f}, P&L: {profit:.2f} '
                          f'({profit_pct:.2f}%), Bars: {self.bars_in_trade}')
                
                self.entry_price = None
                self.stop_price = None
                self.profit_target = None
                self.trail_stop = None
                self.bars_in_trade = 0
        
        self.order = None

    def get_position_size(self):
        """Balanced position sizing"""
        portfolio_value = self.broker.getvalue()
        
        # Reduce size after consecutive losses (but not too much)
        loss_adjustment = max(0.5, 1 - (self.consecutive_losses * 0.15))
        
        # Base risk adjusted for losses
        adjusted_risk = self.params.base_risk * loss_adjustment
        
        if self.atr[0] > 0:
            risk_amount = portfolio_value * adjusted_risk
            stop_distance = self.atr[0] * self.params.stop_atr_mult
            shares = risk_amount / stop_distance
            position_value = shares * self.data.close[0]
            position_percent = position_value / portfolio_value
            
            # Cap at maximum position size
            final_size = min(position_percent, self.params.max_position)
            return max(final_size, 0.01)  # Minimum 1%
        
        return 0.01

    def should_exit_position(self):
        """Balanced exit logic"""
        if not self.position:
            return False
        
        current_price = self.data.close[0]
        self.bars_in_trade += 1
        
        # Update trailing stop when profitable
        if self.trail_stop and current_price > self.entry_price * 1.01:  # 1% profit
            new_trail = current_price - (self.atr[0] * self.params.trail_atr_mult)
            self.trail_stop = max(self.trail_stop, new_trail)
        
        # Exit conditions
        
        # 1. Hard stop loss
        if current_price <= self.stop_price:
            print(f'STOP LOSS at {current_price:.2f}')
            return True
        
        # 2. Profit target
        if current_price >= self.profit_target:
            print(f'PROFIT TARGET at {current_price:.2f}')
            return True
        
        # 3. Trailing stop (only if profitable)
        if (self.trail_stop and current_price <= self.trail_stop and 
            current_price > self.entry_price * 1.005):  # 0.5% minimum profit
            print(f'TRAILING STOP at {current_price:.2f}')
            return True
        
        # 4. MA trend reversal
        if self.ma_crossover < 0:
            print(f'TREND REVERSAL at {current_price:.2f}')
            return True
        
        # 5. MACD bearish crossover (with small profit buffer)
        if self.macd_crossover < 0 and current_price > self.entry_price * 1.003:
            print(f'MACD REVERSAL at {current_price:.2f}')
            return True
        
        # 6. Time exit
        if self.bars_in_trade > self.params.max_bars_hold:
            print(f'TIME EXIT at {current_price:.2f}')
            return True
        
        # 7. RSI extreme overbought (with profit buffer)
        if self.rsi[0] > 75 and current_price > self.entry_price * 1.005:
            print(f'RSI OVERBOUGHT at {current_price:.2f}')
            return True
        
        return False

    def check_entry_conditions_debug(self):
        """Debug entry conditions to see what's failing"""
        self.entry_attempts += 1
        conditions = {}
        
        try:
            # Check each condition individually
            conditions['ma_crossover'] = self.ma_crossover > 0
            conditions['macd_crossover'] = self.macd_crossover > 0
            conditions['rsi_range'] = self.params.rsi_oversold < self.rsi[0] < self.params.rsi_overbought
            conditions['sufficient_data'] = len(self.data) > 50
            conditions['no_consecutive_losses'] = self.consecutive_losses < 4  # Allow more losses
            
            # Optional conditions (more forgiving)
            try:
                conditions['volume_ok'] = self.volume_ratio[0] > 0.8  # Very low bar
            except:
                conditions['volume_ok'] = True
            
            try:
                conditions['price_above_fast_ma'] = self.data.close[0] > self.ma_fast[0]
            except:
                conditions['price_above_fast_ma'] = True
            
            # Log failures for debugging
            failed_conditions = [k for k, v in conditions.items() if not v]
            if failed_conditions and self.entry_attempts % 50 == 0:  # Log every 50 attempts
                self.entry_failures.append({
                    'bar': len(self.data),
                    'failed': failed_conditions,
                    'rsi': self.rsi[0] if len(self.rsi) > 0 else 'N/A',
                    'ma_cross': self.ma_crossover[0] if len(self.ma_crossover) > 0 else 'N/A',
                    'macd_cross': self.macd_crossover[0] if len(self.macd_crossover) > 0 else 'N/A'
                })
        
        except Exception as e:
            print(f"Error in entry condition check: {e}")
            return False
        
        return all(conditions.values())

    def next(self):
        # Track performance
        if len(self.portfolio_values) > 0:
            daily_return = (self.broker.getvalue() - self.portfolio_values[-1]) / self.portfolio_values[-1]
            self.daily_returns.append(daily_return)
        self.portfolio_values.append(self.broker.getvalue())
        
        if self.order:
            return

        if not self.position:
            # SIMPLIFIED entry conditions - focus on core signals
            try:
                # Core conditions (must have all)
                core_conditions = [
                    self.ma_crossover > 0,                    # MA bullish crossover
                    self.macd_crossover > 0,                  # MACD bullish crossover
                    len(self.data) > 50,                      # Sufficient data
                ]
                
                # Optional quality filters (more forgiving)
                quality_conditions = [
                    self.params.rsi_oversold < self.rsi[0] < self.params.rsi_overbought,  # RSI in range
                    self.consecutive_losses < 4,              # Allow up to 3 losses
                ]
                
                # Combine conditions - require all core + most quality
                all_conditions = core_conditions + quality_conditions
                passed_conditions = sum(all_conditions)
                required_conditions = len(core_conditions) + len(quality_conditions) - 1  # Allow 1 quality failure
                
                if passed_conditions >= required_conditions:
                    size_percent = self.get_position_size()
                    if size_percent > 0.005:  # Minimum threshold
                        self.order = self.buy(size=None)
                        print(f'Entry conditions met - Balanced strategy')
                else:
                    # Debug logging
                    if self.entry_attempts % 100 == 0:  # Every 100 bars
                        print(f"Bar {len(self.data)}: Passed {passed_conditions}/{len(all_conditions)} conditions")
                        if len(self.rsi) > 0:
                            print(f"  RSI: {self.rsi[0]:.1f}, MA Cross: {self.ma_crossover[0]:.0f}, MACD Cross: {self.macd_crossover[0]:.0f}")
            
            except Exception as e:
                # Fallback to most basic conditions
                if (self.ma_crossover > 0 and len(self.data) > 50):
                    size_percent = self.get_position_size()
                    if size_percent > 0.005:
                        self.order = self.buy(size=None)
                        print(f'Entry conditions met - Basic fallback')
        else:
            # Check exit conditions
            if self.should_exit_position():
                self.order = self.close()


def run_enhanced_backtest(symbol='BTC-USD', period='2y', initial_cash=10000.0):
    """Enhanced backtest with balanced approach"""
    
    cerebro = bt.Cerebro()
    
    print(f"Loading {symbol} data...")
    try:
        df = pd.read_csv('BTC-USD.csv', index_col=0, parse_dates=True)
        
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        
        df = df.dropna()
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None, None
        
        print(f"Loaded {len(df)} rows of data from {df.index[0].date()} to {df.index[-1].date()}")
        
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None, None
    
    try:
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
    except Exception as e:
        print(f"Error creating data feed: {e}")
        return None, None
    
    cerebro.addstrategy(BalancedStrategy)
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0008)
    
    # Moderate position size
    cerebro.addsizer(bt.sizers.PercentSizer, percents=2)  # 2% base
    
    cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.02, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bta.Returns, _name="returns")
    cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bta.SQN, _name="sqn")
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    
    try:
        results = cerebro.run()
        strategy = results[0]
        
        final_value = cerebro.broker.getvalue()
        total_return = ((final_value / initial_cash) - 1) * 100
        
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Print debug info if no trades
        if strategy.trade_count == 0:
            print(f"\n🔍 DEBUG INFO - No trades executed:")
            print(f"Entry attempts: {strategy.entry_attempts}")
            if len(strategy.entry_failures) > 0:
                print("Recent entry failures:")
                for failure in strategy.entry_failures[-5:]:
                    print(f"  Bar {failure['bar']}: Failed {failure['failed']}")
                    print(f"    RSI: {failure['rsi']}, MA Cross: {failure['ma_cross']}, MACD Cross: {failure['macd_cross']}")
        
        print_enhanced_analysis(strategy, initial_cash, final_value)
        
        return cerebro, strategy
        
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def print_enhanced_analysis(strategy, initial_cash, final_value):
    """Enhanced performance analysis"""
    print("\n" + "="*80)
    print("         BALANCED STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    try:
        sharpe_analysis = getattr(strategy.analyzers, 'sharpe', None)
        returns_analysis = getattr(strategy.analyzers, 'returns', None)  
        trades_analysis = getattr(strategy.analyzers, 'trades', None)
        drawdown_analysis = getattr(strategy.analyzers, 'drawdown', None)
        sqn_analysis = getattr(strategy.analyzers, 'sqn', None)
        
        total_return = ((final_value / initial_cash) - 1) * 100
        print(f"Initial Capital:      ${initial_cash:,.2f}")
        print(f"Final Value:          ${final_value:,.2f}")
        print(f"Total Return:         {total_return:.2f}%")
        
        if sharpe_analysis:
            sharpe_data = sharpe_analysis.get_analysis()
            sharpe_ratio = sharpe_data.get('sharperatio', None)
            if sharpe_ratio is not None:
                print(f"Sharpe Ratio:         {sharpe_ratio:.3f}")
                if sharpe_ratio > 1.0:
                    print("                      ✓ Excellent (>1.0)")
                elif sharpe_ratio > 0.5:
                    print("                      ✓ Good (>0.5)")
                elif sharpe_ratio > 0:
                    print("                      ⚠ Fair (>0)")
                else:
                    print("                      ⚠ Needs improvement (<0)")
            else:
                print("Sharpe Ratio:         N/A")
        
        if sqn_analysis:
            sqn_data = sqn_analysis.get_analysis()
            sqn = sqn_data.get('sqn', None)
            if sqn is not None:
                print(f"System Quality No.:   {sqn:.3f}")
        
        if trades_analysis:
            trades_data = trades_analysis.get_analysis()
            total_trades = trades_data.get('total', {}).get('total', 0) if trades_data.get('total') else 0
            won_trades = trades_data.get('won', {}).get('total', 0) if trades_data.get('won') else 0
            lost_trades = trades_data.get('lost', {}).get('total', 0) if trades_data.get('lost') else 0
            
            if total_trades > 0:
                win_rate = (won_trades / total_trades) * 100
                print(f"\nTrade Statistics:")
                print(f"Total Trades:         {total_trades}")
                print(f"Winning Trades:       {won_trades}")
                print(f"Losing Trades:        {lost_trades}")
                print(f"Win Rate:             {win_rate:.1f}%")
                
                if won_trades > 0 and trades_data.get('won'):
                    avg_win = trades_data.get('won', {}).get('pnl', {}).get('average', 0)
                    print(f"Average Win:          ${avg_win:.2f}")
                
                if lost_trades > 0 and trades_data.get('lost'):
                    avg_loss = trades_data.get('lost', {}).get('pnl', {}).get('average', 0)
                    print(f"Average Loss:         ${avg_loss:.2f}")
                    
                    if avg_loss != 0 and won_trades > 0:
                        avg_win = trades_data.get('won', {}).get('pnl', {}).get('average', 0)
                        profit_factor = abs((avg_win * won_trades) / (avg_loss * lost_trades))
                        print(f"Profit Factor:        {profit_factor:.2f}")
            else:
                print(f"\n⚠️  NO TRADES EXECUTED - Strategy too restrictive")
        
        if drawdown_analysis:
            drawdown_data = drawdown_analysis.get_analysis()
            max_drawdown = drawdown_data.get('max', {}).get('drawdown', 0)
            max_drawdown_pct = max_drawdown * 100 if max_drawdown else 0
            
            print(f"\nRisk Analysis:")
            print(f"Max Drawdown:         {max_drawdown_pct:.2f}%")
            
            if max_drawdown_pct < 5:
                print("                      ✓ Very Low Risk (<5%)")
            elif max_drawdown_pct < 10:
                print("                      ✓ Low Risk (<10%)")
            elif max_drawdown_pct < 20:
                print("                      ⚠ Moderate Risk (10-20%)")
            else:
                print("                      ⚠ High Risk (>20%)")
        
        print("="*80)
        
    except Exception as e:
        print(f"Error in analysis: {e}")


def optimize_strategy_parameters():
    """Balanced parameter optimization"""
    print("Starting balanced parameter optimization...")
    
    # More aggressive parameters to ensure trades
    test_combinations = [
        {'fast_period': 5, 'slow_period': 15, 'rsi_oversold': 45, 'rsi_overbought': 55, 
         'stop_atr_mult': 1.5, 'base_risk': 0.015},
        {'fast_period': 8, 'slow_period': 21, 'rsi_oversold': 35, 'rsi_overbought': 65, 
         'stop_atr_mult': 1.8, 'base_risk': 0.012},
        {'fast_period': 10, 'slow_period': 25, 'rsi_oversold': 30, 'rsi_overbought': 70, 
         'stop_atr_mult': 2.0, 'base_risk': 0.010},
    ]
    
    best_score = -999
    best_params = {}
    results = []
    
    for i, params in enumerate(test_combinations):
        print(f"\nTesting combination {i+1}/{len(test_combinations)}")
        print(f"Parameters: {params}")
        
        try:
            cerebro = bt.Cerebro()
            
            df = pd.read_csv('BTC-USD.csv', index_col=0, parse_dates=True)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            if 'adj_close' in df.columns:
                df['close'] = df['adj_close']
            df = df.dropna()
            
            data = bt.feeds.PandasData(
                dataname=df, datetime=None,
                open='open', high='high', low='low', close='close',
                volume='volume', openinterest=None
            )
            cerebro.adddata(data)
            
            cerebro.addstrategy(BalancedStrategy, **params)
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0008)
            cerebro.addsizer(bt.sizers.PercentSizer, percents=2)
            
            cerebro.addanalyzer(bta.SharpeRatio, _name="sharpe", riskfreerate=0.02)
            cerebro.addanalyzer(bta.Returns, _name="returns")
            cerebro.addanalyzer(bta.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bta.TradeAnalyzer, _name="trades")
            
            result = cerebro.run()[0]
            
            sharpe_data = result.analyzers.sharpe.get_analysis()
            returns_data = result.analyzers.returns.get_analysis()
            drawdown_data = result.analyzers.drawdown.get_analysis()
            trades_data = result.analyzers.trades.get_analysis()
            
            sharpe = sharpe_data.get('sharperatio', 0) if sharpe_data.get('sharperatio') is not None else 0
            returns = returns_data.get('rtot', 0) * 100 if returns_data.get('rtot') is not None else 0
            max_dd = drawdown_data.get('max', {}).get('drawdown', 0) * 100 if drawdown_data.get('max', {}).get('drawdown') is not None else 0
            total_trades = trades_data.get('total', {}).get('total', 0) if trades_data.get('total') else 0
            
            # Balanced scoring (penalize no trades)
            if total_trades == 0:
                score = -10  # Heavy penalty for no trades
            else:
                score = sharpe * 0.4 + (returns/100) * 0.3 - (max_dd/100) * 0.2 + min(total_trades/10, 1) * 0.1
            
            print(f"Results: Trades: {total_trades}, Sharpe: {sharpe:.3f}, Return: {returns:.1f}%, MaxDD: {max_dd:.1f}%, Score: {score:.3f}")
            
            results.append({
                'params': params, 'sharpe': sharpe, 'returns': returns, 
                'max_dd': max_dd, 'score': score, 'trades': total_trades
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        except Exception as e:
            print(f"Error testing parameters: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("BALANCED OPTIMIZATION RESULTS")
    print('='*60)
    
    if results:
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("Top parameter combinations:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.3f}")
            print(f"   Trades: {result['trades']}, Sharpe: {result['sharpe']:.3f}, Return: {result['returns']:.1f}%, MaxDD: {result['max_dd']:.1f}%")
            print(f"   Parameters: {result['params']}")
        
        print(f"\nBest Balanced Parameters: {best_params}")
    else:
        print("No successful optimization runs")
    
    return best_params


def create_simple_clean_plot(cerebro):
    """Create a simple, completely grid-free plot"""
    try:
        print("Creating simple grid-free plot...")
        
        # Get the data safely
        data_feed = cerebro.datas[0]
        
        # Extract data manually to avoid array issues
        dates = []
        closes = []
        volumes = []
        
        for i in range(len(data_feed)):
            try:
                dates.append(data_feed.datetime.date(i))
                closes.append(data_feed.close[i])
                volumes.append(data_feed.volume[i])
            except IndexError:
                break
        
        if len(dates) == 0:
            print("No data available for plotting")
            return
        
        # Create figure with clean style
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='black')
        
        # Price chart
        ax1.plot(dates, closes, color='lime', linewidth=1.5, label='BTC Close Price')
        ax1.set_title('BTC Balanced Trading Strategy - Price Chart', color='white', fontsize=14, pad=20)
        ax1.set_ylabel('Price ($)', color='white')
        ax1.legend(loc='upper left')
        
        # Volume chart
        ax2.bar(dates, volumes, color='cyan', alpha=0.6, width=1)
        ax2.set_title('Volume', color='white', fontsize=12)
        ax2.set_ylabel('Volume', color='white')
        ax2.set_xlabel('Date', color='white')
        
        # Remove ALL grids and clean up both axes
        for ax in [ax1, ax2]:
            ax.grid(False)
            ax.set_facecolor('black')
            
            # Clean up spines
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_linewidth(0.8)
            
            # Clean tick parameters
            ax.tick_params(colors='white', which='both')
            ax.tick_params(axis='x', rotation=45)
        
        # Tight layout and show
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        
        print("✅ Grid-free plot created successfully!")
        
    except Exception as e:
        print(f"Plotting failed: {e}")


def check_boolean_operation(cerebro, strategy):
    """Safe boolean check"""
    if cerebro is None or strategy is None:
        return False
    return True


if __name__ == "__main__":
    print("🚀 BALANCED TRADING STRATEGY BACKTEST")
    print("-" * 60)
    
    print("\n1. Running Balanced Backtest...")
    cerebro, strategy = run_enhanced_backtest('BTC-USD', '2y', 10000.0)
    
    if check_boolean_operation(cerebro, strategy):
        print("✅ Balanced backtest completed successfully!")
        
        optimize = input("\n2. Run balanced parameter optimization? (y/n): ").lower().strip()
        if optimize == 'y':
            best_params = optimize_strategy_parameters()
        
        plot = input("\n3. Generate plot? (y/n): ").lower().strip()
        if plot == 'y':
            create_simple_clean_plot(cerebro)
    else:
        print("❌ Backtest failed")
    
    print("\n🎯 Balanced Analysis Complete!")