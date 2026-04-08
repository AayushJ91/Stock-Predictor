"""
Backtesting Pipeline

Simulates a trading strategy based on sentiment predictions:
- Uses initial capital to invest in 5 banking stocks
- Fetches historical articles and makes predictions
- Compares predictions against actual price movements
- Tracks portfolio performance and trading results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from prediction_pipeline import StockMovementPredictor

ROOT   = Path(__file__).parent.parent
PROC   = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results' / 'backtesting'
RESULTS.mkdir(parents=True, exist_ok=True)


class BacktestingEngine:
    """
    Backtesting engine that:
    1. Loads historical articles and their predicted movements
    2. Fetches actual stock prices from yfinance
    3. Simulates trades based on predictions
    4. Tracks portfolio performance
    """
    
    TICKERS = ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS']
    TICKER_NAMES = {
        'HDFCBANK.NS': 'HDFCBANK',
        'ICICIBANK.NS': 'ICICIBANK',
        'SBIN.NS': 'SBIN',
        'AXISBANK.NS': 'AXISBANK',
        'KOTAKBANK.NS': 'KOTAKBANK'
    }
    
    def __init__(self, initial_capital=100000, verbose=True):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in INR (default: 100,000)
        verbose : bool
            Print progress information
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.verbose = verbose
        self.portfolio = {}  # {ticker: shares_held}
        self.trades = []     # List of all trades executed
        self.daily_values = []  # List of daily portfolio values
        self.portfolio_history = []  # Daily portfolio snapshots
        
        # Load predictor
        try:
            self.predictor = StockMovementPredictor(verbose=False)
        except FileNotFoundError:
            raise FileNotFoundError("⚠️  Run notebook 04 first to train and save the model.")
    
    def fetch_historical_data(self, start_date, end_date):
        """
        Fetch historical price data from yfinance
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date (e.g., '2024-01-01')
        end_date : str or datetime
            End date (e.g., '2024-12-31')
            
        Returns:
        --------
        dict : {ticker: pd.DataFrame with OHLCV data}
        """
        if self.verbose:
            print(f"\n📊 Fetching {len(self.TICKERS)} tickers from {start_date} to {end_date}...")
        
        price_data = {}
        for ticker in self.TICKERS:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    print(f"⚠️  No data for {ticker}")
                else:
                    price_data[ticker] = df
                    if self.verbose:
                        print(f"  ✓ {ticker}: {len(df)} days")
            except Exception as e:
                print(f"⚠️  Error fetching {ticker}: {e}")
        
        return price_data
    
    def load_articles(self, start_date=None, end_date=None):
        """
        Load articles from processed features.csv
        
        Parameters:
        -----------
        start_date : str or datetime, optional
            Filter articles from this date onward
        end_date : str or datetime, optional
            Filter articles up to this date
            
        Returns:
        --------
        pd.DataFrame : Articles with columns [ticker, date, headline, text, ...]
        """
        if self.verbose:
            print("\n📰 Loading articles from features.csv...")
        
        try:
            df = pd.read_csv(PROC / 'features.csv', parse_dates=['date'])
        except FileNotFoundError:
            raise FileNotFoundError("⚠️  features.csv not found. Run notebooks 01-03 first.")
        
        # Filter for our 5 tickers
        ticker_names = list(self.TICKER_NAMES.values())
        df = df[df['ticker'].isin(ticker_names)].copy()
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        if df.empty:
            raise ValueError("No articles found in the specified date range")
        
        df = df.sort_values('date').reset_index(drop=True)
        
        if self.verbose:
            print(f"  ✓ Loaded {len(df):,} articles")
            print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"    Tickers: {df['ticker'].unique().tolist()}")
        
        return df
    
    def get_actual_price_move(self, ticker, date, horizon_days=1):
        """
        Get actual price movement after a given date
        
        Parameters:
        -----------
        ticker : str
            Ticker with .NS suffix (e.g., 'HDFCBANK.NS')
        date : datetime
            Article publication date
        horizon_days : int
            Number of days ahead to look (default: 1)
            
        Returns:
        --------
        str : 'UP', 'DOWN', or 'NEUTRAL'
        float : Price change percentage
        """
        if ticker not in self.price_data:
            return None, None
        
        prices = self.price_data[ticker]
        
        # Find the first trading day on or after the article date
        future_prices = prices[prices.index.date >= date.date()]
        if len(future_prices) < horizon_days + 1:
            return None, None
        
        start_price = future_prices.iloc[0]['Open']
        end_price = future_prices.iloc[min(horizon_days, len(future_prices) - 1)]['Close']
        
        pct_change = ((end_price - start_price) / start_price) * 100
        
        if pct_change > 1:
            movement = 'UP'
        elif pct_change < -1:
            movement = 'DOWN'
        else:
            movement = 'NEUTRAL'
        
        return movement, pct_change
    
    def backtest(self, articles_df, price_data_dict, position_size_pct=0.2, horizon_days=1):
        """
        Run the backtest
        
        Parameters:
        -----------
        articles_df : pd.DataFrame
            Articles with [ticker, date, headline, text]
        price_data_dict : dict
            {ticker: pd.DataFrame with OHLCV}
        position_size_pct : float
            Percentage of capital to allocate per trade (default: 20%)
        horizon_days : int
            Days ahead to measure price movement (default: 1)
        """
        self.price_data = price_data_dict
        self.position_size_pct = position_size_pct
        
        if self.verbose:
            print(f"\n🎯 Starting backtest...")
            print(f"   Initial capital: ₹{self.initial_capital:,.0f}")
            print(f"   Position size: {position_size_pct*100:.0f}% per trade")
            print(f"   Horizon: {horizon_days} day(s)")
        
        # Initialize portfolios for each ticker
        for ticker in self.TICKERS:
            ticker_name = self.TICKER_NAMES[ticker]
            self.portfolio[ticker_name] = 0
        
        correct_predictions = 0
        total_predictions = 0
        
        # Process each article
        for idx, row in articles_df.iterrows():
            ticker_name = row['ticker']
            date = row['date']
            headline = row['headline']
            text = row.get('text', '')
            
            # Get prediction
            try:
                pred = self.predictor.predict(
                    headline, 
                    date.strftime('%Y-%m-%d'), 
                    ticker=ticker_name,
                    body=text,
                    verbose=False
                )
                predicted_movement = pred['predicted_movement']
                confidence = pred['confidence']
            except Exception as e:
                if self.verbose and idx % 100 == 0:
                    print(f"⚠️  Error predicting for {ticker_name}: {e}")
                continue
            
            # Get actual movement
            ticker_yf = [t for t in self.TICKERS if self.TICKER_NAMES[t] == ticker_name][0]
            actual_movement, pct_change = self.get_actual_price_move(
                ticker_yf, date, horizon_days
            )
            
            if actual_movement is None:
                continue
            
            total_predictions += 1
            
            # Check if prediction was correct
            is_correct = predicted_movement == actual_movement
            if is_correct:
                correct_predictions += 1
            
            # Execute trade if high confidence and predicted UP
            if predicted_movement == 'UP' and confidence > 0.55:
                self._execute_trade(
                    ticker_name, ticker_yf, date, 
                    investment_amount=self.initial_capital * position_size_pct,
                    expected_return=pct_change
                )
            
            # Record trade
            self.trades.append({
                'date': date,
                'ticker': ticker_name,
                'headline': headline[:100],
                'predicted_movement': predicted_movement,
                'confidence': confidence,
                'actual_movement': actual_movement,
                'pct_change': pct_change,
                'correct': is_correct
            })
        
        if self.verbose:
            print(f"\n✅ Backtest complete!")
            print(f"   Total predictions: {total_predictions}")
            print(f"   Correct predictions: {correct_predictions} ({correct_predictions/total_predictions*100:.1f}%)")
            print(f"   Total trades executed: {len(self.trades)}")
        
        return self.trades
    
    def _execute_trade(self, ticker_name, ticker_yf, trade_date, 
                       investment_amount, expected_return):
        """Execute a single trade"""
        try:
            # Get price at trade date
            prices = self.price_data[ticker_yf]
            trade_date_prices = prices[prices.index.date >= trade_date.date()]
            
            if len(trade_date_prices) == 0:
                return
            
            buy_price = trade_date_prices.iloc[0]['Open']
            shares = investment_amount / buy_price
            
            # Record trade
            self.portfolio[ticker_name] += shares
            self.current_capital -= investment_amount
        except Exception as e:
            pass  # Silently skip failed trades
    
    def calculate_returns(self):
        """
        Calculate backtesting returns and metrics
        
        Returns:
        --------
        dict : Performance metrics
        """
        trades_df = pd.DataFrame(self.trades)
        
        if trades_df.empty:
            print("⚠️  No trades executed")
            return {}
        
        correct = trades_df['correct'].sum()
        total = len(trades_df)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate returns for UP predictions only
        up_trades = trades_df[trades_df['predicted_movement'] == 'UP']
        if len(up_trades) > 0:
            correct_up = up_trades['correct'].sum()
            up_accuracy = correct_up / len(up_trades)
        else:
            up_accuracy = 0
        
        # Average confidence
        avg_confidence = trades_df['confidence'].mean()
        
        metrics = {
            'total_trades': len(trades_df),
            'correct_predictions': correct,
            'total_predictions': total,
            'overall_accuracy': f"{accuracy*100:.1f}%",
            'up_prediction_accuracy': f"{up_accuracy*100:.1f}%",
            'average_confidence': f"{avg_confidence:.2f}",
            'avg_predicted_pct_change': f"{trades_df[trades_df['predicted_movement']=='UP']['pct_change'].mean():.2f}%",
            'avg_actual_pct_change': f"{trades_df['pct_change'].mean():.2f}%",
        }
        
        return metrics, trades_df
    
    def generate_report(self, output_path=None):
        """Generate a detailed backtesting report"""
        metrics, trades_df = self.calculate_returns()
        
        if output_path is None:
            output_path = RESULTS / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        trades_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\n📈 Backtesting Results:")
            print(f"   Overall Accuracy: {metrics['overall_accuracy']}")
            print(f"   UP Prediction Accuracy: {metrics['up_prediction_accuracy']}")
            print(f"   Average Confidence: {metrics['average_confidence']}")
            print(f"   Avg Predicted Return: {metrics['avg_predicted_pct_change']}")
            print(f"   Avg Actual Return: {metrics['avg_actual_pct_change']}")
            print(f"\n   Report saved to: {output_path}")
        
        return metrics, trades_df


def run_backtest(initial_capital=100000, start_date='2024-01-01', 
                 end_date='2024-12-31', horizon_days=1, 
                 position_size_pct=0.2, verbose=True):
    """
    Run a complete backtest from scratch
    
    Parameters:
    -----------
    initial_capital : float
        Starting capital in INR
    start_date : str
        Backtest start date
    end_date : str
        Backtest end date
    horizon_days : int
        Days ahead to measure price movement
    position_size_pct : float
        Percentage of capital per trade
    verbose : bool
        Print progress
        
    Returns:
    --------
    BacktestingEngine : The backtesting engine with results
    """
    engine = BacktestingEngine(initial_capital, verbose=verbose)
    
    # Load data
    price_data = engine.fetch_historical_data(start_date, end_date)
    articles = engine.load_articles(start_date, end_date)
    
    # Run backtest
    trades = engine.backtest(articles, price_data, position_size_pct, horizon_days)
    
    # Generate report
    metrics, trades_df = engine.generate_report()
    
    return engine


if __name__ == '__main__':
    # Example usage
    engine = run_backtest(
        initial_capital=100000,
        start_date='2024-01-01',
        end_date='2024-12-31',
        horizon_days=1,
        position_size_pct=0.2
    )
