"""
Combined Prediction Pipeline

Integrates sentiment and trend predictors to make unified stock movement predictions.
- Loads models from both sentiment-predictor and trend-predictor
- Makes predictions from headlines (sentiment) and technical data (trend)
- Combines results using weighted ensemble based on model confidence
- Provides advisory with combined confidence and reasoning

Usage:
------
from combined_prediction_pipeline import CombinedPredictionPipeline

predictor = CombinedPredictionPipeline()
result = predictor.predict(
    headline="HDFC Bank Q3 net profit rises 18%",
    date="2024-01-15",
    ticker="HDFCBANK",
    combine_method="weighted"  # 'weighted', 'majority', or 'confidence_threshold'
)
print(result)

# Output:
# {
#     'sentiment': {'movement': 'UP', 'confidence': 0.78},
#     'trend': {'movement': 'UP', 'confidence': 0.72},
#     'combined': {'movement': 'UP', 'confidence': 0.75, 'consensus': 'strong'},
#     'recommendation': 'BUY',
#     'reasoning': {...},
#     'timestamp': '2024-01-15'
# }
"""

import os
import pickle
import warnings
import importlib.util
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings('ignore')

# Add submodule paths
ROOT = Path(__file__).parent
SENTIMENT_DIR = ROOT / 'sentiment-predictor'
TREND_DIR = ROOT / 'trend-predictor'

# Import sentiment predictor using importlib
sentiment_spec = importlib.util.spec_from_file_location(
    "sentiment_pipeline",
    str(SENTIMENT_DIR / 'prediction_pipeline.py')
)
sentiment_module = importlib.util.module_from_spec(sentiment_spec)
sentiment_spec.loader.exec_module(sentiment_module)
SentimentPredictor = sentiment_module.StockMovementPredictor

# Import trend predictor components
trend_spec = importlib.util.spec_from_file_location(
    "trend_pipeline",
    str(TREND_DIR / 'prediction_pipeline.py')
)
trend_module = importlib.util.module_from_spec(trend_spec)
trend_spec.loader.exec_module(trend_module)

CNNLSTMDualHead = trend_module.CNNLSTMDualHead
FEATURE_COLS = trend_module.FEATURE_COLS
TREND_LABELS = trend_module.TREND_LABELS
TICKER_MAP = trend_module.TICKER_MAP
compute_technical_features = trend_module.compute_technical_features
download_price_data = trend_module.download_price_data
build_inference_batch = trend_module.build_inference_batch


class CombinedPredictionPipeline:
    """
    Unified prediction pipeline combining sentiment and trend models.
    
    Attributes:
    -----------
    sentiment_predictor : StockMovementPredictor
        Trained ensemble model for sentiment-based predictions
    trend_model : CNNLSTMDualHead
        Trained CNN-LSTM model for technical trend predictions
    scaler : sklearn.preprocessing.StandardScaler
        Scaler for trend model features
    yr_mean, yr_std : float
        Normalization parameters for price regression
    device : torch.device
        CPU or CUDA device
    """
    
    # Model weights for combining predictions (adjustable)
    DEFAULT_WEIGHTS = {
        'sentiment': 0.5,  # Weight for sentiment model
        'trend': 0.5,      # Weight for trend model
    }
    
    # Performance thresholds (from historical testing)
    PERFORMANCE_METRICS = {
        'sentiment': {
            'avg_accuracy': 0.62,  # Based on backtesting results
            'best_threshold': 0.70,
            'optimal_hold_days': 3,
        },
        'trend': {
            'avg_accuracy': 0.58,  # CNN-LSTM on test set
        }
    }
    
    def __init__(self, verbose=True):
        """Initialize both predictors and models."""
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.verbose:
            print(f"🚀 Initializing Combined Prediction Pipeline (Device: {self.device})")
        
        # Load sentiment predictor
        try:
            sentiment_path = SENTIMENT_DIR / 'saved_models' / 'ensemble_stock_predictor.pkl'
            self.sentiment_predictor = SentimentPredictor(
                ensemble_path=sentiment_path,
                verbose=False
            )
            if self.verbose:
                print("  ✅ Sentiment predictor loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load sentiment predictor: {e}")
        
        # Load trend predictor
        try:
            self.trend_model, self.scaler, self.yr_mean, self.yr_std = self._load_trend_artifacts()
            if self.verbose:
                print("  ✅ Trend predictor loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load trend predictor: {e}")
        
        if self.verbose:
            print("✨ Pipeline ready for predictions\n")
    
    def _load_trend_artifacts(self):
        """Load trend model weights and scaler."""
        model_path = TREND_DIR / 'saved_model_cnn' / 'dual_head_transformer.pt'
        scaler_path = TREND_DIR / 'saved_model_cnn' / 'scaler.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trend model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        scaler = artifacts['scaler']
        yr_mean = float(artifacts['yr_mean'])
        yr_std = float(artifacts['yr_std'])
        
        # Initialize and load model
        model = CNNLSTMDualHead(
            input_dim=len(FEATURE_COLS),
            hidden_dim=128,
            num_layers=3,
            cnn_channels=64,
            dropout=0.2,
            bidirectional=True,
        ).to(self.device)
        
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        
        return model, scaler, yr_mean, yr_std

    def _normalize_ticker(self, ticker):
        """Normalize user ticker to trend model key format (e.g., HDFCBANK)."""
        if ticker is None:
            return None

        t = str(ticker).strip().upper()
        if not t:
            return None

        # Handle forms like HDFCBANK.NS by mapping value->key from TICKER_MAP.
        if t in TICKER_MAP:
            return t

        reverse_map = {v.upper(): k for k, v in TICKER_MAP.items()}
        return reverse_map.get(t)
    
    def predict_sentiment(self, headline, date_str, ticker=None, body=''):
        """
        Get sentiment-based prediction from headline.
        
        Parameters:
        -----------
        headline : str
            News headline
        date_str : str
            Publication date (YYYY-MM-DD)
        ticker : str
            Stock ticker (optional, for logging)
        body : str
            Article body text (optional)
        
        Returns:
        --------
        dict : {
            'movement': str (UP/DOWN/NEUTRAL),
            'confidence': float (0-1),
            'probabilities': dict,
            'reasoning': str
        }
        """
        try:
            result = self.sentiment_predictor.predict(
                headline=headline,
                date_str=date_str,
                ticker=ticker,
                body=body,
                verbose=False
            )
            
            return {
                'movement': result['predicted_movement'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'reasoning': f"Sentiment analysis: {result['predicted_movement']} "
                           f"({result['confidence']:.1%} confidence)"
            }
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Sentiment prediction failed: {e}")
            return None
    
    def predict_trend(self, date_str, ticker):
        """
        Get trend-based prediction from technical indicators.
        
        Parameters:
        -----------
        date_str : str
            As-of date (YYYY-MM-DD)
        ticker : str
            Stock ticker (e.g., 'HDFCBANK')
        
        Returns:
        --------
        dict : {
            'movement': str (UP/DOWN/NEUTRAL),
            'confidence': float (0-1),
            'pred_next_close': float,
            'last_close': float,
            'probabilities': dict,
            'reasoning': str
        }
        """
        try:
            ticker_key = self._normalize_ticker(ticker)

            # Validate ticker
            if ticker_key not in TICKER_MAP:
                return None
            
            # Download price data
            start_date = (pd.to_datetime(date_str) - timedelta(days=400)).strftime('%Y-%m-%d')
            price_df = download_price_data(TICKER_MAP, start_date=start_date, end_date=date_str)
            
            if price_df.empty:
                return None
            
            # Compute technical features
            full_data = compute_technical_features(price_df)
            
            # Build inference batch
            X, meta = build_inference_batch(full_data, self.scaler, date_str)
            
            if not meta or not X.size(0):
                return None
            
            # Find prediction for specific ticker
            ticker_idx = None
            for i, item in enumerate(meta):
                if item['stock_id'] == ticker_key:
                    ticker_idx = i
                    break
            
            if ticker_idx is None:
                return None
            
            X = X.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                cls_logits, reg_pred = self.trend_model(X)
            
            probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()
            cls_idx = cls_logits.argmax(dim=1).cpu().numpy()
            pred_close = (reg_pred.cpu().numpy() * self.yr_std) + self.yr_mean
            
            direction = TREND_LABELS[int(cls_idx[ticker_idx])]
            confidence = float(probs[ticker_idx][int(cls_idx[ticker_idx])])
            
            meta_item = meta[ticker_idx]
            
            return {
                'movement': direction,
                'confidence': confidence,
                'pred_next_close': round(float(pred_close[ticker_idx]), 2),
                'last_close': meta_item['last_close'],
                'probabilities': {
                    'DOWN': round(float(probs[ticker_idx][0]), 4),
                    'NEUTRAL': round(float(probs[ticker_idx][1]), 4),
                    'UP': round(float(probs[ticker_idx][2]), 4),
                },
                'reasoning': f"Technical trend: {direction} "
                           f"({confidence:.1%} confidence). "
                           f"Expected close: ₹{pred_close[ticker_idx]:.2f}"
            }
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Trend prediction failed: {e}")
            return None
    
    def combine_predictions(self, sentiment_pred, trend_pred, method='weighted'):
        """
        Combine sentiment and trend predictions.
        
        Parameters:
        -----------
        sentiment_pred : dict
            Sentiment model output
        trend_pred : dict
            Trend model output
        method : str
            Combination method:
            - 'weighted': Confidence-weighted average
            - 'majority': Majority vote (requires agreement)
            - 'confidence_threshold': Use prediction with higher confidence
        
        Returns:
        --------
        dict : {
            'movement': str (UP/DOWN/NEUTRAL),
            'confidence': float (0-1),
            'consensus': str (strong/moderate/disagreement),
            'reasoning': str
        }
        """
        if sentiment_pred is None or trend_pred is None:
            # Fallback to available prediction
            if sentiment_pred:
                return {
                    'movement': sentiment_pred['movement'],
                    'confidence': sentiment_pred['confidence'],
                    'consensus': 'sentiment_only',
                    'reasoning': 'Using sentiment only (trend prediction unavailable)'
                }
            elif trend_pred:
                return {
                    'movement': trend_pred['movement'],
                    'confidence': trend_pred['confidence'],
                    'consensus': 'trend_only',
                    'reasoning': 'Using trend only (sentiment prediction unavailable)'
                }
            else:
                return None
        
        sentiment_move = sentiment_pred['movement']
        trend_move = trend_pred['movement']
        sentiment_conf = sentiment_pred['confidence']
        trend_conf = trend_pred['confidence']
        
        # Determine consensus strength
        if sentiment_move == trend_move:
            consensus = 'strong'
        else:
            consensus = 'disagreement'
        
        if method == 'weighted':
            # Weighted average of confidences
            combined_conf = (sentiment_conf * self.DEFAULT_WEIGHTS['sentiment'] +
                           trend_conf * self.DEFAULT_WEIGHTS['trend'])
            
            if sentiment_move == trend_move:
                combined_move = sentiment_move
            else:
                # Use prediction with higher confidence
                combined_move = sentiment_move if sentiment_conf > trend_conf else trend_move
                consensus = 'moderate'
        
        elif method == 'confidence_threshold':
            # Use prediction with higher confidence
            if sentiment_conf > trend_conf:
                combined_move = sentiment_move
                combined_conf = sentiment_conf
            else:
                combined_move = trend_move
                combined_conf = trend_conf
        
        elif method == 'majority':
            # Must agree
            if sentiment_move == trend_move:
                combined_move = sentiment_move
                combined_conf = (sentiment_conf + trend_conf) / 2
            else:
                return None  # No consensus
        
        else:
            raise ValueError(f"Unknown combine method: {method}")
        
        reasoning = self._build_reasoning(
            sentiment_move, sentiment_conf,
            trend_move, trend_conf,
            combined_move, consensus
        )
        
        return {
            'movement': combined_move,
            'confidence': combined_conf,
            'consensus': consensus,
            'reasoning': reasoning
        }
    
    def _build_reasoning(self, sent_move, sent_conf, trend_move, trend_conf, combined_move, consensus):
        """Build human-readable reasoning for the combined prediction."""
        parts = []
        
        if consensus == 'strong':
            parts.append(f"✅ Strong agreement: Both models predict {combined_move}")
            parts.append(f"   - Sentiment: {sent_move} ({sent_conf:.1%})")
            parts.append(f"   - Trend: {trend_move} ({trend_conf:.1%})")
        else:
            parts.append(f"⚠️  Disagreement between models")
            parts.append(f"   - Sentiment: {sent_move} ({sent_conf:.1%})")
            parts.append(f"   - Trend: {trend_move} ({trend_conf:.1%})")
            parts.append(f"   → Selected: {combined_move} (higher confidence)")
        
        return "\n".join(parts)
    
    def get_recommendation(self, combined_pred, sentiment_pred, trend_pred):
        """
        Generate trading recommendation.
        
        Parameters:
        -----------
        combined_pred : dict
            Combined prediction
        sentiment_pred : dict
            Sentiment prediction
        trend_pred : dict
            Trend prediction
        
        Returns:
        --------
        str : Trading recommendation (BUY/HOLD/SELL)
        """
        if combined_pred is None:
            return 'HOLD'
        
        movement = combined_pred['movement']
        confidence = combined_pred['confidence']
        consensus = combined_pred.get('consensus', 'disagreement')
        
        # Decision logic
        if movement == 'UP' and confidence > 0.65 and consensus == 'strong':
            return 'BUY'
        elif movement == 'DOWN' and confidence > 0.65 and consensus == 'strong':
            return 'SELL'
        elif movement == 'UP' and confidence > 0.55:
            return 'BUY'
        elif movement == 'DOWN' and confidence > 0.55:
            return 'SELL'
        else:
            return 'HOLD'
    
    def predict(self, headline, date, ticker, body='', combine_method='weighted', verbose=None):
        """
        Make combined prediction from headline, date, and ticker.
        
        Parameters:
        -----------
        headline : str
            News headline
        date : str
            Date in YYYY-MM-DD format
        ticker : str
            Stock ticker (e.g., 'HDFCBANK', 'SBIN')
        body : str
            Article body (optional)
        combine_method : str
            How to combine predictions: 'weighted', 'confidence_threshold', 'majority'
        verbose : bool
            Print details (uses self.verbose if None)
        
        Returns:
        --------
        dict : {
            'sentiment': {...},
            'trend': {...},
            'combined': {...},
            'recommendation': str,
            'timestamp': str
        }
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"\n📊 Combined Prediction")
            print(f"{'─' * 50}")
            print(f"  Ticker:   {ticker}")
            print(f"  Date:     {date}")
            print(f"  Headline: {headline[:60]}...")
        
        # Get sentiment prediction
        sentiment_pred = self.predict_sentiment(headline, date, ticker, body)
        
        # Get trend prediction
        trend_pred = self.predict_trend(date, ticker)
        
        # Combine predictions
        combined_pred = self.combine_predictions(sentiment_pred, trend_pred, method=combine_method)
        
        # Get recommendation
        recommendation = self.get_recommendation(combined_pred, sentiment_pred, trend_pred)
        
        result = {
            'timestamp': date,
            'ticker': ticker,
            'headline': headline,
            'sentiment': sentiment_pred,
            'trend': trend_pred,
            'combined': combined_pred,
            'recommendation': recommendation,
        }
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def _print_result(self, result):
        """Pretty print prediction result."""
        print(f"\n🎯 SENTIMENT ANALYSIS")
        if result['sentiment']:
            sent = result['sentiment']
            print(f"  Movement:     {sent['movement']}")
            print(f"  Confidence:   {sent['confidence']:.1%}")
            print(f"  Reasoning:    {sent['reasoning']}")
            print(f"  Probabilities: {sent['probabilities']}")
        else:
            print("  ⚠️  Sentiment prediction unavailable")
        
        print(f"\n📈 TREND ANALYSIS")
        if result['trend']:
            trend = result['trend']
            print(f"  Movement:     {trend['movement']}")
            print(f"  Confidence:   {trend['confidence']:.1%}")
            print(f"  Last Close:   ₹{trend['last_close']:.2f}")
            print(f"  Expected:     ₹{trend['pred_next_close']:.2f}")
            print(f"  Probabilities: {trend['probabilities']}")
        else:
            print("  ⚠️  Trend prediction unavailable")
        
        print(f"\n🔗 COMBINED PREDICTION")
        if result['combined']:
            comb = result['combined']
            print(f"  Movement:     {comb['movement']}")
            print(f"  Confidence:   {comb['confidence']:.1%}")
            print(f"  Consensus:    {comb['consensus'].upper()}")
            print(f"\n  Reasoning:")
            for line in comb['reasoning'].split('\n'):
                print(f"  {line}")
        else:
            print("  ⚠️  Combined prediction unavailable")
        
        print(f"\n💡 RECOMMENDATION: {result['recommendation']}")
        print(f"{'─' * 50}\n")
    
    def batch_predict(self, data_list, verbose=None):
        """
        Make predictions for multiple headlines.
        
        Parameters:
        -----------
        data_list : list of dict
            Each dict should have: {'headline': str, 'date': str, 'ticker': str, 'body': str (optional)}
        verbose : bool
            Print details
        
        Returns:
        --------
        list : List of prediction results
        """
        results = []
        for i, data in enumerate(data_list, 1):
            print(f"\n[{i}/{len(data_list)}] Processing...")
            result = self.predict(
                headline=data['headline'],
                date=data['date'],
                ticker=data['ticker'],
                body=data.get('body', ''),
                verbose=verbose or self.verbose
            )
            results.append(result)
        
        return results


def main():
    """Demo: Run predictions on sample data."""
    
    # Initialize pipeline
    pipeline = CombinedPredictionPipeline(verbose=True)
    
    # Sample data
    samples = [
        {
            'headline': 'HDFC Bank Q3 net profit rises 18%',
            'date': '2024-01-15',
            'ticker': 'HDFCBANK',
            'body': 'HDFC Bank reported strong Q3 results with net profit growth of 18% YoY.'
        },
        {
            'headline': 'SBIN faces regulatory challenges',
            'date': '2024-01-16',
            'ticker': 'SBIN',
            'body': 'State Bank of India faces regulatory scrutiny over lending practices.'
        }
    ]
    
    # Make predictions
    results = pipeline.batch_predict(samples)
    
    # Save results to CSV
    output_data = []
    for result in results:
        output_data.append({
            'timestamp': result['timestamp'],
            'ticker': result['ticker'],
            'sentiment_movement': result['sentiment']['movement'] if result['sentiment'] else 'N/A',
            'sentiment_confidence': result['sentiment']['confidence'] if result['sentiment'] else 'N/A',
            'trend_movement': result['trend']['movement'] if result['trend'] else 'N/A',
            'trend_confidence': result['trend']['confidence'] if result['trend'] else 'N/A',
            'combined_movement': result['combined']['movement'] if result['combined'] else 'N/A',
            'combined_confidence': result['combined']['confidence'] if result['combined'] else 'N/A',
            'recommendation': result['recommendation'],
        })
    
    output_df = pd.DataFrame(output_data)
    output_path = ROOT / 'combined_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
