import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

ROOT   = Path(__file__).parent.parent
MODELS = ROOT / 'models' / 'saved_models'
PROC   = ROOT / 'data' / 'processed'


class StockMovementPredictor:
    """
    Wraps the trained ensemble for single or batch prediction.

    Usage
    -----
    predictor = StockMovementPredictor()
    result = predictor.predict('HDFC Bank Q3 net profit rises 18%', '2024-01-15', 'HDFCBANK')
    # → {'predicted_movement': 'UP', 'confidence': 0.74, 'probabilities': {...}}
    """

    BANKING_BOOSTS = {
        'npa': -1.5, 'fraud': -2.0, 'default': -1.5, 'downgrade': -1.8,
        'stressed': -1.2, 'recovery': +1.5, 'profit': +1.8, 'dividend': +1.5,
        'acquisition': +0.8, 'merger': +0.8, 'rally': +1.5, 'surge': +1.5,
        'crash': -2.0, 'collapse': -2.0, 'penalty': -1.5, 'fine': -1.0,
        'upgrade': +1.5, 'growth': +1.2, 'record': +1.0
    }

    def __init__(self, ensemble_path=None, verbose=True):
        if ensemble_path is None:
            ensemble_path = MODELS / 'ensemble_stock_predictor.pkl'

        with open(ensemble_path, 'rb') as f:
            bundle = pickle.load(f)

        self.rf            = bundle['rf']
        self.gb            = bundle['gb']
        self.meta_lr       = bundle['meta_lr']
        self.tfidf         = bundle['tfidf']
        self.scaler        = bundle['scaler']
        self.le            = bundle['le']
        self.tabular_cols  = bundle['tabular_cols']

        self.sia = SentimentIntensityAnalyzer()
        self.sia.lexicon.update(self.BANKING_BOOSTS)

        if verbose:
            print(f'✅ StockMovementPredictor loaded — classes: {list(self.le.classes_)}')

    def _build_row(self, headline, date_str, body=''):
        date = pd.to_datetime(date_str)
        vs   = self.sia.polarity_scores(headline)
        text = (headline + ' ' + body).strip()[:1500]
        return {
            'combined_text_512': text,
            'headline_len'     : len(headline.split()),
            'vader_pos'        : vs['pos'],
            'vader_neg'        : vs['neg'],
            'vader_neu'        : vs['neu'],
            'vader_compound'   : vs['compound'],
            'day_of_week'      : date.dayofweek,
            'month'            : date.month,
            'quarter'          : date.quarter,
            'is_monday'        : int(date.dayofweek == 0),
            'is_friday'        : int(date.dayofweek == 4),
            'is_month_end'     : int(date.is_month_end),
            'is_qtr_end'       : int(date.is_quarter_end),
        }

    def predict(self, headline, date_str, ticker=None, body='', verbose=True):
        import scipy.sparse as sp
        row = self._build_row(headline, date_str, body)

        X_tfidf = self.tfidf.transform([row['combined_text_512']])
        X_tab   = self.scaler.transform(
            pd.DataFrame([row])[self.tabular_cols].values
        )
        X = sp.hstack([X_tfidf, sp.csr_matrix(X_tab)])

        meta = np.hstack([
            self.rf.predict_proba(X),
            self.gb.predict_proba(X.toarray())
        ])
        proba     = self.meta_lr.predict_proba(meta)[0]
        pred_idx  = proba.argmax()
        pred_label = self.le.classes_[pred_idx]
        confidence = float(proba[pred_idx])

        result = {
            'predicted_movement': pred_label,
            'confidence'        : confidence,
            'probabilities'     : dict(zip(self.le.classes_, proba.round(4).tolist()))
        }
        if verbose:
            print(f'[{ticker or "?"}]  {pred_label}  ({confidence:.1%})  |  {result["probabilities"]}')
        return result

    def predict_batch(self, df, headline_col='headline', date_col='date',
                      ticker_col='ticker', body_col=None):
        results = []
        for _, row in df.iterrows():
            body = str(row[body_col]) if body_col and body_col in row else ''
            res  = self.predict(str(row[headline_col]), str(row[date_col]),
                                ticker=row.get(ticker_col), body=body, verbose=False)
            results.append(res)
        return pd.DataFrame(results)
