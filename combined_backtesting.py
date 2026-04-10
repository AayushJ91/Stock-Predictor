"""
Combined backtesting pipeline.

Workflow:
1. Load a small sample from data/processed/all_banking_news.csv
2. Use headline/date/ticker with CombinedPredictionPipeline
3. Simulate trading with initial capital and low confidence threshold
4. Verify trades against next-day actual price movement from yfinance
5. Output final net result (PnL, final capital, return %)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from combined_prediction_pipeline import CombinedPredictionPipeline


ROOT = Path(__file__).parent
DEFAULT_NEWS_PATH = ROOT / "data" / "processed" / "all_banking_news.csv"
DEFAULT_OUT_PATH = ROOT / "results" / "reports" / "combined_backtest_results.csv"
DEFAULT_PLOTS_DIR = ROOT / "results" / "combined_backtesting"


class CombinedBacktestingPipeline:
    def __init__(self, news_path: Path = DEFAULT_NEWS_PATH, verbose: bool = True):
        self.news_path = Path(news_path)
        self.verbose = verbose
        self.predictor = CombinedPredictionPipeline(verbose=False)

        if not self.news_path.exists():
            raise FileNotFoundError(f"News file not found: {self.news_path}")

    @staticmethod
    def _to_yf_ticker(ticker: str) -> str:
        t = str(ticker).strip().upper()
        if not t:
            return t
        if t.endswith(".NS"):
            return t
        return f"{t}.NS"

    def load_news_sample(
        self,
        n_samples: int = 25,
        start_date: str | None = None,
        end_date: str | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.news_path)

        required_cols = {"headline", "date", "ticker"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        df = df.dropna(subset=["headline", "date", "ticker"]).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if df.empty:
            raise ValueError("No rows available after date filters.")

        n = min(n_samples, len(df))
        sample_df = df.sample(n=n, random_state=random_state).sort_values("date").reset_index(drop=True)
        return sample_df

    def _get_actual_next_day_move(self, ticker: str, date_str: str) -> Dict[str, Any]:
        yf_ticker = self._to_yf_ticker(ticker)
        start = pd.to_datetime(date_str)
        end = start + pd.Timedelta(days=7)

        price_df = yf.download(
            yf_ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if price_df is None or price_df.empty:
            return {
                "actual_movement": None,
                "actual_return_pct": None,
                "entry_price": None,
                "exit_price": None,
            }

        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)

        if "Open" not in price_df.columns or "Close" not in price_df.columns:
            return {
                "actual_movement": None,
                "actual_return_pct": None,
                "entry_price": None,
                "exit_price": None,
            }

        price_df = price_df.dropna(subset=["Open", "Close"])
        if len(price_df) < 2:
            return {
                "actual_movement": None,
                "actual_return_pct": None,
                "entry_price": None,
                "exit_price": None,
            }

        entry_price = float(price_df.iloc[0]["Open"])
        exit_price = float(price_df.iloc[1]["Close"])
        ret = (exit_price - entry_price) / entry_price
        ret_pct = ret * 100.0

        if ret > 0.01:
            movement = "UP"
        elif ret < -0.01:
            movement = "DOWN"
        else:
            movement = "NEUTRAL"

        return {
            "actual_movement": movement,
            "actual_return_pct": round(float(ret_pct), 4),
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
        }

    def _predict_row(self, row: pd.Series, combine_method: str) -> Dict[str, Any]:
        headline = str(row["headline"])
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        ticker = str(row["ticker"])
        body = str(row.get("text", ""))

        pred = self.predictor.predict(
            headline=headline,
            date=date_str,
            ticker=ticker,
            body=body,
            combine_method=combine_method,
            verbose=False,
        )

        sentiment = pred.get("sentiment") or {}
        trend = pred.get("trend") or {}
        combined = pred.get("combined") or {}

        return {
            "input_date": date_str,
            "input_ticker": ticker,
            "input_headline": headline,
            "sentiment_movement": sentiment.get("movement"),
            "sentiment_confidence": sentiment.get("confidence"),
            "trend_movement": trend.get("movement"),
            "trend_confidence": trend.get("confidence"),
            "trend_last_close": trend.get("last_close"),
            "trend_pred_next_close": trend.get("pred_next_close"),
            "combined_movement": combined.get("movement"),
            "combined_confidence": combined.get("confidence"),
            "combined_consensus": combined.get("consensus"),
            "recommendation": pred.get("recommendation"),
        }

    def run(
        self,
        n_samples: int = 25,
        start_date: str | None = None,
        end_date: str | None = None,
        combine_method: str = "weighted",
        random_state: int = 42,
        initial_capital: float = 100000.0,
        decision_threshold: float = 0.45,
        position_size: float = 0.2,
    ) -> pd.DataFrame:
        if position_size <= 0 or position_size > 1:
            raise ValueError("position_size must be in (0, 1].")

        sample_df = self.load_news_sample(
            n_samples=n_samples,
            start_date=start_date,
            end_date=end_date,
            random_state=random_state,
        )

        cash = float(initial_capital)
        rows: List[Dict[str, Any]] = []

        for i, row in sample_df.iterrows():
            if self.verbose:
                d = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                print(f"[{i + 1}/{len(sample_df)}] Predicting {row['ticker']} on {d}")

            pred_row = self._predict_row(row, combine_method=combine_method)
            actual = self._get_actual_next_day_move(pred_row["input_ticker"], pred_row["input_date"])

            conf = float(pred_row.get("combined_confidence") or 0.0)
            move = str(pred_row.get("combined_movement") or "")

            decision = "HOLD"
            traded_amount = 0.0
            pnl = 0.0
            decision_correct = None

            # Lower threshold by default so backtest takes more signals.
            if conf >= decision_threshold and move in {"UP", "DOWN"} and actual["actual_return_pct"] is not None:
                decision = "BUY" if move == "UP" else "SELL"
                traded_amount = cash * position_size
                realized_ret = float(actual["actual_return_pct"]) / 100.0

                if decision == "BUY":
                    pnl = traded_amount * realized_ret
                else:
                    # SELL treated as a short over one period.
                    pnl = traded_amount * (-realized_ret)

                cash += pnl

                if actual["actual_movement"] in {"UP", "DOWN"}:
                    decision_correct = (
                        (decision == "BUY" and actual["actual_movement"] == "UP")
                        or (decision == "SELL" and actual["actual_movement"] == "DOWN")
                    )

            pred_row.update(
                {
                    "decision": decision,
                    "decision_threshold": decision_threshold,
                    "traded_amount": round(traded_amount, 2),
                    "pnl": round(pnl, 2),
                    "cash_after": round(cash, 2),
                    "actual_movement": actual["actual_movement"],
                    "actual_return_pct": actual["actual_return_pct"],
                    "entry_price": actual["entry_price"],
                    "exit_price": actual["exit_price"],
                    "decision_correct": decision_correct,
                }
            )

            rows.append(pred_row)

        out = pd.DataFrame(rows)
        out.attrs["initial_capital"] = float(initial_capital)
        out.attrs["final_capital"] = float(cash)
        return out

    @staticmethod
    def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
        if results_df.empty:
            return pd.DataFrame()

        traded = results_df[results_df["decision"].isin(["BUY", "SELL"])].copy()
        wins = traded[traded["decision_correct"] == True]

        initial_capital = float(results_df.attrs.get("initial_capital", 0.0))
        final_capital = float(results_df.attrs.get("final_capital", initial_capital))
        total_return_pct = ((final_capital - initial_capital) / initial_capital * 100.0) if initial_capital else 0.0

        summary = {
            "rows": len(results_df),
            "pred_buy_count": int((results_df["recommendation"] == "BUY").sum()),
            "pred_sell_count": int((results_df["recommendation"] == "SELL").sum()),
            "pred_hold_count": int((results_df["recommendation"] == "HOLD").sum()),
            "decision_buy_count": int((results_df["decision"] == "BUY").sum()),
            "decision_sell_count": int((results_df["decision"] == "SELL").sum()),
            "decision_hold_count": int((results_df["decision"] == "HOLD").sum()),
            "avg_combined_conf": float(pd.to_numeric(results_df["combined_confidence"], errors="coerce").mean()),
            "trend_available_count": int(results_df["trend_movement"].notna().sum()),
            "trades_taken": int(len(traded)),
            "win_rate": float((len(wins) / len(traded)) if len(traded) else 0.0),
            "total_pnl": float(pd.to_numeric(results_df["pnl"], errors="coerce").fillna(0).sum()),
            "initial_capital": round(initial_capital, 2),
            "final_capital": round(final_capital, 2),
            "total_return_pct": round(total_return_pct, 4),
        }
        return pd.DataFrame([summary])

    @staticmethod
    def save_plots(results_df: pd.DataFrame, plot_dir: Path) -> List[Path]:
        """Save backtesting plots and return created file paths."""
        if results_df.empty:
            return []

        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        created: List[Path] = []

        plot_df = results_df.copy()
        plot_df["input_date"] = pd.to_datetime(plot_df["input_date"], errors="coerce")
        plot_df = plot_df.sort_values("input_date").reset_index(drop=True)

        # 1) Equity curve
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(plot_df.index + 1, plot_df["cash_after"], marker="o", linewidth=1.7)
        ax.set_title("Backtest Equity Curve")
        ax.set_xlabel("Trade Step")
        ax.set_ylabel("Cash")
        eq_path = plot_dir / "equity_curve.png"
        fig.tight_layout()
        fig.savefig(eq_path, dpi=140)
        plt.close(fig)
        created.append(eq_path)

        # 2) Decision distribution
        fig, ax = plt.subplots(figsize=(7, 4.8))
        decision_counts = plot_df["decision"].fillna("UNKNOWN").value_counts()
        sns.barplot(x=decision_counts.index, y=decision_counts.values, ax=ax)
        ax.set_title("Decision Distribution")
        ax.set_xlabel("Decision")
        ax.set_ylabel("Count")
        dec_path = plot_dir / "decision_distribution.png"
        fig.tight_layout()
        fig.savefig(dec_path, dpi=140)
        plt.close(fig)
        created.append(dec_path)

        # 3) PnL distribution
        fig, ax = plt.subplots(figsize=(8, 4.8))
        pnl_series = pd.to_numeric(plot_df["pnl"], errors="coerce").fillna(0)
        sns.histplot(pnl_series, bins=20, kde=True, ax=ax)
        ax.set_title("PnL Distribution Per Step")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Frequency")
        pnl_path = plot_dir / "pnl_distribution.png"
        fig.tight_layout()
        fig.savefig(pnl_path, dpi=140)
        plt.close(fig)
        created.append(pnl_path)

        # 4) Confidence vs actual return scatter
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        conf = pd.to_numeric(plot_df["combined_confidence"], errors="coerce")
        ret = pd.to_numeric(plot_df["actual_return_pct"], errors="coerce")
        scatter_df = pd.DataFrame({"combined_confidence": conf, "actual_return_pct": ret}).dropna()
        if not scatter_df.empty:
            sns.scatterplot(
                data=scatter_df,
                x="combined_confidence",
                y="actual_return_pct",
                ax=ax,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Confidence vs Actual Next-Day Return")
        ax.set_xlabel("Combined Confidence")
        ax.set_ylabel("Actual Return %")
        sc_path = plot_dir / "confidence_vs_actual_return.png"
        fig.tight_layout()
        fig.savefig(sc_path, dpi=140)
        plt.close(fig)
        created.append(sc_path)

        return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest sampled rows from all_banking_news.csv with capital simulation")
    parser.add_argument("--news-path", type=str, default=str(DEFAULT_NEWS_PATH), help="Path to all_banking_news.csv")
    parser.add_argument("--samples", type=int, default=25, help="How many rows to sample")
    parser.add_argument("--start-date", type=str, default=None, help="Optional filter start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Optional filter end date YYYY-MM-DD")
    parser.add_argument(
        "--combine-method",
        type=str,
        default="weighted",
        choices=["weighted", "confidence_threshold", "majority"],
        help="How to combine sentiment/trend predictions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT_PATH), help="Output CSV path")
    parser.add_argument("--plots-dir", type=str, default=str(DEFAULT_PLOTS_DIR), help="Directory to save plots")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting capital")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Decision threshold on combined confidence (lower default)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.2,
        help="Fraction of current cash used per trade",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = CombinedBacktestingPipeline(news_path=Path(args.news_path), verbose=True)

    results = engine.run(
        n_samples=args.samples,
        start_date=args.start_date,
        end_date=args.end_date,
        combine_method=args.combine_method,
        random_state=args.seed,
        initial_capital=args.initial_capital,
        decision_threshold=args.threshold,
        position_size=args.position_size,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    plot_paths = engine.save_plots(results, Path(args.plots_dir))

    summary_df = engine.summarize(results)

    print("\nBacktest complete.")
    print(f"Results saved to: {out_path}")
    print("\nSample rows:")
    print(results.head())

    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df)

        s = summary_df.iloc[0]
        print("\nNet Result:")
        print(f"Initial Capital: {s['initial_capital']:.2f}")
        print(f"Final Capital:   {s['final_capital']:.2f}")
        print(f"Net PnL:         {s['total_pnl']:.2f}")
        print(f"Return (%):      {s['total_return_pct']:.4f}")

    if plot_paths:
        print("\nSaved plots:")
        for p in plot_paths:
            print(f"- {p}")


if __name__ == "__main__":
    main()
