"""
Stock Data Alignment Module
============================
Aligns news headlines with stock price data from Yahoo Finance.

Functions:
    - extracting_prices(): Downloads historical price data using yfinance
    - aligning_csv(): Aligns news headlines with trading dates and calculates returns
"""

import pandas as pd
from datetime import time, timedelta, date, datetime
import yfinance as yf
from pathlib import Path
import logging
from .logger import setup_logger

logger = setup_logger("data_aligner")

stock_dict = pd.read_excel("stock_dict.xlsx")

def extracting_prices(stock_name):
    """
    Download 3-year historical stock price data from Yahoo Finance.
    
    Parameters
    ----------
    stock_name : str
        Company name (must exist in stock_dict.xlsx)
        
    Returns
    -------
    None
        Saves data to stocks_data/raw/yahoo/{stock_name}_yahoo.csv
        
    Raises
    ------
    Exception
        If stock not found in stock_dict.xlsx or download fails
        
    Examples
    --------
    >>> extracting_prices("idfc")  # Downloads IDFC stock data (3 years)
    """
    logger.info(f"Downloading stock data for: {stock_name}")
    try:
        stock_tick = stock_dict.loc[stock_dict["Company Name"] == stock_name, "Stock Name"].values[0]
        logger.info(f"Stock ticker: {stock_tick}")
        data = yf.download(stock_tick, period='10y', multi_level_index=False)
        data.to_csv(f"stocks_data/raw/yf_prices/{stock_name}_yahoo.csv")
        logger.info(f"Saved price data to stocks_data/raw/yahoo/{stock_name}_yahoo.csv ({len(data)} records)")
    except Exception as e:
        logger.error(f"Error extracting prices for {stock_name}: {e}", exc_info=True)

price_base_path = "stocks_data/raw/yf_prices"
news_base_path = "stocks_data/raw/news"


def aligning_csv(stock_name, price_base_path=price_base_path, news_base_path=news_base_path):
    """
    Align news headlines with stock trading dates and calculate price returns.
    
    Matches each news headline with its corresponding trading date, then calculates
    1-day, 2-day, and 3-day price returns. News released after market close (15:30)
    is mapped to the next trading day.
    
    Parameters
    ----------
    stock_name : str
        Stock identifier (used to find files: {stock_name}.csv and {stock_name}_yahoo.csv)
    price_base_path : str, optional
        Path to directory containing Yahoo Finance price data (default: "stocks_data/raw/yf_prices")
    news_base_path : str, optional
        Path to directory containing news CSV files (default: "stocks_data/raw/news")
        
    Returns
    -------
    None
        Saves aligned data to stocks_data/aligned/{stock_name}_aligned.csv
        
    Output Columns
    -------
    - news_id: Unique identifier for news item
    - headline: News headline text
    - news_time: Timestamp when news was published
    - event_date: Trading date associated with the news
    - close_T: Stock closing price on event_date
    - ret_1d: 1-day return (%)
    - ret_2d: 2-day return (%)
    - ret_3d: 3-day return (%)
    
    Examples
    --------
    >>> aligning_csv('idfc')  # Aligns idfc news with price data
    """
    logger.info(f"Starting alignment for stock: {stock_name}")
    
    price_path = price_base_path+"/"+stock_name+"_yahoo.csv"
    news_path = news_base_path+"/"+stock_name+".csv"
    
    try:
        price_df = pd.read_csv(price_path)
        news_df = pd.read_csv(news_path)
        logger.info(f"Loaded {len(price_df)} price records and {len(news_df)} news records")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return

    # ======================
    # DATETIME PROCESSING
    # ======================
    news_df["news_datetime"] = pd.to_datetime(news_df["news_datetime"])
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    # normalize price date
    price_df["trade_date"] = price_df["Date"].dt.date

    # set index for fast lookup
    price_df.set_index("trade_date", inplace=True)

    # set of valid trading days
    trading_days = set(price_df.index)
    
    # Get max trading date for boundary checking
    max_trading_date = max(trading_days)


    # ======================
    # MARKET CLOSE TIME
    # ======================
    market_close = time(15, 30) # time - 15:30


    # ======================
    # FUNCTION: MAP NEWS → TRADING DAY
    # ======================

    def get_event_date(news_dt):
        """
        Map news datetime to corresponding trading date.
        
        If news is released after market close (15:30), it's mapped to the next day.
        Then, the function finds the next valid trading day (market must be open).
        
        Parameters
        ----------
        news_dt : datetime
            Timestamp when the news was published
            
        Returns
        -------
        date
            The trading date to associate with this news
            
        Raises
        ------
        OverflowError
            If no trading day can be found within reasonable bounds
        """
        news_date = news_dt.date()

        # if news released after market close → shift to next day
        if news_dt.time() > market_close:
            news_date += timedelta(days=1)

        # move forward until valid trading day, with safety limit
        max_iterations = 30  # Prevent infinite loops (covers weekends, holidays)
        iterations = 0
        
        while news_date not in trading_days and iterations < max_iterations:
            if news_date > max_trading_date:
                # News is beyond available price data
                raise OverflowError(f"News date {news_date} exceeds latest trading date {max_trading_date}")
            news_date += timedelta(days=1)
            iterations += 1
        
        if iterations >= max_iterations:
            raise OverflowError(f"Could not find trading day for news date {news_dt.date()} within {max_iterations} days")

        return news_date


    # ======================
    # FUNCTION: GET FUTURE PRICE
    # ======================

    def get_future_price(date, offset):
        """
        Get stock closing price at future trading date.
        
        Given a starting trading date, finds the N-th trading day in the future
        and returns the closing price on that date.
        
        Parameters
        ----------
        date : date
            Starting trading date
        offset : int
            Number of trading days to move forward (1 = next trading day, 2 = day after, etc.)
            
        Returns
        -------
        float
            Stock closing price on the target trading date
        """
        d = date
        count = 0

        while count < offset:
            d += timedelta(days=1)
            if d in trading_days:
                count += 1

        return price_df.loc[d]["Close"]


    # ======================
    # ALIGNMENT LOOP
    # ======================

    records = []
    missed_urls = []
    count = 0
    for _, row in news_df.iterrows():

        news_dt = row["news_datetime"]
        
        # Skip if news_datetime is missing
        if pd.isna(news_dt):
            logger.debug(f"Skipping entry: news_datetime is None - {row['link']}")
            missed_urls.append({"url": row["link"], "error": "missing news_datetime"})
            continue
        
        try:
            event_date = get_event_date(news_dt) # getting the price date of yahoo finance
        except (ValueError, OverflowError) as e:
            logger.warning(f"Skipping entry: {str(e)[:50]} - {row['headline'][:50]}")
            missed_urls.append({"url" : row["link"],
                                "error" : str(e)})
            continue
        if event_date not in price_df.index:
            logger.debug(f"Skipping entry: News date not in price df - {row['headline'][:50]}")
            missed_urls.append({"url" : row["link"],
                               "error" : "news date not matched with price date"})
            continue

        # Skip if not enough price data after event_date for 3-day return calculation
        if max_trading_date - event_date < timedelta(days=3):
            logger.debug(f"Skipping entry: Not enough price data after event_date - {row['headline'][:50]}")
            missed_urls.append({"url" : row["link"],
                               "error" : "insufficient price data for 3-day returns"})
            continue 


        close_T = price_df.loc[event_date]["Close"]

        # future prices
        close_T1 = get_future_price(event_date, 1)
        close_T2 = get_future_price(event_date, 2)
        close_T3 = get_future_price(event_date, 3)

        # returns
        r1 = (close_T1 - close_T) / close_T
        r2 = (close_T2 - close_T) / close_T
        r3 = (close_T3 - close_T) / close_T

        records.append({
            "news_id": row["news_id"],
            "headline": row["headline"],
            "news_time": news_dt,
            "event_date": event_date,
            "close_T": close_T,
            "ret_1d": r1,
            "ret_2d": r2,
            "ret_3d": r3
        })
        count += 1

    # ======================
    # FINAL DATAFRAME
    # ======================

    aligned_df = pd.DataFrame(records)

    # print(aligned_df.head)

    # save
    aligned_df.to_csv(f"stocks_data/aligned/{stock_name}_aligned.csv", index=False)
    logger.info(f"Alignment complete: {len(records)} aligned records, {len(missed_urls)} missed")
    logger.info(f"Saved to stocks_data/aligned/{stock_name}_aligned.csv")
    if missed_urls:
        logger.warning(f"Missed {len(missed_urls)} entries - check logs for details")


def aligning_csv_1(stock_name, price_base_path=price_base_path, news_base_path=news_base_path):
    logger.info(f"Starting alignment for stock: {stock_name}")
    
    price_path = price_base_path + "/" + stock_name + "_yahoo.csv"
    news_path = news_base_path + "/" + stock_name + ".csv"
    
    try:
        price_df = pd.read_csv(price_path)
        news_df = pd.read_csv(news_path)
        logger.info(f"Loaded {len(price_df)} price records and {len(news_df)} news records")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return

    # ======================
    # DATETIME PROCESSING
    # ======================

    # FIX 1: Strip timezone info to avoid comparison issues with naive datetimes
    news_df["news_datetime"] = pd.to_datetime(news_df["news_datetime"], errors="coerce")
    if news_df["news_datetime"].dt.tz is not None:
        news_df["news_datetime"] = news_df["news_datetime"].dt.tz_localize(None)

    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df["trade_date"] = price_df["Date"].dt.date

    # FIX 2: Normalize column names (MoneyControl may export lowercase)
    price_df.columns = [c.strip().lower().replace(" ", "_") for c in price_df.columns]
    # Find the close column flexibly
    close_col = next((c for c in price_df.columns if "close" in c), None)
    if close_col is None:
        logger.error(f"No 'close' column found. Columns are: {price_df.columns.tolist()}")
        return
    logger.info(f"Using '{close_col}' as close price column")

    # FIX 3: Remove commas from price strings if present (MoneyControl quirk)
    price_df[close_col] = pd.to_numeric(
        price_df[close_col].astype(str).str.replace(",", ""), errors="coerce"
    )

    price_df.set_index("trade_date", inplace=True)
    trading_days = sorted(price_df.index)  # sorted list for offset lookups
    trading_days_set = set(trading_days)
    max_trading_date = max(trading_days)

    logger.info(f"Price data: {min(trading_days)} to {max_trading_date} ({len(trading_days)} trading days)")
    logger.info(f"News data range: {news_df['news_datetime'].min()} to {news_df['news_datetime'].max()}")

    market_close = time(15, 30)

    # ======================
    # FUNCTION: MAP NEWS → TRADING DAY
    # ======================
    def get_event_date(news_dt):
        news_date = news_dt.date()

        if news_dt.time() > market_close:
            news_date += timedelta(days=1)

        max_iterations = 10
        iterations = 0
        while news_date not in trading_days_set:
            if news_date > max_trading_date:
                raise OverflowError(
                    f"News date {news_date} exceeds latest trading date {max_trading_date}"
                )
            news_date += timedelta(days=1)
            iterations += 1
            if iterations >= max_iterations:
                raise OverflowError(
                    f"Could not find trading day within {max_iterations} days of {news_dt.date()}"
                )
        return news_date

    # ======================
    # FUNCTION: GET FUTURE PRICE (FIX 4: uses sorted list, bounded)
    # ======================
    def get_future_price(event_date, offset):
        try:
            idx = trading_days.index(event_date)
        except ValueError:
            raise ValueError(f"{event_date} not in trading days")
        
        target_idx = idx + offset
        if target_idx >= len(trading_days):
            raise OverflowError(
                f"Not enough future trading data after {event_date} for offset {offset}"
            )
        target_date = trading_days[target_idx]
        return price_df.loc[target_date][close_col]

    # ======================
    # ALIGNMENT LOOP
    # ======================
    records = []
    missed_urls = []
    skip_reasons = {}  # FIX 5: track skip reasons for diagnosis

    for _, row in news_df.iterrows():
        news_dt = row["news_datetime"]

        if pd.isna(news_dt):
            reason = "missing news_datetime"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            missed_urls.append({"url": row.get("link", ""), "error": reason})
            continue

        try:
            event_date = get_event_date(news_dt)
        except (ValueError, OverflowError) as e:
            reason = str(e)[:60]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            missed_urls.append({"url": row.get("link", ""), "error": str(e)})
            continue

        if event_date not in trading_days_set:
            reason = "event_date not in price data"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            missed_urls.append({"url": row.get("link", ""), "error": reason})
            continue

        # FIX 6: Check using trading day index, not calendar days
        try:
            event_idx = trading_days.index(event_date)
        except ValueError:
            continue
        
        if event_idx + 3 >= len(trading_days):
            reason = "insufficient future trading days for 3-day return"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            missed_urls.append({"url": row.get("link", ""), "error": reason})
            continue

        try:
            close_T = price_df.loc[event_date][close_col]
            close_T1 = get_future_price(event_date, 1)
            close_T2 = get_future_price(event_date, 2)
            close_T3 = get_future_price(event_date, 3)
        except (KeyError, OverflowError, ValueError) as e:
            reason = f"price lookup error: {str(e)[:40]}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            missed_urls.append({"url": row.get("link", ""), "error": str(e)})
            continue

        if any(pd.isna(v) for v in [close_T, close_T1, close_T2, close_T3]):
            reason = "NaN in price data"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            missed_urls.append({"url": row.get("link", ""), "error": reason})
            continue

        r1 = (close_T1 - close_T) / close_T
        r2 = (close_T2 - close_T) / close_T
        r3 = (close_T3 - close_T) / close_T

        records.append({
            "news_id": row["news_id"],
            "headline": row["headline"],
            "news_time": news_dt,
            "event_date": event_date,
            "close_T": close_T,
            "ret_1d": r1,
            "ret_2d": r2,
            "ret_3d": r3,
        })

    # ======================
    # DIAGNOSIS SUMMARY
    # ======================
    logger.info(f"Alignment complete: {len(records)} aligned, {len(missed_urls)} skipped")
    logger.info("Skip reason breakdown:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  {count:4d} rows — {reason}")

    aligned_df = pd.DataFrame(records)
    aligned_df.to_csv(f"stocks_data/aligned/{stock_name}_aligned.csv", index=False)
    logger.info(f"Saved to stocks_data/aligned/{stock_name}_aligned.csv")