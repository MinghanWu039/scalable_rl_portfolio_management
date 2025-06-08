from finrl.meta.preprocessor.yahoodownloader import YahooDownloader # TODO
from pathlib import Path
import pandas as pd

from .helper import file_path

market_tics = {
    "S&P 500": "^GSPC",
    "DJIA": "^DJI",
    "NASDAQ": "^IXIC"
}

def get_market_df(start, end, market="S&P 500", dir="data"):
    assert market in market_tics, f"market `{market}` not avaliable"

    market_tic = market_tics[market]
    market_data_file = Path(dir) / f"{market_tic}_{start}_{end}.csv"
    if market_data_file.is_file():
        market_df = pd.read_csv(market_data_file) 
    else:
        market_tic = market_tics[market]
        df_market = YahooDownloader(start_date = start,
                                    end_date = end,
                                    ticker_list = [market_tic]).fetch_data()
        market_data_file.mkdir(parents=True, exist_ok=True)
        market_df.to_csv(market_data_file, index=False)
    return df_market

def get_rf_rate(start, end, tic="^IRX", dir="data"):
    rf_data_file = Path(dir) / f"{tic}_{start}_{end}.csv"

    if rf_data_file.is_file():
        df = pd.read_csv(rf_data_file) 
    else:
        df = YahooDownloader(start_date = start,
                            end_date = end,
                            ticker_list = [tic]).fetch_data()
        df['rf_rate'] = df['close'] / 100.0
        rf_data_file.mkdir(parents=True, exist_ok=True)
        df.to_csv(rf_data_file, index=False)
    return df[['date', 'rf_rate']]


def get_data(tics, start, end, dir="data"):
    tics_data_file = file_path(dir, tics, start, end)
    if tics_data_file.is_file():
        tics_df = pd.read_csv(tics_data_file) 
    else: 
        tics_df = YahooDownloader(
            start_date = start,
            end_date = end,
            ticker_list = tics
        ).fetch_data()
        tics_df.to_csv(tics_data_file, index=False)

    return tics_df