import sys
# sys.path.append("/dsmlp/home-fs04/19/019/riling/scalable_rl_portfolio_management")
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from pathlib import Path
import pandas as pd
import hashlib

market_tics = {
    "S&P 500": "^GSPC",
    "DJIA": "^DJI",
    "NASDAQ": "^IXIC"
}



def get_market_df(start, end, market="S&P 500"):
    assert market in market_tics, f"market `{market}` not avaliable"
    market_tic = market_tics[market]
    df_market = YahooDownloader(start_date = start,
                                end_date = end,
                                ticker_list = [market_tic]).fetch_data()
    
    return df_market

def get_rf_rate(start, end, tic="^IRX"):
    df = YahooDownloader(start_date = start,
                         end_date = end,
                         ticker_list = [tic]).fetch_data()
    df['rf_rate'] = df['close'] / 100.0
    return df[['date', 'rf_rate']]

def short_name_sha256(s: str, length: int = 16) -> str:
    """
    对字符串 s 计算 SHA-256，取前 length 个 hex 字符作为短名。
    默认取16字符（即 64bit），碰撞风险极低，且足够短。
    """
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return h[:length]

def get_data(tics, start, end, market_tic, rf_tic):
    tics_data_file = Path("data") / f"{short_name_sha256('_'.join(tics))}_{start}_{end}.csv"
    if tics_data_file.is_file():
        tics_df = pd.read_csv(tics_data_file) 
    else: 
        tics_df = YahooDownloader(
            start_date = start,
            end_date = end,
            ticker_list = tics
        ).fetch_data()
        tics_df.to_csv(tics_data_file, index=False)
    
    market_data_file = Path("data") / f"{market_tic}_{start}_{end}.csv"
    if market_data_file.is_file():
        market_df = pd.read_csv(market_data_file) 
    else:
        market_df = get_market_df(start, end, market_tic)
        market_df.to_csv(market_data_file, index=False)

    rf_data_file = Path("data") / f"{rf_tic}_{start}_{end}.csv"
    if rf_data_file.is_file():
        rf_df = pd.read_csv(rf_data_file) 
    else:
        rf_df = get_rf_rate(start, end, rf_tic)
        rf_df.to_csv(rf_data_file, index=False)

    return tics_df, market_df, rf_df