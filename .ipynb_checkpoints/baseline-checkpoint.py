import argparse
import sys
sys.path.append('../../FinRL-dev')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import yaml
# matplotlib.use('Agg')
import datetime

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
# sys.path.append("../FinRL")

def load_yaml(configpath):
    with open(configpath, 'r') as f:
        return yaml.safe_load(f)

def preprocess(df):
    fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list = INDICATORS,
                use_vix=True,
                use_turbulence=True,
                user_defined_feature = False)

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    processed_full = processed_full.fillna(0)
    return processed_full

def train(algo, env_train, total_timesteps=50000):
    agent = DRLAgent(env = env_train)
    model = agent.get_model(algo)

    # set up logger
    tmp_path = RESULTS_DIR + f'/{algo}'
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model.set_logger(new_logger)

    # Train the model
    trained_model = agent.train_model(model=model, tb_log_name=algo, total_timesteps=total_timesteps)

    return trained_model

def backtesting(env_test, model):
    print("==============Get Backtest Stats===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, 
        environment = env_test)

    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

    stats = backtest_stats(baseline_df, value_col_name = 'close')


if __name__ == "__main__":
    print(sys.executable)

    parser = argparse.ArgumentParser(description='FinRL Baseline')
    parser.add_argument('--config', type=str, default='config/baseline_config.yaml', help='Path to the config file')
    parser.add_argument('--algo', type=str, default='sac', help='Algorithm to use (ppo, a2c, ddpg, sac, td3)')
    args = parser.parse_args()

    config = load_yaml(args.config)

    from finrl import config_tickers
    import os
    from finrl.main import check_and_make_directories
    from finrl.config import (
        DATA_SAVE_DIR,
        TRAINED_MODEL_DIR,
        TENSORBOARD_LOG_DIR,
        RESULTS_DIR,
        INDICATORS,
    )
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    TRAIN_START_DATE = config['train_start_date']
    TRAIN_END_DATE = config['train_end_date']
    TRADE_START_DATE = config['trade_start_date']
    TRADE_END_DATE = config['trade_end_date']

    df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
    print('Data downloaded')
    
    processed_full = preprocess(df)
    print('Data processed')

    train_df = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
    trade_df = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
    print('train samples:', len(train_df))
    print('trade samples:', len(trade_df))

    stock_dimension = len(train_df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": np.float(config['hmax']),
        "initial_amount": np.float(config['initial_amount']),
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": np.float(config['reward_scaling']),
    }


    # training
    print('Begin training...')
    e_train_gym = StockTradingEnv(df = train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    trained_model = train(args.algo, env_train, total_timesteps=config['timesteps'])

    # backtesting
    print('Begin backtesting...')
    e_trade_gym = StockTradingEnv(df = trade_df, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
    backtesting(e_trade_gym, trained_model)



