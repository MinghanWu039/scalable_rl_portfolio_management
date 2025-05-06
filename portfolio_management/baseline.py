import torch
import pandas as pd
import numpy as np
import yaml
import sys
import argparse
import itertools
from datetime import datetime
import inspect
import hashlib
import importlib
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append("/home/miw039/scalable_rl_portfolio_management/FinRL-dev")
import finrl.meta.env_portfolio_optimization.env_portfolio_optimization
importlib.reload(finrl.meta.env_portfolio_optimization.env_portfolio_optimization)
# import finrl.agents.stablebaselines3.models
# importlib.reload(finrl.agents.stablebaselines3.models)

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from finrl.config import INDICATORS
from finrl import config_tickers
from tic_config import tics_grouped

# print(inspect.getfile(PortfolioOptimizationEnv))

def short_name_sha256(s: str, length: int = 16) -> str:
    """
    Compute the SHA-256 hash of the string s and take the first length hex characters as a short name.
    By default, 16 characters (i.e., 64 bits) are taken, which has a very low collision risk and is sufficiently short.
    """
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return h[:length]

def load_yaml(configpath):
    with open(configpath, 'r') as f:
        return yaml.safe_load(f)

def preprocess(df):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list = INDICATORS,
        use_vix=False,
        use_turbulence=False,
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

def backtesting(env_test, model):
    print("==============Get Backtest Stats===========")
    # Get model states
    account_value = []
    dates = []
    max_steps = environment.episode_length
    test_env, test_obs = environment.get_sb_env()
    test_env.reset()
    for i in range(max_steps):
        action, _states = model.predict(test_obs, deterministic=True)
        test_obs, rewards, dones, info = env_test.step(action)
        account_value.append(env_test.get_portfolio_value())
        dates.append(env_test.get_date())
    df_account_value = pd.DataFrame({'account_value': account_value, 'date': dates})

    now = datetime.now().strftime('%Y%m%d-%Hh%M')
    print('Model Backtest Stats')
    model_stats = backtest_stats(account_value=df_account_value)
    model_stats = pd.DataFrame(model_stats)
    
    # Get baseline stats
    baseline_df = pd.read_csv('data/dow_jones_data.csv')
    baseline_df['date'] = pd.to_datetime(baseline_df['date'])
    baseline_df = baseline_df[(baseline_df['date'] > df_account_value.loc[0,'date']) &
                              (baseline_df['date'] < df_account_value.loc[len(df_account_value)-1,'date'])]
    print('Baseline Backtest Stats')
    baseline_stats = backtest_stats(baseline_df, value_col_name = 'close')
    baseline_stats = pd.DataFrame(baseline_stats)

    merged = model_stats.merge(baseline_stats, left_index=True, right_index=True, suffixes=('_model', '_baseline'))
    merged.to_csv(f'backtests/backtest_{now}.csv')
    print(f'backtesting stats saved to backtest_{now}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FinRL Baseline')
    parser.add_argument('--config', type=str, default='config/baseline_config.yaml', help='Path to the config file')
    parser.add_argument('--algo', type=str, default='sac', help='Algorithm to use (ppo, a2c, ddpg, sac, td3)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the processed data file')
    args = parser.parse_args()

    config = load_yaml(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    features = ['close', 'high', 'low']
    tics = tics_grouped[2]

    if args.test:
        if args.algo == "ppo":
            from stable_baselines3 import PPO as model_class
        elif args.algo == "sac":
            from stable_baselines3 import SAC as model_class
        elif args.algo == "a2c":
            from stable_baselines3 import A2C as model_class
        elif args.algo == "ddpg":
            from stable_baselines3 import DDPG as model_class
        elif args.algo == "td3":
            from stable_baselines3 import TD3 as model_class
        else:
            raise ValueError(f"Unsupported algorithm: {args.algo}")
        trained_model = model_class.load(args.model_path)
        if not args.data_path:
            # raw_df = YahooDownloader(start_date = config['test_start_date'],
            #                         end_date = config['test_end_date'],
            #                         ticker_list = tics).fetch_data()
            raw_df = pd.read_csv(f"data/sub/{short_name_sha256('_'.join(tics))}_2009-01-01_2020-07-01.csv")
        else:
            raw_df = pd.read_csv(args.data_path)
        raw_df = raw_df[["date", "tic", "close", "high", "low", "volume"]]
        print('PROCESSING DATA')
        df = preprocess(raw_df)
        environment = PortfolioOptimizationEnv(
            df,
            initial_amount=config['initial_amount'],
            comission_fee_pct=0.0025,
            time_window=50,
            reward_scaling=float(config['reward_scaling']),
            features=features) # ,'close_30_sma', 'close_60_sma', 'volume'
        backtesting(environment, trained_model)
        
    else:
        if not args.data_path:
            # raw_df = YahooDownloader(start_date = config['test_start_date'],
            #                         end_date = config['test_end_date'],
            #                         ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
            raw_df = pd.read_csv(f"data/sub/{short_name_sha256('_'.join(tics))}_2009-01-01_2020-07-01.csv")
        else:
            raw_df = pd.read_csv(args.data_path)
            
        raw_df = raw_df[["date", "tic", "close", "high", "low", "volume"]]

        print('PROCESSING DATA')
        df = preprocess(raw_df)
        # 'close', 'high', 'low', 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix'
        assert np.all(np.isfinite(df[features]).values), "Dataframe contains non-finite values (NaN, Inf, or -Inf)"

        environment = PortfolioOptimizationEnv(
            df,
            initial_amount=config['initial_amount'],
            comission_fee_pct=0.0025,
            time_window=50,
            reward_scaling=float(config['reward_scaling']),
            features=features)
        
        algo = args.algo
        sb_env, _ = environment.get_sb_env()
        agent = DRLAgent(env=sb_env)
        model_params = config.get("model", {})
        model_params['device'] = device
        model = agent.get_model(algo, model_kwargs=model_params)

        # set up logger
        now = datetime.now().strftime('%Y%m%d-%Hh%M')
        tmp_path = f'log/{algo}/{now}'
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model.set_logger(new_logger)

        # Train the model
        checkpoint_dir = f"checkpoints/{algo}/{now}"
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir, name_prefix=f"{algo}_model")
        print('BEGIN TRAINING')
        trained_model = agent.train_model(model=model, tb_log_name=algo, total_timesteps=config['timesteps'], callback=checkpoint_callback)

        # Save the model
        if args.model_path is not None:
            trained_model.save(f"{args.model_path}/{short_name_sha256('_'.join(tics))}.zip")
            # trained_model.save(args.model_path)
            print(f"Model saved to {args.model_path}")
        else:
            trained_model.save(f"trained_models/{algo}/{now}.zip")
            print(f"Model saved to models/{algo}/{now}")
    