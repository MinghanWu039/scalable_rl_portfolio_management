import torch
import pandas as pd
import yaml
import sys
import argparse
import itertools
import os
from datetime import datetime

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

sys.path.append("/home/miw039/scalable_rl_portfolio_management/FinRL-dev")

from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.agents.stablebaselines3.models import DRLAgent

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from hierarchicalPortfolioOptEnv import hierarchicalPortfolioOptEnv

from finrl.config import INDICATORS
from tic_config import tics_grouped

def load_yaml(configpath):
    with open(configpath, 'r') as f:
        return yaml.safe_load(f)

def preprocess(df):
    fe = FeatureEngineer(
        use_technical_indicator=False,
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
    parser = argparse.ArgumentParser(description='FinRL Hierarchical Portfolio Management')
    parser.add_argument('--config', type=str, default='config/manager_config.yaml', help='Path to the config file')
    parser.add_argument('--algo', type=str, default='sac', help='Algorithm to use (ppo, a2c, ddpg, sac, td3)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the processed data file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    config = load_yaml(args.config)
    model_params = config.get("model", {})
    model_params['device'] = device
    source = config.get("source", {})


    # get models and data
    hash_codes = source['hash_codes'].strip().split(' ')
    model_lst = [os.path.join(args.model_path, h) for h in hash_codes]
    ticker_grp_idx = source['ticker_grp_idx'].strip().split(' ')
    ticker_grps = [tics_grouped[int(idx)] for idx in ticker_grp_idx]
    data_pths = [os.path.join(args.data_path, eval(source['data_file_format']).format(h=h)) for h in hash_codes]
    df = pd.DataFrame(columns=['date','close','high','low','open','volume','tic','day'])
    for pth in data_pths:
        df = pd.concat([df, pd.read_csv(pth)])

    # create environment
    features = ['close', 'high', 'low']
    print('PROCESSING DATA')
    df = preprocess(df)
    environment = hierarchicalPortfolioOptEnv(df, 100_000, model_lst, ticker_grps, comission_fee_pct=0.0025,
                                      time_window=50, reward_scaling=1e-4, features=features)

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
        backtesting(environment, trained_model)
        
    else:
        algo = args.algo
        sb_env, _ = environment.get_sb_env()
        agent = DRLAgent(env=sb_env)
        model = agent.get_model(algo, model_kwargs=model_params)

        # set up logger
        now = datetime.now().strftime('%Y%m%d-%Hh%M')
        tmp_path = f'log/{algo}_manager/{now}'
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model.set_logger(new_logger)

        # Train the model
        checkpoint_dir = f"checkpoints/{algo}_manager/{now}"
        checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=checkpoint_dir, name_prefix=f"{algo}_model")
        print('BEGIN TRAINING')
        trained_model = agent.train_model(model=model, tb_log_name=algo, total_timesteps=config['timesteps'], callback=checkpoint_callback)

        # Save the model
        if args.model_path is not None:
            trained_model.save(os.path.join(args.model_path, f"{algo}_manager_{now}.zip"))
            print("Model saved to", os.path.join(args.model_path, f"{algo}_manager_{now}.zip"))
        else:
            trained_model.save(os.path.join('trained_models', f"{algo}_manager_{now}.zip"))
            print(f"Model saved to", os.path.join('trained_models', f"{algo}_manager_{now}.zip"))
    