import torch
import pandas as pd
import numpy as np
import yaml
import argparse
import itertools
from datetime import datetime
import os
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from finrl.config import INDICATORS

def load_yaml(configpath):
    with open(configpath, 'r') as f:
        return yaml.safe_load(f)
    
def load_model(algo, path, sb_env=None, model_params=None):
    assert path is not None, "Model path must be provided for retraining"
    if algo == "ppo":
        from stable_baselines3 import PPO as model_class
    elif algo == "sac":
        from stable_baselines3 import SAC as model_class
    elif algo == "a2c":
        from stable_baselines3 import A2C as model_class
    elif algo == "ddpg":
        from stable_baselines3 import DDPG as model_class
    elif algo == "td3":
        from stable_baselines3 import TD3 as model_class
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    if sb_env is not None:
        return model_class.load(path, env=sb_env, **model_params)
    else:
        return model_class.load(path, **model_params)

def preprocess(df, start_date, end_date):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list = INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature = False)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] > pd.to_datetime(start_date)) & (df['date'] < pd.to_datetime(end_date))]
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    processed_full = processed_full.fillna(0)
    return processed_full

def backtesting(env_test, model, log_path):
    print("=" * 10 + "Get Backtest Stats" + "=" * 10)
    now = datetime.now().strftime('%Y%m%d-%Hh%M')
    # Get model states
    account_value = []
    dates = []
    max_steps = env_test.episode_length
    test_env, test_obs = env_test.get_sb_env()
    test_env.reset()
    weight_history = []
    for i in range(max_steps):
        action, _states = model.predict(test_obs, deterministic=True)
        test_obs, rewards, dones, info = env_test.step(action)
        weight_history.append(env_test.get_final_weights())
        account_value.append(env_test.get_portfolio_value())
        dates.append(env_test.get_date())
    df_account_value = pd.DataFrame({'account_value': account_value, 'date': dates})

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
    
    output = {}
    merged = model_stats.merge(baseline_stats, left_index=True, right_index=True, suffixes=('_model', '_baseline'))
    output['backtest'] = merged
    output['weights'] = pd.DataFrame({'date': dates, 'weights': weight_history})
    output['account_value'] = df_account_value
    return output

def train(config, model_path, data_path, model_name, log_path=None,
          data_df=None, algo='sac', retrain=False, features=['close', 'high', 'low'], device='cpu'):
    if data_path is not None:
        assert data_df is None, "data_df must not be provided if data_path is not None"
        raw_df = pd.read_csv(data_path)
        raw_df = raw_df[["date", "tic", "close", "high", "low", "volume"]]
    else:
        assert data_df is not None, "data_df must be provided if data_path is None"
        raw_df = data_df.copy()
        raw_df = raw_df[["date", "tic", "close", "high", "low", "volume"]]

    print('PROCESSING DATA')
    df = preprocess(raw_df, config['train_start_date'], config['train_end_date'])
    assert np.all(np.isfinite(df[["date", "tic", "close", "high", "low", "volume"]]).values), "Dataframe contains non-finite values (NaN, Inf, or -Inf)"

    environment = PortfolioOptimizationEnv(
        df,
        initial_amount=config['initial_amount'],
        comission_fee_pct=0.0025,
        time_window=50,
        reward_scaling=float(config['reward_scaling']),
        features=features)
    
    sb_env, _ = environment.get_sb_env()
    agent = DRLAgent(env=sb_env)
    model_params = config.get("model", {})
    model_params['device'] = device
    if retrain:
        model = load_model(algo, os.path.join(model_path, model_name + ".zip"), sb_env, model_params)
    else:
        model = agent.get_model(algo, model_kwargs=model_params)

    if log_path is not None:
        # set up logger
        tmp_path = f'{log_path}/log'
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model.set_logger(new_logger)

        # Train the model
        checkpoint_dir = f"{log_path}/checkpoint"
        checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=checkpoint_dir, name_prefix=model_name)
        print('BEGIN TRAINING')
        trained_model = agent.train_model(model=model, tb_log_name=algo, total_timesteps=config['timesteps'], callback=checkpoint_callback)
    else:
        print('BEGIN TRAINING')
        trained_model = agent.train_model(model=model, total_timesteps=config['timesteps'])

    # Save the model
    trained_model.save(f"{model_path}/{model_name}.zip")
    print(f"Model saved to {model_path}")

def test(config, model_path, data_path, model_name, log_path,
         data_df=None, algo='sac', features=['close', 'high', 'low'], device='cpu',
         save_weights=True, save_backtest=True, save_account_value=True):
    if algo == "ppo":
        from stable_baselines3 import PPO as model_class
    elif algo == "sac":
        from stable_baselines3 import SAC as model_class
    elif algo == "a2c":
        from stable_baselines3 import A2C as model_class
    elif algo == "ddpg":
        from stable_baselines3 import DDPG as model_class
    elif algo == "td3":
        from stable_baselines3 import TD3 as model_class
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    trained_model = load_model(algo, os.path.join(model_path, model_name + ".zip"), sb_env=None, model_params={'device': device})
    if data_path is not None:
        assert data_df is None, "data_df must not be provided if data_path is not None"
        raw_df = pd.read_csv(data_path)
        raw_df = raw_df[["date", "tic", "close", "high", "low", "volume"]].drop_duplicates()
    else:
        assert data_df is not None, "data_df must be provided if data_path is None"
        raw_df = data_df.copy()
        raw_df = raw_df[["date", "tic", "close", "high", "low", "volume"]].drop_duplicates()

    print('PROCESSING DATA')
    df = preprocess(raw_df, config['test_start_date'], config['test_end_date'])
    environment = PortfolioOptimizationEnv(
        df,
        initial_amount=config['initial_amount'],
        comission_fee_pct=0.0025,
        time_window=50,
        reward_scaling=float(config['reward_scaling']),
        features=features) # ,'close_30_sma', 'close_60_sma', 'volume'
    result = backtesting(environment, trained_model, log_path, save_account_value=save_account_value,
                        save_weights=save_weights, save_test=save_test)
    if save_backtest:
        result['backtest'].to_csv(os.path.join(log_path, 'backtest', f'backtest_{model_name}.csv'))
        print('Backtest results saved to:', os.path.join(log_path, 'backtest', f'backtest_{model_name}.csv'))
    if save_weights:
        result['weights'].to_csv(os.path.join(log_path, 'weights', f'weights_{model_name}.csv'), index=False)
        print('Weights saved to:', os.path.join(log_path, 'weights', f'weights_{model_name}.csv'))
    if save_account_value:
        result['account_value'].to_csv(os.path.join(log_path, 'account_values', f'account_values_{model_name}.csv'), index=False)
        print('Account value saved to:', os.path.join(log_path, 'account_values', f'account_values_{model_name}.csv'))
    return result
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FinRL Baseline')
    parser.add_argument('--config', type=str, default='config/baseline_config.yaml', help='Path to the config file')
    parser.add_argument('--algo', type=str, default='sac', help='Algorithm to use (ppo, a2c, ddpg, sac, td3)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load the model')
    parser.add_argument('--log_path', type=str, default='logs', help='Path to save logs and results')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to the processed data file')
    args = parser.parse_args()

    config = load_yaml(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    features = ['close', 'high', 'low']

    if args.test:
        model_dir, model_file = os.path.split(args.model_path)
        model_name, _ = os.path.splitext(model_file)
        test(config, model_dir, args.data_path, model_name, args.log_path, algo=args.algo, device=device, features=features)
    else:
        model_dir, model_file = os.path.split(args.model_path)
        model_name, _ = os.path.splitext(model_file)
        train(config, model_dir, args.data_path, model_name, args.log_path, algo=args.algo, retrain=args.retrain, device=device, features=features)
    