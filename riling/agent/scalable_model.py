import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC as model_class

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from .data_downloader import get_data
from ...agent.split import construct_stock_features, cluster_tic
from .helper import short_name_sha256

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

class Scalable():
    def __init__(self, super_env, sub_env):
        self.super_env = super_env
        self.sub_env = sub_env

        self.tics_lst = None
        self.tics = None
        self.data = None

        self.sub_models = None

    def split(
            self, tics, start_date, end_date, 
            market_tic, rf_tic,
            avg_sub_model_size, allow_size_diff, 
            n_PCA_components, random_state
        ):
        self.data, market_df, rf_df = get_data(
            tics, 
            start_date, 
            end_date, 
            market_tic, 
            rf_tic
        )

        X = construct_stock_features(self.data, market_df, rf_df)

        self.tics_lst = cluster_tic(
            X, avg_sub_model_size, allow_size_diff,
            n_PCA_components, random_state
        )

    def load_sub(self, tics_list, model_path, train_start_date, train_end_date):
        self.tics_lst = tics_list
        self.sub_models = []
        for sub_tics in self.tics_lst:
            model_path = Path(model_path) / f"{short_name_sha256('_'.join(sub_tics))}_{train_start_date}_{train_end_date}.zip"
            self.sub_models.append(model_class.load(model_path))

    def get_data():


    def train(
            self, tics, start_date, end_date, 
            market_tic="S&P 500", rf_tic="^IRX",
            avg_sub_model_size=30, allow_size_diff=5,
            n_PCA_components=2, random_state=42
        ):


        self.split(
            tics, start_date, end_date, 
            market_tic, rf_tic,
            avg_sub_model_size, allow_size_diff, 
            n_PCA_components, random_state
        )

        


    # def train_sub(algo, env_train, total_timesteps=50000):
    #     agent = DRLAgent(env = env_train)
    #     model = agent.get_model(algo)

    #     # set up logger
    #     tmp_path = RESULTS_DIR + f'/{algo}'
    #     new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    #     # Set new logger
    #     model.set_logger(new_logger)

    #     # Train the model
    #     trained_model = agent.train_model(model=model, tb_log_name=algo, total_timesteps=total_timesteps)

    #     return trained_model
        
    def test():
        pass