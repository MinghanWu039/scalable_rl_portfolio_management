import pandas as pd

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from .data_downloader import get_data
from .split import construct_stock_features, cluster_tic

class Scalable():
    def __init__(self, super_env, sub_env):
        self.super_env = super_env
        self.sub_env = sub_env

        self.tics_lst = None
        self.tics = None
        self.data = None

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
        
    def test():
        pass