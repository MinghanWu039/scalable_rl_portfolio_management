import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC as model_class
import os

from .data_downloader import get_data, get_market_df, get_rf_rate
from .split import construct_stock_features, cluster_tic
from .helper import short_name_sha256, tics_group_name, file_path, compute_sub_df

from . import baseline

class Scalable():
    def __init__(self, super_env, sub_env, dir=None, device='cpu', algo='sac'):
        self.super_env = super_env
        self.sub_env = sub_env
        self.dir = dir
        self.device=device
        self.algo = algo

        self.tics = None
        self.tics_lst = None
        self.data = None

        self.manager_model = None

    def load_data(self, data):
        self.data = data

    def load_grouped_tics(self, tics_lst):
        self.tics_lst = tics_lst

    def load_sub(self, tics, model_dir, train_start_date, train_end_date):
        model_path = file_path(model_dir, tics, train_start_date, train_end_date, suffix='zip', type='r')
        if not model_path.is_file():
            self.train_sub()

        return baseline.load_model(self.algo, model_path)

    def split(
            self, tics, start_date, end_date, 
            market_tic="S&P 500", rf_tic="^IRX",
            avg_sub_model_size=30, allow_size_diff=5,
            n_PCA_components=2, random_state=42
        ):

        market_df =  get_market_df(start_date, end_date, market_tic, dir = "data" if dir is None else f'{dir}/data')
        rf_df = get_rf_rate(start_date, end_date, rf_tic, dir = "data" if dir is None else f'{dir}/data')

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))].empty:
            self.data = get_data(tics, start_date, end_date, dir = "data" if dir is None else f'{dir}/data')

        X = construct_stock_features(self.data, market_df, rf_df)

        return cluster_tic(
            X, avg_sub_model_size, allow_size_diff,
            n_PCA_components, random_state
        )


    def train_sub(self, algo, tics, start_date, end_date, config, model_dir):
        sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))]
        if sub_data.empty:
            print('No data available for the specified date range and tics. Fetching data...')
            self.data = get_data(self.tics, start_date, end_date)
            sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))]
        
        total_path = file_path(model_dir, tics, start_date, end_date, suffix='zip', type='r')
        model_dir, model_file = os.path.split(total_path)
        model_name, _ = os.path.splitext(model_file)
        baseline.train(config, model_dir, None, model_name, log_path=None, algo=self.algo, device=self.device)

    def test_sub(self, start_date, end_date, tics_list=None):
        #TODO
        pass

    def construct_manager_df(self, start_date, end_date):
        sub_data = []
        for i, sub_tics in enumerate(self.tics_lst):
            #TODO
            weights_df = ...
            value_df = ...
            tics_df = self.data[
                (self.data['date'] >= start_date) & 
                (self.data['date'] <= end_date) & 
                (self.data['tic'].isin(sub_tics))
            ].sort_values('date').reset_index(drop=True)

            sub_data.append(
                compute_sub_df(
                    tics_df,
                    weights_df,
                    value_df,
                    tics_group_name(sub_tics)
                )
            )
        
        manager_data = pd.concat(sub_data, ignore_index=True)
    
    def train(
            self, tics, start_date, end_date, 
            market_tic="S&P 500", rf_tic="^IRX",
            avg_sub_model_size=30, allow_size_diff=5,
            n_PCA_components=2, random_state=42, config = None
        ):
        self.tics = tics

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))].empty:
            self.data = get_data(self.tics, start_date, end_date)

        if self.tics_lst is None or all(sub_tics in self.tics for lst in self.tics_lst for sub_tics in lst):
            self.tics_list = self.split(
                tics, start_date, end_date, 
                market_tic, rf_tic,
                avg_sub_model_size, allow_size_diff, 
                n_PCA_components, random_state
            )

        manager_data = self.construct_manager_df(self, start_date, end_date)

        #TODO train manager
        return self.manager_model
        
    def test(self, start_date, end_date):
        assert self.manager_model is not None, "Manager model not trained yet."

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(self.tics))].empty:
            self.data = get_data(self.tics, start_date, end_date)

        manager_data = self.construct_manager_df(self, start_date, end_date)


        # manager_test
