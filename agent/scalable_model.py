import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC as model_class

from .data_downloader import get_data, get_market_df, get_rf_rate
from .split import construct_stock_features, cluster_tic
from .helper import short_name_sha256, tics_group_name, file_path, compute_sub_df

class Scalable():
    def __init__(self, super_env, sub_env, dir=None):
        self.super_env = super_env
        self.sub_env = sub_env
        self.dir = dir

        self.tics = None
        self.tics_lst = None
        self.data = None

        self.sub_models = None
        self.manager_model = None

    def load_data(self, data):
        self.data = data

    def load_sub(self, tics_list, model_dir, train_start_date, train_end_date):
        self.tics_lst = tics_list
        self.sub_models = []
        for sub_tics in self.tics_lst:
            model_path = file_path(model_dir, sub_tics, train_start_date, train_end_date, suffix='zip', type='r')
            if not model_path.is_file():
                return False
            self.sub_models.append(model_class.load(model_path))

        return True

    def split(
            self, tics, start_date, end_date, 
            market_tic, rf_tic,
            avg_sub_model_size, allow_size_diff, 
            n_PCA_components, random_state
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


    def train_sub(self, start_date, end_date, config):
        """
        Train sub-models for each sub-tic.
        """
        self.sub_models = []
        for sub_tics in self.tics_lst:
            sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(sub_tics))]
            if sub_data.empty:
                self.data = get_data(self.tics, start_date, end_date)
                sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(sub_tics))]
            
            # TODO
            # train(config, model_path, data_path, model_name,
            #     algo='sac', retrain=False, features=['close', 'high', 'low'], device='cpu')

            # self.sub_models.append(sub_model)

    def test_sub(self, start_date, end_date):
        #TODO
        pass

    def train(
            self, tics, start_date, end_date, 
            market_tic="S&P 500", rf_tic="^IRX",
            avg_sub_model_size=30, allow_size_diff=5,
            n_PCA_components=2, random_state=42
        ):

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))].empty:
            self.data = get_data(tics, start_date, end_date)

        self.tics = tics
        if self.tics_lst is None or all(sub_tics in self.tics for lst in self.tics_lst for sub_tics in lst):
            self.split(
                tics, start_date, end_date, 
                market_tic, rf_tic,
                avg_sub_model_size, allow_size_diff, 
                n_PCA_components, random_state
            )

        if self.sub_models is None or len(self.sub_models) != len(self.tics_lst):
            if not self.load_sub(
                self.tics_lst, 
                model_dir="models" if dir is None else f'{dir}/models', 
                train_start_date=start_date, 
                train_end_date=end_date
            ):
                # TODO
                self.train_sub(
                    start_date, end_date, 
                    # config={
                    #     'algo': 'sac',
                    #     'total_timesteps': 50000,
                    #     'features': ['close', 'high', 'low'],
                    #     'device': 'cpu'
                    # }
                )

        sub_data = []
        for sub_tics, sub_model in zip(self.tics_lst, self.sub_models):
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

        #TODO train manager
        return self.manager_model
        
    def test(self, start_date, end_date):
        assert self.manager_model is not None, "Manager model not trained yet."

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(self.tics))].empty:
            self.data = get_data(self.tics, start_date, end_date)

        sub_data = []
        for sub_tics, sub_model in zip(self.tics_lst, self.sub_models):
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

        # manager_test
