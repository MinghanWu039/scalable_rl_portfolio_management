import pandas as pd
from stable_baselines3 import SAC as model_class
import os
from pathlib import Path

from .data_downloader import get_data, get_market_df, get_rf_rate
from .split import construct_stock_features, cluster_tic
from .helper import tics_group_name, file_path, compute_sub_df

from . import baseline


class Scalable():
    def __init__(self, config, dir=None, device='cpu', algo='sac', ):
        self.dir = dir
        self.device=device
        self.algo = algo
        self.config = config

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
            print("Could not find model file, training sub-models...")
            self.train_sub(tics, train_start_date, train_end_date, model_dir)

        return baseline.load_model(self.algo, model_path)

    def split(
            self, tics, start_date, end_date, 
            market_tic="S&P 500", rf_tic="^IRX",
            avg_sub_model_size=30, allow_size_diff=5,
            n_PCA_components=2, random_state=42
        ):

        market_df =  get_market_df(start_date, end_date, market_tic, dir = "data" if self.dir is None else f'{self.dir}/data')
        rf_df = get_rf_rate(start_date, end_date, rf_tic, dir = "data" if self.dir is None else f'{self.dir}/data')

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))].empty:
            print('No data available for the specified date range and tics. Fetching data...')
            self.data = get_data(tics, start_date, end_date, dir = "data" if self.dir is None else f'{self.dir}/data')

        X = construct_stock_features(self.data, market_df, rf_df)

        return cluster_tic(
            X, avg_sub_model_size, allow_size_diff,
            n_PCA_components, random_state
        )


    def train_sub(self, tics, start_date, end_date, model_dir = 'models'):
        sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))]
        if sub_data.empty:
            print('No data available for the specified date range and tics. Fetching data...')
            self.data = get_data(self.tics, start_date, end_date)
            sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))]
        
        total_path = file_path(model_dir, tics, start_date, end_date, suffix='zip', type='w')
        model_dir, model_file = os.path.split(total_path)
        model_name, _ = os.path.splitext(model_file)

        return baseline.train(self.config, model_dir, None, model_name, log_path=None, data_df=sub_data, algo=self.algo, device=self.device)

    def test_sub(self, tics, start_date, end_date, backtest=True, wights=False, valuse=False):
        results = {}
        if backtest:
            backtest_path = file_path(self.dir/'result/backtest', tics, start_date, end_date, suffix='csv', type='w')
            if backtest_path.is_file():
                results['backtest'] = pd.read_csv(backtest_path)
                backtest = False

        if wights:
            weights_path = file_path(self.dir/'result/weights', tics, start_date, end_date, suffix='csv', type='w')
            if weights_path.is_file():
                results['weights'] = pd.read_csv(weights_path)
                wights = False

        if valuse:
            values_path = file_path(self.dir/'result/account_value', tics, start_date, end_date, suffix='csv', type='w')
            if values_path.is_file():
                results['account_value'] = pd.read_csv(values_path)
                valuse = False

        if backtest or wights or valuse:
            sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))]
            if sub_data.empty:
                print('No data available for the specified date range and tics. Fetching data...')
                self.data = get_data(self.tics, start_date, end_date)
                sub_data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(tics))]
            total_path = file_path(self.dir/'models/sub', tics, start_date, end_date, suffix='zip', type='r')
            model_dir, model_file = os.path.split(total_path)
            model_name, _ = os.path.splitext(model_file)
            log_path = file_path(self.dir/'result', tics, start_date, end_date, suffix='csv', type='w')
            log_path, _ = os.path.split(log_path)
            results.update(baseline.test(self.config, model_dir, None, model_name, log_path, data_df=sub_data, algo=self.algo, device=self.device, save_weights=wights, save_test=backtest, save_account_value=valuse))

        if len(results) == 1:
            results = results[list(results.keys())[0]]

        return results

        

    def construct_manager_df(self, start_date, end_date):
        sub_data = []
        for sub_tics in enumerate(self.tics_lst):
            results = self.test_sub(self, sub_tics, start_date, end_date, backtest=False, wights=True, valuse=True)
            weights_df = results['wights']
            value_df = results['account_value']
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
        
        return pd.concat(sub_data, ignore_index=True)
    
    def train(
            self, tics, start_date, end_date, 
            market_tic="S&P 500", rf_tic="^IRX",
            avg_sub_model_size=30, allow_size_diff=5,
            n_PCA_components=2, random_state=42
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

        tics_hashed = [tics_group_name(sub_tics) for sub_tics in self.tics_lst]

        total_path = file_path(self.dir/'model', tics_hashed, start_date, end_date, suffix='zip', type='w')
        model_dir, model_file = os.path.split(total_path)
        model_name, _ = os.path.splitext(model_file)
        self.manager_model = baseline.train(self.config, model_dir, None, model_name, log_path=self.dir, data_df=manager_data, algo=self.algo, device=self.device)
        
        return self.manager_model
        
    def test(self, start_date, end_date):
        assert self.manager_model is not None, "Manager model not trained yet."
        assert self.tics_lst is not None, "Tics list not defined. Please run the split method first."

        if self.data is None or self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date) & (self.data['tic'].isin(self.tics))].empty:
            print('No data available for the specified date range and tics. Fetching data...')
            self.data = get_data(self.tics, start_date, end_date)

        manager_data = self.construct_manager_df(self, start_date, end_date)

        tics_hashed = [tics_group_name(sub_tics) for sub_tics in self.tics_lst]

        total_path = file_path(self.dir/'model', tics_hashed, start_date, end_date, suffix='zip', type='w')
        model_dir, model_file = os.path.split(total_path)
        model_name, _ = os.path.splitext(model_file)
        results = baseline.test(self.config, model_dir, None, model_name, log_path=self.dir, data_df=manager_data, algo=self.algo, device=self.device)

        if len(results) == 1:
            return results[list(results.keys())[0]]
        return results
