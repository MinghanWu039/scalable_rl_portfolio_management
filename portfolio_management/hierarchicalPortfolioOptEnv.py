import gym
import math
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import sys
import quantstats as qs

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
sys.path.append("/home/miw039/scalable_rl_portfolio_management/FinRL-dev")
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv


class hierarchicalPortfolioOptEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a hierarchical portfolio optimization environment for reinforcement learning.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(        
        self,
        df,
        initial_amount,
        model_pths,
        ticker_grps,
        model_cls="sac",
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=1,
        cwd="./",
        new_gym_api=False
    ):
        super(hierarchicalPortfolioOptEnv, self).__init__()
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api

        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # initialize price variation
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, normalize_df, tics_in_portfolio)

        # Initialize the models and the tickers
        if model_cls == "ppo":
            from stable_baselines3 import PPO as model_class
        elif model_cls == "sac":
            from stable_baselines3 import SAC as model_class
        elif model_cls == "a2c":
            from stable_baselines3 import A2C as model_class
        elif model_cls == "ddpg":
            from stable_baselines3 import DDPG as model_class
        elif model_cls == "td3":
            from stable_baselines3 import TD3 as model_class
        else:
            raise ValueError(f"Unsupported algorithm: {model_cls}")
        
        assert len(model_pths) == len(ticker_grps), "model_pths and ticker_grps must have the same length"
        self._num_grps = len(model_pths)
        self.ticker_grps = ticker_grps
        self.models = []
        self.envs = []
        self.env_obs = []
        for model_path, tickers in zip(model_pths, ticker_grps):
            self.models.append(model_class.load(model_path))
            df_sub = self._df[self._df[tic_column].isin(tickers)]
            environment = PortfolioOptimizationEnv(
                df_sub,
                initial_amount=initial_amount,
                comission_fee_pct=0.0025,
                time_window=50,
                reward_scaling=reward_scaling,
                features=features,
                order_df=order_df,
                return_last_action=return_last_action,
                normalize_df=normalize_df,
                comission_fee_model=comission_fee_model,
                comission_fee_pct=comission_fee_pct,
                valuation_feature=valuation_feature,
                time_column=time_column,
                time_format=time_format,
                tic_column=tic_column,
                tics_in_portfolio=tics_in_portfolio,
                time_window=time_window,) 
            env, obs = environment.get_sb_env()
            env.reset()
            self.envs.append(environment)
            self.env_obs.append([obs])

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self._num_grps,))

        # define observation state
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self._num_grps,),
                    ),
                    "last_action": spaces.Box(low=0, high=1, shape=(self._num_grps,)),
                }
            )
        else:
            # if information about last action is not relevant,
            # a 3D observation space is defined
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._num_grps,),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def step(self, actions):
        self._terminal = self._time_index >= len(self._sorted_times) - 1
        if self._terminal:
            metrics_df = pd.DataFrame(
                {
                    "date": self._date_memory,
                    "returns": self._portfolio_return_memory,
                    "rewards": self._portfolio_reward_memory,
                    "portfolio_values": self._asset_memory["final"],
                }
            )
            metrics_df.set_index("date", inplace=True)
            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory["final"][0]))
            print(f"Final portfolio value: {self._portfolio_value}")
            print(
                "Final accumulative portfolio value: {}".format(
                    self._portfolio_value / self._asset_memory["final"][0]
                )
            )
            print(
                "Maximum DrawDown: {}".format(
                    qs.stats.max_drawdown(metrics_df["portfolio_values"])
                )
            )
            print("Sharpe ratio: {}".format(qs.stats.sharpe(metrics_df["returns"])))
            print("=================================")
        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state = self._get_obs(self._time_index)

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)
            # calculate the new portfolio value after changing the weights
            self._portfolio_value = np.dot(weights, self._state)

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # define portfolio return and env reward
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            portfolio_reward = np.log(rate_of_return)
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        return self._state, self._reward, self._terminal
    

    def reset(self):
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state = self._get_obs()
        self._portfolio_value = self._initial_amount
        self._terminal = False

        return self._state
    
    def render(self, mode="human"):
        """Renders the environment.

        Returns:
            Observation of current simulation step.
        """
        return self._state
    
    def get_portfolio_value(self):
        """Returns the current portfolio value."""
        return self._portfolio_value
    
    def enumerate_portfolio(self):
        weights = self._actions_memory[-1]
        subgrp_vals = weights * self._state
        output = {}
        for i, grp in enumerate(self.ticker_grps):
            environment = self.envs[i]
            for j, ticker in enumerate(grp):
                output[ticker] = subgrp_vals[i] * environment.get_final_weights()[j]
        return output
    
    def get_date(self):
        """Returns the current date."""
        return self._date_memory[-1].strftime("%Y-%m-%d")

    def _get_obs(self):
        """Returns the observation state for the current time step.

        Returns:
            obs: observation state
            info: additional information
        """
        subgrp_vals = []
        for model, env, env_ob in zip(self.models, self.envs, self.env_obs):
            action, _states = model.predict(env_ob[-1], deterministic=True)
            ob, rewards, dones, info = env.step(action)
            env_ob.append(ob)
            subgrp_vals.append(env.get_portfolio_value())
        subgrp_vals = np.array(subgrp_vals)
        return subgrp_vals
    
    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize datetimes
        self._date_memory = [date_time]
    
    def _softmax_normalization(self, actions):
        """Normalizes the action vector using softmax function.

        Returns:
            Normalized action vector (portfolio vector).
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output