import sys
import pandas as pd
import numpy as np

sys.path.append("/home/miw039/scalable_rl_portfolio_management/FinRL-dev")

from hierarchicalPortfolioOptEnv import hierarchicalPortfolioOptEnv
from tic_config import tics_grouped
hash_codes = ['8190e4275b4db67d', '8d1de7fe38361930']
model_lst = [f'trained_models/sac/{h}' for h in hash_codes]
ticker_grps = [tics_grouped[0], tics_grouped[2]]
data_pths = [f'data/sub/{h}_2009-01-01_2020-07-01.csv' for h in hash_codes]

df = pd.DataFrame(columns=['date','close','high','low','open','volume','tic','day'])
for pth in data_pths:
    df = pd.concat([df, pd.read_csv(pth)])

features = ['close', 'high', 'low']

env = hierarchicalPortfolioOptEnv(df, 
                                  100_000, 
                                  model_lst, 
                                  ticker_grps,
                                  comission_fee_pct=0.0025,
                                  time_window=50,
                                  reward_scaling=1e-4,
                                  features=features)
obs = env.reset()
print(obs)
