import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained

def construct_stock_features(df, df_market, df_rf) -> pd.DataFrame:
    # pepare market dataframe
    df_market = df_market[['date','close']].sort_values('date')
    df_market['market_ret'] = df_market['close'].pct_change()

    # pepare portfolio dataframe
    df = df[['date','tic','close', 'volume']].sort_values(['tic','date'])
    df['return'] = df.groupby('tic')['close'].pct_change()

    df_merged = (
        df[['date','tic','return', 'volume']]
        .merge(
            df_market[['date','market_ret']], 
            on="date", how="left"
        ).merge(
            df_rf, on="date", how="left"
        )
    ).dropna(subset=['return','market_ret', 'volume', 'rf_rate'])

    def calc_feats(df):
        stock_ret = df['return'].values
        market_ret = df['market_ret'].values
        rf_rate = df['rf_rate'].values

        # α、β
        cov = np.cov(stock_ret, market_ret, ddof=1)[0,1]
        beta  = cov / np.var(market_ret, ddof=1)
        alpha = np.mean(stock_ret) - beta * np.mean(market_ret)

        # 年化收益 & 波动率
        avg_ret = np.mean(stock_ret) * 252
        volatility     = np.std(stock_ret, ddof=1) * np.sqrt(252)

        # Sharpe
        excess = stock_ret - rf_rate / 252
        sharpe = np.mean(excess) / np.std(stock_ret, ddof=1) * np.sqrt(252)

        # 最大回撤
        cum_ret   = np.cumprod(1 + stock_ret)
        high_water= np.maximum.accumulate(cum_ret)
        drawdown  = cum_ret / high_water - 1
        max_dd    = drawdown.min()

        # avg_volume
        avg_volume = np.mean(df['volume'])

        return pd.Series({
            'alpha':        alpha,
            'beta':         beta,
            'avg_return':   avg_ret,
            'volatility':   volatility,
            'sharpe':       sharpe,
            'max_drawdown': max_dd,
            'avg_volume':   avg_volume
        })

    return (
        df_merged.groupby('tic').apply(calc_feats)
        .replace(np.inf, sys.float_info.max)
        .replace(-np.inf, sys.float_info.min)
    )

def cluster_tic(
    df,
    avg_size,
    diff,
    n_components=2,
    random_state=42
) -> list:
    """
    使用容量受限的 KMeans（来自 sklearn-extra），确保每簇大小在 [avg_size-diff, avg_size+diff] 之间。

    参数：
    - df: 特征 DataFrame，index 为 tic
    - avg_size: 目标平均簇大小
    - diff: 容差
    - random_state: 随机种子
    - n_components: PCA 降维维度

    返回：
    - clusters: List[List[str]]，所有簇，其大小均在 [avg_size-diff, avg_size+diff] 区间内
    """
    # 1) 计算簇数并确保至少 1
    total = df.shape[0]
    n_clusters = max(1, int(round(total / avg_size)))

    # 2) 标准化 + PCA
    X = StandardScaler().fit_transform(df.values)
    X_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X)

    # 3) 用 KMeansConstrained 做聚类
    size_min = max(1, avg_size - diff)
    size_max = avg_size + diff
    kmc = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state,
    )
    labels = kmc.fit_predict(X_pca)

    # 4) 根据 labels 构建簇列表
    clusters = []
    for lbl in range(n_clusters):
        members = df.index[labels == lbl].tolist()
        if members:
            clusters.append(members)

    return clusters