import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
    n_clusters,
    min_cluster_size,
    n_components=2,
    random_state=42
) -> list:
    """
    使用 KMeans 聚类后，将小于 min_cluster_size 的“离群簇”成员重新分配到最近的大簇中，
    最终返回不包含单独小簇（离群组）的 2D 列表，确保没有簇是“离群单独簇”。

    参数：
    - df: 特征 DataFrame，index 为 tic
    - n_clusters: 初始簇数
    - min_cluster_size: 若某簇大小 < min_cluster_size，则视为“离群簇”
    - n_components: PCA 维度
    - random_state: 随机种子

    返回：
    - clusters: List[List[str]]，仅包含“有效簇”（大小 ≥ min_cluster_size）
    """
    # 标准化 + PCA
    X_std = StandardScaler().fit_transform(df.values)
    X_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X_std)
    
    # 初始 KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X_pca)
    centers = km.cluster_centers_
    
    # 统计簇大小
    counts = np.bincount(labels, minlength=n_clusters)
    small = np.where(counts < min_cluster_size)[0]
    large = np.where(counts >= min_cluster_size)[0]
    
    # 对“小簇”内成员重新分配到最近“大簇”
    for i in np.where(np.isin(labels, small))[0]:
        # 计算与所有大簇中心的距离
        dists = np.linalg.norm(X_pca[i] - centers[large], axis=1)
        # 重新指派到最近的大簇
        labels[i] = large[np.argmin(dists)]
    
    # 构建仅含大簇的结果列表
    clusters = [
        df.index[labels == c].tolist() 
        for c in large
    ]
    
    return clusters