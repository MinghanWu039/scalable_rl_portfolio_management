def compute_portfolio_ohlcv_with_values(
    tics_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    value_df: pd.DataFrame,
    name: str
) -> pd.DataFrame:
    """
    计算组合 OHLCV：
      - 跳过首日
      - open 使用前一日收盘市值（cap_prev）
      - 持仓股数 = 当日开盘可投资资本 / 当日开盘价
      - high = 现金 + Σ(shares * 当日high)
      - low  = 现金 + Σ(shares * 当日low)
      - close = 当日收盘市值（account_value）
      - volume = Σ(shares * 当日volume)
    """
    # 重命名并转换日期列
    weights_df = weights_df.rename(columns={'dates':'date'})
    for df in (tics_df, weights_df, value_df):
        df['date'] = pd.to_datetime(df['date'])

    # 准备账户价值和前一日资本
    val = (
        value_df[['date','account_value']]
        .drop_duplicates()
        .sort_values('date')
        .set_index('date')
    )
    val['cap_prev'] = val['account_value'].shift(1)

    # 解析权重数组
    def parse_weights(s: str):
        return np.array([float(x) for x in s.strip('[]').split()])

    wdf = (
        weights_df
        .assign(w_arr=lambda d: d['weights'].apply(parse_weights))
        .sort_values('date')
        .reset_index(drop=True)
    )

    records = []
    # 遍历日期，跳过首日和末日
    for i in range(1, len(wdf)-1):
        date = wdf.at[i, 'date']
        cap_prev = val.at[date, 'cap_prev']
        if pd.isna(cap_prev):
            continue

        # 使用前一日现金权重计算现金资本
        prev_w_all = wdf.at[i-1, 'w_arr']
        prev_cash = prev_w_all[0]
        cash_cap = cap_prev * prev_cash
        stock_cap = cap_prev - cash_cap

        # 当日行情，按ticker排序
        day = (
            tics_df[tics_df['date']==date]
            .sort_values('tic')
            .reset_index(drop=True)
        )
        opens  = day['open'].values
        highs  = day['high'].values
        lows   = day['low'].values
        vols   = day['volume'].values

        # 计算持仓股数：全部股票资本按当日开盘執行
        weights_stock = prev_w_all[1:]
        shares = (weights_stock * stock_cap) / opens

        # 计算 OHLCV
        open_port   = cap_prev
        high_port   = cash_cap + shares.dot(highs)
        low_port    = cash_cap + shares.dot(lows)
        close_port  = val.at[date, 'account_value']
        volume_port = shares.dot(vols)

        records.append({
            'date':   date,
            'open':   open_port,
            'high':   high_port,
            'low':    low_port,
            'close':  close_port,
            'volume': volume_port,
            'tic':    name,
            'day':    day['day'].iloc[0]
        })

    return pd.DataFrame(records)


import plotly.graph_objects as go
import plotly.express as px

def plot_account_value_comparison_plotly(
    models = None,
    model_labels = None,
    baseline = None,
    baseline_label = None,
    manager = None,
    manager_lable = None,
    x_col: str = 'date',
    y_col: str = 'account_value',
    title: str = None,
) -> go.Figure:
    """
    用 Plotly 绘制多条模型的归一化账户价值（收益率）曲线及一条基准曲线，
    其中所有 y_col 已从 1 起归一化，代表收益率曲线。

    参数
    ----
    models : list[pd.DataFrame]
        要对比的模型列表，每个 DataFrame 至少包含 x_col 和 y_col 两列。
    model_labels : list[str]
        与 models 一一对应的图例标签。
    baseline : pd.DataFrame
        基准模型的 DataFrame，结构同上。
    baseline_label : str
        基准模型的图例标签。
    x_col : str, optional
        用作横轴的列名，默认为 'date'。
    y_col : str, optional
        用作纵轴的列名，默认为 'account_value'。
    title : str, optional
        图表标题，默认为 "Normalized Return Comparison"。
    """


    fig = go.Figure()

    if models is not None:
        color_sequence = px.colors.qualitative.Pastel

        for i, (df, label) in enumerate(zip(models, model_labels)):
            # days = list(range(1, len(df) + 1))
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    # x=days,
                    y=df[y_col],
                    mode='lines',
                    name=label,
                    line=dict(color=color_sequence[i % len(color_sequence)], width=2),
                    opacity=0.6
                )
            )

    if manager is not None:
        # days = list(range(1, len(manager) + 1))
        fig.add_trace(
            go.Scatter(
                x=manager[x_col],
                # x=days,
                y=manager[y_col],
                mode='lines',
                name=manager_lable,
                line=dict(color='red')
            )
        )

    if baseline is not None:
        # days = list(range(1, len(baseline) + 1))
        fig.add_trace(
            go.Scatter(
                x=baseline[x_col],
                # x=days,
                y=baseline[y_col],
                mode='lines',
                name=baseline_label,
                line=dict(color='blue')
            )
        )

    scale = 2.4
    base_font = fig.layout.font.size or 12  # 默认12，如果你之前没设置过的话
    new_size = base_font * scale

    fig.update_layout(
        width=1200,
        height=800,
        # margin=dict(l=40, r=40, t=40, b=40),
        title=title,
        xaxis_title=x_col.capitalize(),
        yaxis_title="Total Return Rate",
        template="plotly_white",
        font=dict(
            size=new_size,         # 全局文字大小
            family="Time New Roman"  # 可选：指定字体系列
        ),
        title_font=dict(size=new_size * 1.2),    # 标题稍大一点
        legend_font=dict(size=new_size),         # 图例文字
        xaxis=dict(
            title_font=dict(size=new_size),
            tickfont=dict(size=new_size * 0.6)   # 刻度文字可以略小
        ),
        yaxis=dict(
            title_font=dict(size=new_size),
            tickfont=dict(size=new_size * 0.6)
        )
    )

    return fig