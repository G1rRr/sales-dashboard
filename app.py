import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="电商销售分析", layout="wide", page_icon="📊")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#1a1a2e}
[data-testid="stSidebar"] *{color:#e2e8f0 !important}
[data-testid="stSidebar"] .stSlider label{color:#94a3b8 !important}
[data-testid="stSidebar"] h2{color:#a78bfa !important;font-size:15px;font-weight:600;letter-spacing:0.05em;text-transform:uppercase}
.metric-card{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);border-radius:12px;padding:18px 20px;border:1px solid #4338ca}
.metric-val{font-size:26px;font-weight:700;color:#ffffff;letter-spacing:-0.5px}
.metric-lab{font-size:12px;color:#a5b4fc;margin-top:4px;letter-spacing:0.03em}
.metric-delta{font-size:11px;color:#6ee7b7;margin-top:6px}
.section-hd{font-size:15px;font-weight:600;color:#1e293b;margin:20px 0 8px;padding-left:10px;border-left:3px solid #6366f1}
div[data-testid="stHorizontalBlock"]{gap:12px}
.stPlotlyChart{border-radius:10px;overflow:hidden;border:1px solid #e2e8f0}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    df["日期"] = pd.to_datetime(df["日期"])
    return df

@st.cache_data
def compute_rules(_df):
    df_sold = _df[_df["成交件数（目标变量）"] > 0].copy()
    basket = df_sold.groupby(["日期","商品大类目编号"])["商品编号"].apply(list).reset_index()
    basket = basket[basket["商品编号"].apply(len) > 1]
    transactions = basket["商品编号"].apply(lambda x: [str(i) for i in x]).tolist()
    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    df_te = pd.DataFrame(te_arr, columns=te.columns_)
    fi = apriori(df_te, min_support=0.05, use_colnames=True, max_len=2)
    if len(fi) == 0:
        fi = apriori(df_te, min_support=0.02, use_colnames=True, max_len=2)
    rules = association_rules(fi, metric="lift", min_threshold=1.2, num_itemsets=len(fi))
    rules = rules[rules["consequents"].apply(len)==1].sort_values("lift", ascending=False).head(20)
    rules["前项"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["后项"] = rules["consequents"].apply(lambda x: list(x)[0])
    return rules

df_all = load_data()

# ── 侧边栏 ──
with st.sidebar:
    st.markdown("## 🎛 筛选控制")
    st.markdown("---")

    st.markdown("**📅 日期范围**")
    min_date = df_all["日期"].min().date()
    max_date = df_all["日期"].max().date()
    date_range = st.slider("", min_value=min_date, max_value=max_date,
        value=(min_date, max_date), format="YYYY-MM-DD")

    st.markdown("**📡 流量渠道**")
    channel_map = {
        "全部渠道": None,
        "直通车": "直通车引导浏览次数",
        "淘宝客": "淘宝客引导浏览次数",
        "搜索": "搜索引导浏览次数",
        "聚划算": "聚划算引导浏览次数"
    }
    channel = st.radio("", list(channel_map.keys()), index=0)

    st.markdown("**🗂 商品大类目**")
    all_cats = sorted(df_all["商品大类目编号"].unique())
    cat_options = ["全部类目"] + [str(c) for c in all_cats]
    selected_cat = st.selectbox("", cat_options)

    st.markdown("---")
    st.caption("数据周期：2022.10 — 2023.12")
    st.caption("SKU数：1000 | 记录数：232,621")

# ── 数据过滤 ──
df = df_all[(df_all["日期"].dt.date >= date_range[0]) & (df_all["日期"].dt.date <= date_range[1])]
if selected_cat != "全部类目":
    df = df[df["商品大类目编号"] == int(selected_cat)]

y_col = channel_map[channel] if channel_map[channel] else "成交金额"
y_label = channel if channel != "全部渠道" else "成交金额"

daily = df.groupby("日期")[y_col].sum().reset_index().sort_values("日期")
daily.columns = ["日期", "指标值"]
daily["7日均线"] = daily["指标值"].rolling(7).mean()
avg = daily["指标值"].mean()
std = daily["指标值"].std()
threshold = avg + 2 * std
peaks = daily[daily["指标值"] > threshold]

monthly = daily.set_index("日期").resample("ME")["指标值"].sum().reset_index()
monthly["月份"] = monthly["日期"].dt.strftime("%Y-%m")
promo_months = ["2022-11","2023-06","2023-11"]

q_map = {"Q4'22":["2022-10","2022-11","2022-12"],
         "Q1'23":["2023-01","2023-02","2023-03"],
         "Q2'23":["2023-04","2023-05","2023-06"],
         "Q3'23":["2023-07","2023-08","2023-09"]}
q_data = {k: monthly[monthly["月份"].isin(v)]["指标值"].sum() for k,v in q_map.items()}
q_max = max(q_data.values()) if q_data else 1

with st.spinner("计算关联规则..."):
    rules = compute_rules(df_all if selected_cat=="全部类目" else df_all[df_all["商品大类目编号"]==int(selected_cat)])

top_lift = rules["lift"].max() if len(rules) > 0 else 0
peak_ratio = peaks["指标值"].max() / avg if len(peaks) > 0 and avg > 0 else 0
q_vals = list(q_data.values())
q2_growth = (q_vals[2]-q_vals[1])/q_vals[1]*100 if len(q_vals)>2 and q_vals[1]>0 else 0

# ── 标题 ──
st.markdown("# 📊 电商销售数据分析")
st.caption(f"当前筛选：{date_range[0]} → {date_range[1]} | 渠道：{channel} | 类目：{selected_cat}")
st.markdown("---")

# ── 指标卡 ──
c1,c2,c3,c4 = st.columns(4)
cards = [
    (f"¥{avg/1e6:.1f}M" if channel=="全部渠道" else f"{avg/1e4:.1f}万", "日均指标值", "基于筛选范围"),
    (f"{peak_ratio:.1f}x", "促销峰值倍数", "双11 vs 日均"),
    (f"+{q2_growth:.0f}%", "Q2环比增长", "较Q1季度"),
    (f"{top_lift:.2f}", "最高关联提升度", "Apriori算法"),
]
for col, (val, lab, delta) in zip([c1,c2,c3,c4], cards):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lab">{lab}</div><div class="metric-delta">{delta}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── 趋势图 ──
st.markdown('<div class="section-hd">每日趋势与促销节点</div>', unsafe_allow_html=True)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=daily["日期"], y=daily["指标值"],
    fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    line=dict(color="#6366f1", width=1.5), name=y_label, opacity=0.9))
fig1.add_trace(go.Scatter(x=daily["日期"], y=daily["7日均线"],
    line=dict(color="#f43f5e", width=2.5, dash="solid"), name="7日均线"))
fig1.add_hline(y=threshold, line_dash="dot", line_color="#f59e0b", line_width=1.5,
    annotation_text="2σ异常阈值", annotation_font_size=11, annotation_font_color="#f59e0b")
for _, row in peaks.iterrows():
    fig1.add_annotation(x=row["日期"], y=row["指标值"],
        text=f"▲{row['日期'].strftime('%m/%d')}",
        showarrow=False, font=dict(size=10, color="#dc2626"), yshift=12)
fig1.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=1.12, x=0),
    margin=dict(l=0,r=0,t=10,b=0),
    xaxis=dict(gridcolor="#f1f5f9", showgrid=True),
    yaxis=dict(gridcolor="#f1f5f9", showgrid=True),
    hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# ── 月度面积图 + 季度雷达图 ──
col_a, col_b = st.columns([3,2])

with col_a:
    st.markdown('<div class="section-hd">月度趋势（面积图）</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=monthly["月份"], y=monthly["指标值"],
        fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="#6366f1", width=2),
        mode="lines+markers",
        marker=dict(size=[12 if m in promo_months else 6 for m in monthly["月份"]],
                    color=["#ef4444" if m in promo_months else "#6366f1" for m in monthly["月份"]],
                    symbol=["star" if m in promo_months else "circle" for m in monthly["月份"]]),
        name="月度成交",
        hovertemplate="%{x}<br>成交额：%{y:,.0f}<extra></extra>"))
    for _, row in monthly[monthly["月份"].isin(promo_months)].iterrows():
        fig2.add_annotation(x=row["月份"], y=row["指标值"],
            text="促销", showarrow=False,
            font=dict(size=10, color="#ef4444"), yshift=16)
    fig2.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(gridcolor="#f1f5f9", tickangle=-45),
        yaxis=dict(gridcolor="#f1f5f9"),
        showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="section-hd">季度对比（雷达图）</div>', unsafe_allow_html=True)
    categories = list(q_data.keys())
    values = [v/q_max for v in q_data.values()]
    fig3 = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=8, color="#6366f1"),
        name="季度成交"))
    fig3.update_layout(height=280, paper_bgcolor="white",
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0")),
        margin=dict(l=30,r=30,t=20,b=20),
        showlegend=False)
    for i, (cat, val) in enumerate(q_data.items()):
        fig3.add_annotation(
            text=f"¥{val/1e6:.0f}M",
            font=dict(size=10, color="#374151"),
            showarrow=False,
            xref="paper", yref="paper",
            x=[0.05, 0.95, 0.95, 0.05][i],
            y=[0.95, 0.95, 0.05, 0.05][i])
    st.plotly_chart(fig3, use_container_width=True)

# ── 关联规则 ──
st.markdown('<div class="section-hd">商品关联规则挖掘（Apriori）</div>', unsafe_allow_html=True)
col_c, col_d = st.columns([3,2])

with col_c:
    fig4 = go.Figure(go.Scatter(
        x=rules["support"], y=rules["confidence"],
        mode="markers+text",
        text=rules["前项"].astype(str)+"→"+rules["后项"].astype(str),
        textposition="top center", textfont=dict(size=9, color="#374151"),
        marker=dict(
            size=rules["lift"]*5,
            color=rules["lift"],
            colorscale=[[0,"#ddd6fe"],[0.5,"#8b5cf6"],[1,"#4c1d95"]],
            showscale=True,
            colorbar=dict(title="提升度", thickness=12, len=0.7),
            line=dict(width=1, color="white")),
        hovertemplate="前项：%{text}<br>支持度：%{x:.3f}<br>置信度：%{y:.3f}<extra></extra>"))
    fig4.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(title="支持度", gridcolor="#f1f5f9"),
        yaxis=dict(title="置信度", gridcolor="#f1f5f9"))
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.markdown("**Top 10 强关联商品对**")
    display = rules[["前项","后项","support","confidence","lift"]].head(10).copy()
    display.columns = ["前项","后项","支持度","置信度","提升度"]
    display["支持度"] = display["支持度"].round(3)
    display["置信度"] = display["置信度"].round(3)
    display["提升度"] = display["提升度"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True, height=340)

st.markdown("---")
st.markdown('<p style="font-size:11px;color:#94a3b8;text-align:center">数据分析：薛腾龙 | 方法：时间序列分析 · K-means聚类 · Apriori关联规则挖掘</p>', unsafe_allow_html=True)
