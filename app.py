import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="电商销售数据分析Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<style>
.metric-container{background:#f8f9fa;border-radius:10px;padding:16px;text-align:center;border:1px solid #e9ecef}
.metric-val{font-size:28px;font-weight:600;color:#1a1a2e}
.metric-lab{font-size:13px;color:#6c757d;margin-top:4px}
.section-title{font-size:18px;font-weight:600;color:#1a1a2e;margin:24px 0 12px 0;padding-bottom:6px;border-bottom:2px solid #4f46e5}
</style>
""", unsafe_allow_html=True)

st.title("📊 电商销售数据分析 Dashboard")
st.caption("数据周期：2022.10 — 2023.12 | 商品数：1000 SKU | 记录数：232,621条")

@st.cache_data
def load_data():
    df = pd.read_csv("2023商品销售数据.csv")
    df["日期"] = pd.to_datetime(df["日期"])
    return df

@st.cache_data
def compute_association(_df):
    df_sold = _df[_df["成交件数（目标变量）"] > 0].copy()
    basket = df_sold.groupby(["日期", "商品大类目编号"])["商品编号"].apply(list).reset_index()
    basket = basket[basket["商品编号"].apply(len) > 1]
    transactions = basket["商品编号"].apply(lambda x: [str(i) for i in x]).tolist()
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_te = pd.DataFrame(te_array, columns=te.columns_)
    frequent_itemsets = apriori(df_te, min_support=0.05, use_colnames=True, max_len=2)
    if len(frequent_itemsets) == 0:
        frequent_itemsets = apriori(df_te, min_support=0.02, use_colnames=True, max_len=2)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2, num_itemsets=len(frequent_itemsets))
    rules = rules[rules["consequents"].apply(len) == 1].sort_values("lift", ascending=False).head(15)
    return rules

df = load_data()

daily = df.groupby("日期")["成交金额"].sum().reset_index().sort_values("日期")
daily["7日均线"] = daily["成交金额"].rolling(7).mean()
avg = daily["成交金额"].mean()
std = daily["成交金额"].std()
threshold = avg + 2 * std
peaks = daily[daily["成交金额"] > threshold]
monthly = daily.set_index("日期").resample("ME")["成交金额"].sum().reset_index()
monthly["月份"] = monthly["日期"].dt.strftime("%Y-%m")
peak_month = monthly.loc[monthly["成交金额"].idxmax()]

q_map = {"Q4 2022": ["2022-10","2022-11","2022-12"],
          "Q1 2023": ["2023-01","2023-02","2023-03"],
          "Q2 2023": ["2023-04","2023-05","2023-06"],
          "Q3 2023": ["2023-07","2023-08","2023-09"]}
q_data = {k: monthly[monthly["月份"].isin(v)]["成交金额"].sum() for k, v in q_map.items()}

with st.spinner("正在挖掘商品关联规则..."):
    rules = compute_association(df)

top_lift = rules["lift"].max()
q2 = q_data["Q2 2023"]
q1 = q_data["Q1 2023"]
q2_growth = (q2 - q1) / q1 * 100
peak_ratio = peaks["成交金额"].max() / avg

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-container"><div class="metric-val">¥{avg/1e6:.1f}M</div><div class="metric-lab">日均成交额</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-container"><div class="metric-val">{peak_ratio:.1f}x</div><div class="metric-lab">双11是日均的倍数</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-container"><div class="metric-val">+{q2_growth:.0f}%</div><div class="metric-lab">Q2环比Q1增长</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-container"><div class="metric-val">{top_lift:.2f}</div><div class="metric-lab">最高关联提升度</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📈 每日成交趋势与促销节点识别</div>', unsafe_allow_html=True)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=daily["日期"], y=daily["成交金额"], mode="lines",
    name="每日成交额", line=dict(color="#93c5fd", width=1), opacity=0.6))
fig1.add_trace(go.Scatter(x=daily["日期"], y=daily["7日均线"], mode="lines",
    name="7日滚动均线", line=dict(color="#ef4444", width=2)))
fig1.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
    annotation_text=f"2σ阈值 ({threshold/1e6:.1f}M)", annotation_position="top left")
for _, row in peaks.iterrows():
    fig1.add_annotation(x=row["日期"], y=row["成交金额"],
        text=row["日期"].strftime("%m-%d"), showarrow=True,
        arrowhead=2, arrowcolor="#dc2626", font=dict(size=11, color="#dc2626"), ax=0, ay=-30)
fig1.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=1.1), margin=dict(l=0,r=0,t=20,b=0),
    yaxis=dict(gridcolor="#f3f4f6"), xaxis=dict(gridcolor="#f3f4f6"))
st.plotly_chart(fig1, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="section-title">📅 月度成交金额</div>', unsafe_allow_html=True)
    promo = ["2022-11", "2023-06", "2023-11"]
    colors = ["#ef4444" if m in promo else "#6366f1" for m in monthly["月份"]]
    fig2 = go.Figure(go.Bar(x=monthly["月份"], y=monthly["成交金额"],
        marker_color=colors, text=(monthly["成交金额"]/1e6).round(1).astype(str)+"M",
        textposition="outside", textfont=dict(size=10)))
    fig2.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=0), xaxis_tickangle=-45,
        yaxis=dict(gridcolor="#f3f4f6"),
        annotations=[dict(x=0.5, y=1.05, xref="paper", yref="paper",
            text="红色=促销峰值月", showarrow=False, font=dict(size=11, color="#6b7280"))])
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title">📊 季度成交对比</div>', unsafe_allow_html=True)
    qdf = pd.DataFrame({"季度": list(q_data.keys()), "成交额": list(q_data.values())})
    fig3 = go.Figure(go.Bar(x=qdf["季度"], y=qdf["成交额"],
        marker_color=["#6366f1","#6366f1","#ef4444","#6366f1"],
        text=(qdf["成交额"]/1e6).round(1).astype(str)+"M",
        textposition="outside", textfont=dict(size=12)))
    fig3.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(gridcolor="#f3f4f6"),
        annotations=[dict(x=0.5, y=1.05, xref="paper", yref="paper",
            text="红色=最高季度(Q2 2023)", showarrow=False, font=dict(size=11, color="#6b7280"))])
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('<div class="section-title">🔗 商品关联规则分析</div>', unsafe_allow_html=True)
col_c, col_d = st.columns([3, 2])
with col_c:
    rules["前项"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["后项"] = rules["consequents"].apply(lambda x: list(x)[0])
    fig4 = go.Figure(go.Scatter(
        x=rules["support"], y=rules["confidence"],
        mode="markers+text",
        text=rules["前项"].astype(str) + "→" + rules["后项"].astype(str),
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=rules["lift"]*5, color=rules["lift"],
            colorscale="YlOrRd", showscale=True,
            colorbar=dict(title="提升度"), line=dict(width=1, color="#9ca3af"))))
    fig4.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(title="支持度(Support)", gridcolor="#f3f4f6"),
        yaxis=dict(title="置信度(Confidence)", gridcolor="#f3f4f6"))
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.markdown('<div class="section-title">Top 10 关联规则明细</div>', unsafe_allow_html=True)
    display_rules = rules[["前项","后项","support","confidence","lift"]].head(10).copy()
    display_rules.columns = ["前项商品","后项商品","支持度","置信度","提升度"]
    display_rules["支持度"] = display_rules["支持度"].round(3)
    display_rules["置信度"] = display_rules["置信度"].round(3)
    display_rules["提升度"] = display_rules["提升度"].round(2)
    st.dataframe(display_rules, use_container_width=True, hide_index=True, height=360)

st.markdown("---")
st.markdown('<p style="font-size:12px;color:#9ca3af;text-align:center">数据分析：薛腾龙 | 方法：时间序列分析 · Apriori关联规则挖掘</p>', unsafe_allow_html=True)
