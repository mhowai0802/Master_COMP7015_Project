from datetime import date
import os
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so that `api`, `ml`, and `domain` are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.price_data import (
    fetch_fundamental_snapshot,
    fetch_price_history,
    fetch_sentiment_score,
    fetch_intraday_history,
)
from api.news_sentiment import fetch_news_headlines
from model_code.baseline_model import run_baseline_model
from model_code.lab2_mlp_model import predict_with_mlp, _build_tabular_features
from model_code.lab5_transformer_model import predict_with_transformer
from model_code.scenario_generator import simulate_paths, summarize_paths, compute_scenario_params
from model_code.train_mlp_model import train_mlp
from model_code.train_transformer_model import train_transformer
from domain.stocks import Stock
from domain.predictions import PredictionInput
from domain.timing import analyze_intraday_volatility, get_best_monitoring_hours, format_hour_label


WATCHLIST = [
    Stock(name="Apple", ticker="AAPL", description="蘋果（Apple）"),
    Stock(name="Microsoft", ticker="MSFT", description="微軟（Microsoft）"),
    Stock(name="NVIDIA", ticker="NVDA", description="英偉達（NVIDIA）"),
    Stock(name="Alphabet (Google)", ticker="GOOGL", description="Alphabet（Google）"),
    Stock(name="Amazon", ticker="AMZN", description="亞馬遜（Amazon）"),
    Stock(name="Meta", ticker="META", description="Meta（Facebook）"),
    Stock(name="Tesla", ticker="TSLA", description="特斯拉（Tesla）"),
    Stock(name="Broadcom", ticker="AVGO", description="博通（Broadcom）"),
    Stock(name="TSMC", ticker="TSM", description="台積電（TSMC）"),
    Stock(name="Super Micro Computer", ticker="SMCI", description="超微電腦（SMCI）"),
]


def apply_custom_css():
    """Apply custom CSS for better UI styling"""
    st.markdown("""
    <style>
    /* Main container styling - reduce padding */
    .main {
        padding-top: 0.5rem;
    }
    
    /* Header styling - more compact */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 0.25rem;
        margin-top: 0.25rem;
        font-size: 1.8rem;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.25rem;
        font-size: 1.3rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        font-size: 1.1rem;
    }
    
    h4 {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
        font-size: 1rem;
    }
    
    /* Card-like containers */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Success/Warning/Info boxes */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Spacing improvements - reduce margins */
    .element-container {
        margin-bottom: 0.25rem;
    }
    
    /* Prediction card containers - more compact */
    .stContainer {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 0.25rem;
    }
    
    /* Reduce metric spacing */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    
    /* Reduce block spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Compact info boxes */
    .stAlert {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Custom badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    
    .badge-warning {
        background-color: #ffc107;
        color: #212529;
    }
    
    .badge-info {
        background-color: #17a2b8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="AI Stocks — Buy/Sell Decision",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📈"
    )
    
    apply_custom_css()

    # Header
    st.title("📈 AI Stocks — Buy/Sell Decision Helper")

    with st.sidebar:
        st.header("⚙️ 設定")
        stock_names = [f"{s.name} ({s.ticker})" for s in WATCHLIST]
        choice = st.selectbox("📊 選擇股票", stock_names, index=0)
        selected_stock = WATCHLIST[stock_names.index(choice)]
        st.caption("分析區間：過去一年；預測日期：**今天**")
        
        st.markdown("---")
        analyze_button = st.button("🚀 開始分析", use_container_width=True, type="primary")
        
        st.markdown("---")
        if st.button("🔄 重新訓練模型", use_container_width=True):
            with st.spinner("⏳ 訓練中..."):
                try:
                    train_mlp()
                    train_transformer()
                    st.success("✅ 完成")
                except Exception as e:
                    st.error(f"❌ 錯誤: {e}")
        
        st.markdown("---")
        st.markdown("**👁️ 顯示選項**")
        show_details = st.checkbox("顯示詳細數據", value=False)
        show_scenario_details = st.checkbox("顯示場景細節", value=False)

    # Stock info display
    st.markdown(f"**{selected_stock.name} ({selected_stock.ticker})**")
    
    if analyze_button:
        today = date.today()

        with st.spinner("⏳ 從 Yahoo Finance 下載數據、新聞與基本面並運行模型..."):
            try:
                # 過去一年的價格數據
                series = fetch_price_history(selected_stock, lookback_days=365)
                pred_input = PredictionInput(
                    stock=selected_stock,
                    as_of_date=today,
                    horizon_days=30,
                )
                sentiment = fetch_sentiment_score(selected_stock)
                fundamentals = fetch_fundamental_snapshot(selected_stock)
                # 取得新聞標題（會自動快取）
                # 如果沒有設定 NEWSAPI_KEY，這裡可能會失敗，但模型仍可運行
                try:
                    from config.api_keys import NEWSAPI_KEY

                    news_headlines = (
                        fetch_news_headlines(selected_stock, api_key=NEWSAPI_KEY)
                        if NEWSAPI_KEY
                        else []
                    )
                except Exception:
                    news_headlines = []

                baseline_pred = run_baseline_model(series, pred_input)
                mlp_pred = predict_with_mlp(
                    series,
                    pred_input,
                    sentiment=sentiment,
                    fundamentals=fundamentals,
                )
                transformer_pred = predict_with_transformer(
                    series,
                    pred_input,
                    sentiment=sentiment,
                    fundamentals=fundamentals,
                )
            except Exception as e:  # pragma: no cover - UI convenience
                st.error(f"分析時出現錯誤: {e}")
                return

        # Compact price display
        if len(series.prices) >= 2:
            last_close = series.prices[-1].close
            prev_close = series.prices[-2].close
            price_change = (last_close/prev_close - 1)*100
            
            price_cols = st.columns(4)
            with price_cols[0]:
                st.metric("前一日", f"${prev_close:,.2f}")
            with price_cols[1]:
                st.metric("最近一日", f"${last_close:,.2f}")
            with price_cols[2]:
                st.metric("變化", f"{price_change:+.2f}%", delta=f"{price_change:+.2f}%")
            with price_cols[3]:
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Use tabs to organize content
        tab1, tab2, tab3, tab4 = st.tabs(["📊 模型預測", "🎲 場景模擬", "📰 新聞與基本面", "⏰ 最佳監控時機"])
        
        with tab1:
            cols = st.columns(3)

            def render_prediction(col, title: str, pred, icon: str = "📈"):
                with col:
                    if pred is None:
                        st.info(f"{icon} {title}\n\n尚未有訓練好的模型權重。")
                        return
                    
                    direction_map = {"up": "📈 上升", "down": "📉 下降", "flat": "➡️ 橫向"}
                    direction = pred.expected_direction
                    direction_text = direction_map.get(direction, "❓ 未知")
                    
                    # Determine recommendation
                    if pred.should_buy:
                        rec_text = "✅ 買入"
                        rec_type = "success"
                    elif pred.should_sell:
                        rec_text = "⚠️ 賣出"
                        rec_type = "warning"
                    else:
                        rec_text = "⏸️ 觀望"
                        rec_type = "info"
                    
                    st.markdown(f"**{icon} {title}**")
                    pred_cols = st.columns(2)
                    with pred_cols[0]:
                        st.metric("走勢", direction_text, delta=None)
                        st.metric("信心", f"{pred.confidence:.2f}")
                    with pred_cols[1]:
                        st.metric("買入價", f"${pred.suggested_buy_price:,.2f}")
                        st.metric("賣出價", f"${pred.suggested_sell_price:,.2f}")
                    
                    if rec_type == "success":
                        st.success(rec_text)
                    elif rec_type == "warning":
                        st.warning(rec_text)
                    else:
                        st.info(rec_text)

            render_prediction(cols[0], "基線模型", baseline_pred, "📊")
            render_prediction(cols[1], "MLP 模型", mlp_pred, "🧠")
            render_prediction(cols[2], "Transformer 模型", transformer_pred, "🤖")

        with tab2:
            # Scenario simulation
            horizon_days = 20
            paths = simulate_paths(
                series,
                horizon_days=horizon_days,
                n_paths=1000,
                sentiment=sentiment if isinstance(sentiment, (int, float)) else None,
            )
            scen = summarize_paths(paths)
            if scen:
                scen_cols = st.columns(4)
                with scen_cols[0]:
                    st.metric("上漲機率", f"{scen['up_prob']:.1%}")
                with scen_cols[1]:
                    st.metric("中位數報酬", f"{scen['median_return']*100:+.2f}%")
                with scen_cols[2]:
                    st.metric("最差", f"{scen['worst_return']*100:.2f}%")
                with scen_cols[3]:
                    st.metric("最好", f"{scen['best_return']*100:.2f}%")
                
                st.caption(f"報酬區間: 10%分位={scen['p10_return']*100:.2f}%, 90%分位={scen['p90_return']*100:.2f}%")
                
                if show_scenario_details:
                    with st.expander("📊 場景模型參數詳情"):
                        scen_params = compute_scenario_params(
                            series,
                            sentiment=sentiment if isinstance(sentiment, (int, float)) else None,
                        )
                        st.write(f"歷史樣本數: {scen_params['n_obs']}")
                        st.write(f"歷史平均日報酬: {scen_params['mu']*100:.4f}% (年化 ≈ {scen_params['mu']*252*100:.2f}%)")
                        st.write(f"歷史日波動: {scen_params['sigma']*100:.4f}% (年化 ≈ {scen_params['sigma']*(252**0.5)*100:.2f}%)")
                        st.write(f"情緒修正後平均日報酬: {scen_params['mu_tilted']*100:.4f}%")

        with tab3:
            # 額外資訊：新聞與基本面
            info_col, news_col = st.columns(2)

            with info_col:
                st.markdown("**💼 基本面**")
                if fundamentals:
                    mc = fundamentals.get("market_cap", 0.0)
                    pe = fundamentals.get("pe_ratio", 0.0)
                    ps = fundamentals.get("ps_ratio", 0.0)
                    dy = fundamentals.get("dividend_yield", 0.0)
                    pm = fundamentals.get("profit_margin", 0.0)
                    rg = fundamentals.get("revenue_growth", 0.0)
                    cur = fundamentals.get("financial_currency") or "USD"
                    
                    fund_cols = st.columns(3)
                    with fund_cols[0]:
                        st.metric("市值", f"{mc:,.0f} {cur}")
                        st.metric("PE", f"{pe:.2f}")
                    with fund_cols[1]:
                        st.metric("P/S", f"{ps:.2f}")
                        st.metric("股息率", f"{dy:.2%}")
                    with fund_cols[2]:
                        st.metric("淨利率", f"{pm:.2%}")
                        st.metric("收入增長", f"{rg:+.2%}", delta=f"{rg:+.2%}")
                    
                    last_two_years = fundamentals.get("financials_last_two_years") or []
                    if last_two_years:
                        with st.expander("📊 年度財報"):
                            fy_df = pd.DataFrame(last_two_years)
                            fy_df.rename(
                                columns={
                                    "year": "年度",
                                    "total_revenue": "總收入",
                                    "net_income": "淨利",
                                },
                                inplace=True,
                            )
                            st.dataframe(fy_df.style.format({"總收入": "{:,.0f}", "淨利": "{:,.0f}"}), use_container_width=True)
                else:
                    st.info("未能取得基本面資料")

            with news_col:
                st.markdown("**📰 近期新聞**")
                if news_headlines:
                    for idx, item in enumerate(news_headlines[:5], 1):
                        if isinstance(item, dict):
                            title = item.get("title", "")
                            published_at = item.get("publishedAt", "")
                        else:
                            title = str(item)
                            published_at = ""
                        
                        date_str = published_at[:10] if published_at else "日期未知"
                        st.caption(f"{date_str}")
                        st.write(f"{title}")
                        if idx < len(news_headlines[:5]):
                            st.markdown("---")
                else:
                    st.info("未能取得新聞")
        
        with tab4:
            # Best timing to monitor tab
            st.markdown("**⏰ 最佳監控時機分析**")
            st.caption("根據歷史盤中波動模式與今日預測，推薦最佳監控時段")
            
            try:
                # Fetch intraday data
                # Note: yfinance limits 30m intervals to max 60 days
                with st.spinner("⏳ 分析盤中數據..."):
                    intraday_df = fetch_intraday_history(selected_stock, period_days=60, interval="30m")
                
                if intraday_df.empty:
                    st.warning("⚠️ 無法取得盤中數據。此功能需要歷史盤中價格數據。")
                    st.info("💡 提示：yfinance 對30分鐘間隔的數據有60天的限制。如果持續無法取得數據，可能是：\n"
                           "- 股票代碼不正確\n"
                           "- 網絡連接問題\n"
                           "- yfinance API暫時無法訪問\n"
                           "請稍後再試或檢查股票代碼。")
                else:
                    # Get the primary prediction direction (prefer MLP or Transformer over baseline)
                    primary_pred = mlp_pred or transformer_pred or baseline_pred
                    
                    if primary_pred is None:
                        st.info("請先運行模型預測以獲得今日走勢預測。")
                    else:
                        predicted_direction = primary_pred.expected_direction
                        direction_map = {"up": "📈 上升", "down": "📉 下降", "flat": "➡️ 橫向"}
                        direction_text = direction_map.get(predicted_direction, "❓ 未知")
                        
                        st.markdown(f"**今日預測走勢：** {direction_text}")
                        
                        # Get best monitoring hours
                        best_hours = get_best_monitoring_hours(
                            intraday_df,
                            series,
                            predicted_direction,
                            top_n=3,
                        )
                        
                        if best_hours:
                            st.markdown("---")
                            st.markdown("**🎯 推薦監控時段（按波動率排序）**")
                            
                            for idx, hour_info in enumerate(best_hours, 1):
                                with st.container():
                                    hour_cols = st.columns([2, 1, 1, 1])
                                    with hour_cols[0]:
                                        st.markdown(f"**{idx}. {format_hour_label(hour_info['hour'])}**")
                                    with hour_cols[1]:
                                        st.metric(
                                            "波動率",
                                            f"{hour_info['avg_volatility']*100:.2f}%",
                                            help="平均價格波動百分比"
                                        )
                                    with hour_cols[2]:
                                        st.metric(
                                            "平均價差",
                                            f"${hour_info['avg_range']:.2f}",
                                            help="該時段平均高低價差"
                                        )
                                    with hour_cols[3]:
                                        st.metric(
                                            "樣本數",
                                            f"{hour_info['count']}",
                                            help="歷史數據點數量"
                                        )
                                    if idx < len(best_hours):
                                        st.markdown("---")
                            
                            # Visual chart showing intraday volatility pattern
                            st.markdown("---")
                            st.markdown("**📊 盤中波動模式圖表**")
                            
                            # Analyze all intraday volatility for visualization
                            all_hourly_stats = analyze_intraday_volatility(intraday_df)
                            
                            if not all_hourly_stats.empty:
                                # Create a bar chart
                                chart_data = pd.DataFrame({
                                    "時段": [
                                        format_hour_label(int(row["hour"]), int(row["minute"]))
                                        for _, row in all_hourly_stats.iterrows()
                                    ],
                                    "平均波動率 (%)": all_hourly_stats["avg_volatility"] * 100,
                                })
                                
                                st.bar_chart(
                                    chart_data.set_index("時段"),
                                    y="平均波動率 (%)",
                                    use_container_width=True,
                                )
                                
                                # Show explanation
                                st.markdown("---")
                                st.markdown("**💡 說明**")
                                if predicted_direction == "up":
                                    st.info(
                                        "根據歷史數據，在預測為上升的日子中，上述時段通常出現較高的價格波動，"
                                        "是捕捉買入機會的最佳時機。建議在這些時段密切關注市場動態。"
                                    )
                                elif predicted_direction == "down":
                                    st.warning(
                                        "根據歷史數據，在預測為下降的日子中，上述時段通常出現較高的價格波動，"
                                        "可能出現較好的買入價格。建議在這些時段密切關注市場動態。"
                                    )
                                else:
                                    st.info(
                                        "根據歷史數據，在預測為橫向整理的日子中，上述時段通常出現較高的價格波動。"
                                        "建議在這些時段密切關注市場動態。"
                                    )
                        else:
                            st.info("無法計算最佳監控時段。請確保有足夠的歷史數據。")
            except Exception as e:
                st.error(f"分析監控時機時出現錯誤: {e}")
        
        # Put details in expander
        if show_details:
            with st.expander("🔍 詳細數據與計算過程"):
                price_records = [
                    {
                        "日期": p.date,
                        "開盤": p.open,
                        "最高": p.high,
                        "最低": p.low,
                        "收盤": p.close,
                        "成交量": p.volume,
                    }
                    for p in series.prices
                ]
                price_df = pd.DataFrame.from_records(price_records).sort_values("日期")
                st.dataframe(
                    price_df.style.format(
                        {
                            "開盤": "{:,.2f}",
                            "最高": "{:,.2f}",
                            "最低": "{:,.2f}",
                            "收盤": "{:,.2f}",
                            "成交量": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    height=300,
                )
                
                feats, last_close = _build_tabular_features(
                    series, sentiment=sentiment, fundamentals=fundamentals
                )
                feature_names = [
                    "last_close（最新收盤價）",
                    "ma_10（過去10日收盤價平均）",
                    "ma_30（過去30日收盤價平均）",
                    "std_10（過去10日收盤波動率）",
                    "std_30（過去30日收盤波動率）",
                    "sentiment（新聞情緒分數）",
                    "pe_ratio（市盈率）",
                    "ps_ratio（市銷率）",
                ]
                feature_df = pd.DataFrame(
                    {"特徵": feature_names, "數值": feats.astype(float)}
                )
                st.dataframe(
                    feature_df.style.format({"數值": "{:,.4f}"}),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()


