from datetime import date
import os
import sys

import streamlit as st

# Ensure project root is on sys.path so that `api`, `ml`, and `domain` are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.price_data import (
    fetch_fundamental_snapshot,
    fetch_price_history,
    fetch_sentiment_score,
)
from api.news_sentiment import fetch_news_headlines
from ml.baseline_model import run_baseline_model
from ml.lab2_mlp_model import predict_with_mlp
from ml.lab5_transformer_model import predict_with_transformer
from domain.stocks import PredictionInput, Stock


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


def main() -> None:
    st.set_page_config(page_title="AI Stocks — Buy/Sell Decision", layout="wide")

    st.title("AI Stocks — Buy/Sell Decision Helper")
    st.markdown(
        "系統會根據過去一個月的數據、新聞情緒和基本面，給出**今日**的買入/賣出建議。"
    )

    with st.sidebar:
        st.header("設定")
        stock_names = [f"{s.name} ({s.ticker})" for s in WATCHLIST]
        choice = st.selectbox("選擇股票", stock_names, index=0)
        selected_stock = WATCHLIST[stock_names.index(choice)]

        st.caption("分析區間：過去一個月；預測日期：**今天**。")

    st.subheader("股票資訊")
    st.write(f"**名稱**: {selected_stock.name} ({selected_stock.ticker})")
    if selected_stock.description:
        st.write(f"**簡介**: {selected_stock.description}")

    if st.button("開始分析"):
        today = date.today()

        with st.spinner("從 Yahoo Finance 下載數據、新聞與基本面並運行模型..."):
            try:
                # 過去一個月的價格數據
                series = fetch_price_history(selected_stock, lookback_days=30)
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

        st.markdown("### 模型結果比較（預測日期：今日）")
        cols = st.columns(3)

        def render_prediction(col, title: str, pred):
            with col:
                st.markdown(f"#### {title}")
                if pred is None:
                    st.info("尚未有訓練好的模型權重（weight）。")
                    return
                direction_map = {"up": "上升", "down": "下降", "flat": "橫向/不明確"}
                st.write(f"**預期走勢**: {direction_map.get(pred.expected_direction, '未知')}")
                st.write(f"**信心指標**: {pred.confidence:.2f}")
                st.write(f"**建議買入價**: {pred.suggested_buy_price:.2f}")
                st.write(f"**建議賣出價**: {pred.suggested_sell_price:.2f}")

                st.write("**操作建議**:")
                if pred.should_buy:
                    st.success("模型偏向 **買入**。")
                elif pred.should_sell:
                    st.warning("模型偏向 **減倉/賣出**。")
                else:
                    st.info("模型建議 **觀望**。")

        render_prediction(cols[0], "基線模型（移動平均）", baseline_pred)
        render_prediction(cols[1], "Lab 2 MLP 模型", mlp_pred)
        render_prediction(cols[2], "Lab 5 Transformer 模型", transformer_pred)

        # 額外資訊：新聞與基本面
        st.markdown("### 新聞與基本面概覽")
        info_col, news_col = st.columns(2)

        with info_col:
            st.markdown("#### 基本面（Yahoo Finance）")
            if fundamentals:
                mc = fundamentals.get("market_cap", 0.0)
                pe = fundamentals.get("pe_ratio", 0.0)
                ps = fundamentals.get("ps_ratio", 0.0)
                dy = fundamentals.get("dividend_yield", 0.0)
                pm = fundamentals.get("profit_margin", 0.0)
                rg = fundamentals.get("revenue_growth", 0.0)

                st.write(f"**市值**: {mc:,.0f} USD")
                st.write(f"**市盈率 (PE)**: {pe:.2f}")
                st.write(f"**市銷率 (P/S)**: {ps:.2f}")
                st.write(f"**股息率**: {dy:.2%}")
                st.write(f"**淨利率**: {pm:.2%}")
                st.write(f"**收入增長 (YoY)**: {rg:.2%}")
            else:
                st.info("未能取得基本面資料（可能是 Yahoo Finance 限制）。")

        with news_col:
            st.markdown("#### 近期新聞（NewsAPI）")
            if news_headlines:
                for h in news_headlines[:5]:
                    st.write(f"- {h}")
            else:
                st.info("未能取得新聞（請檢查 NewsAPI Key 或稍後再試）。")

        st.markdown(
            """
            ### 提示
            - 分析區間固定為**過去一個月**，預測目標為**今日**的方向與買賣區間。
            - 基線模型只用價格，MLP / Transformer 模型會額外利用新聞情緒與基本面（當有訓練好的權重時）。
            """
        )


if __name__ == "__main__":
    main()


