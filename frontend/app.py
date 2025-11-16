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
)
from api.news_sentiment import fetch_news_headlines
from ml.baseline_model import run_baseline_model
from ml.lab2_mlp_model import predict_with_mlp, _build_tabular_features
from ml.lab5_transformer_model import predict_with_transformer
from ml.scenario_generator import simulate_paths, summarize_paths, compute_scenario_params
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

        st.caption("分析區間：過去一年；預測日期：**今天**。")
        show_details = st.checkbox("顯示詳細數據與計算過程", value=False)
        show_scenario_details = st.checkbox("顯示場景模擬的歷史波動 + 情緒計算細節", value=False)

    st.subheader("股票資訊")
    st.write(f"**名稱**: {selected_stock.name} ({selected_stock.ticker})")
    if selected_stock.description:
        st.write(f"**簡介**: {selected_stock.description}")

    if st.button("開始分析"):
        today = date.today()

        with st.spinner("從 Yahoo Finance 下載數據、新聞與基本面並運行模型..."):
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

        # Show latest與前一日收盤價
        st.markdown("### 模型結果比較（預測日期：今日）")
        if len(series.prices) >= 2:
            last_close = series.prices[-1].close
            prev_close = series.prices[-2].close
            st.write(
                f"**最近兩個交易日收盤價**：前一日 = {prev_close:,.2f}，最近一日 = {last_close:,.2f} "
                f"（變化：約 {(last_close/prev_close - 1)*100:.2f}%）"
            )
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

        # Scenario simulation based on historical volatility + sentiment
        st.markdown("### 場景模擬：未來報酬分佈（基於歷史波動 + 情緒）")
        horizon_days = 20
        paths = simulate_paths(
            series,
            horizon_days=horizon_days,
            n_paths=1000,
            sentiment=sentiment if isinstance(sentiment, (int, float)) else None,
        )
        scen = summarize_paths(paths)
        if scen:
            st.write(
                f"**未來 {horizon_days} 日上漲機率 (P[回報 > 0])**: {scen['up_prob']:.2%}"
            )
            st.write(
                f"**中位數報酬**: {scen['median_return']*100:.2f}%  "
                f"（10%分位: {scen['p10_return']*100:.2f}%，90%分位: {scen['p90_return']*100:.2f}%）"
            )
            st.write(
                f"**極端情境**: 最差約 {scen['worst_return']*100:.2f}%，最好約 {scen['best_return']*100:.2f}%"
            )
            st.caption(
                "※ 此模擬使用過去一年日報酬的均值與波動度，並根據當前新聞情緒（sentiment）對平均報酬做小幅調整，"
                "以 Monte Carlo 方式生成多條價格路徑，計算未來報酬的機率分佈。"
            )

            # Optional: show detailed parameters when user wants
            if show_scenario_details:
                scen_params = compute_scenario_params(
                    series,
                    sentiment=sentiment if isinstance(sentiment, (int, float)) else None,
                )
                st.markdown("#### 場景模型參數（每日報酬）")
                st.write(f"- 歷史樣本數 `n_obs`: {scen_params['n_obs']}")
                st.write(
                    f"- 歷史平均日報酬 `mu`: {scen_params['mu']*100:.4f}% "
                    f"(約年化 ≈ {scen_params['mu']*252*100:.2f}%)"
                )
                st.write(
                    f"- 歷史日波動 `sigma`: {scen_params['sigma']*100:.4f}% "
                    f"(約年化 ≈ {scen_params['sigma']*(252**0.5)*100:.2f}%)"
                )
                st.write(
                    f"- 情緒修正後平均日報酬 `mu_tilted`: {scen_params['mu_tilted']*100:.4f}% "
                    f"(若 sentiment 為正 → 期望略上調；為負 → 略下調)"
                )
                st.caption(
                    "※ 上述 mu / sigma 來自過去一年日報酬分佈，"
                    "場景模擬假設未來日報酬 ~ Normal(mu_tilted, sigma)，獨立抽樣組合出 20 日路徑。"
                )

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
                fy = fundamentals.get("fiscal_year_end") or "N/A"
                cur = fundamentals.get("financial_currency") or "USD"
                last_fy_date = fundamentals.get("last_fiscal_year_end_date")
                mrq_date = fundamentals.get("most_recent_quarter_end")

                st.write(f"**市值**: {mc:,.0f} {cur}")
                st.write(f"**市盈率 (PE)**: {pe:.2f}")
                st.write(f"**市銷率 (P/S)**: {ps:.2f}")
                st.write(f"**股息率**: {dy:.2%}")
                st.write(f"**淨利率**: {pm:.2%}")
                st.write(f"**收入增長 (YoY)**: {rg:.2%}")
                st.write(f"**會計年度結算月份**: {fy}")
                if last_fy_date:
                    year = last_fy_date.split("-")[0]
                    st.write(f"**上個財政年度結算日**: {last_fy_date}（報表年度：約為 {year} 年）")
                if mrq_date:
                    st.write(f"**最近公佈季度報告結束日**: {mrq_date}")
                st.caption("※ 數據與日期來自 Yahoo Finance (`lastFiscalYearEnd`, `mostRecentQuarter` 等欄位)。")

                # 最近兩年年度財報重點
                last_two_years = fundamentals.get("financials_last_two_years") or []
                if last_two_years:
                    st.markdown("**最近兩個年度財報摘要（收入與淨利）**")
                    fy_df = pd.DataFrame(last_two_years)
                    fy_df.rename(
                        columns={
                            "year": "年度",
                            "total_revenue": "總收入 (Total Revenue)",
                            "net_income": "淨利 (Net Income)",
                        },
                        inplace=True,
                    )
                    st.table(
                        fy_df.style.format(
                            {
                                "總收入 (Total Revenue)": "{:,.0f}",
                                "淨利 (Net Income)": "{:,.0f}",
                            }
                        )
                    )
            else:
                st.info("未能取得基本面資料（可能是 Yahoo Finance 限制）。")

        with news_col:
            st.markdown("#### 近期新聞（NewsAPI）")
            if news_headlines:
                for item in news_headlines[:5]:
                    # 兼容舊版快取（純字串）與新版 dict 結構
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        published_at = item.get("publishedAt", "")
                    else:
                        title = str(item)
                        published_at = ""
                    if published_at:
                        st.write(f"- {published_at[:10]} – {title}")
                    else:
                        st.write(f"- {title}")
            else:
                st.info("未能取得新聞（請檢查 NewsAPI Key 或稍後再試）。")

        st.markdown(
            """
            ### 提示
            - 分析區間固定為**過去一個月**，預測目標為**今日**的方向與買賣區間。
            - 基線模型只用價格，MLP / Transformer 模型會額外利用新聞情緒與基本面（當有訓練好的權重時）。
            """
        )

        if show_details:
            st.markdown("### 詳細數據與計算過程（後端輸入與特徵）")

            # 價格時間序列表：對應後端 StockPriceSeries / DataFrame 輸入
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

            st.subheader("過去一年每日價格（後端使用的時間序列資料）")
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
            )

            # 特徵計算：與 MLP / Transformer 使用的 tabular features 對應
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

            st.subheader("MLP / Transformer 特徵向量（模型輸入）")
            st.table(feature_df.style.format({"數值": "{:,.4f}"}))

            st.markdown(
                """
                **計算方式說明：**

                - `ma_10` = 最近 10 天收盤價的平均值  
                - `ma_30` = 最近 30 天收盤價的平均值  
                - `std_10` = 最近 10 天收盤價的標準差（波動度）  
                - `std_30` = 最近 30 天收盤價的標準差  
                - `sentiment` = 利用 NewsAPI 抓取近期新聞標題，經 VADER 計算平均情緒分數（約在 [-1, 1]）  
                - `pe_ratio` / `ps_ratio` 等基本面數據來自 Yahoo Finance 的最新財報資訊  

                模型會根據這些特徵，輸出預期未來一段時間的報酬率，再轉換為：
                - **預期走勢**（up / down / flat）  
                - 對應的 **建議買入價 / 賣出價**（在最近收盤價附近加上 +/- 若干百分比）
                """
            )


if __name__ == "__main__":
    main()


