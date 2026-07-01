import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Target 종목 리스트
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMD", "NVDA", "TSLA", "AMZN", "META", "NFLX", "INTC",
    "QCOM", "ADBE", "CRM", "ORCL", "SMCI", "AVGO", "TXN", "MU", "MRVL", "CDNS"
]

# 🌟 실시간 원/달러 환율을 가져오는 함수
def get_current_usd_krw():
    try:
        usd_krw_data = yf.download("USDKRW=X", period="1d", interval="1m", progress=False)
        if not usd_krw_data.empty:
            if isinstance(usd_krw_data.columns, pd.MultiIndex):
                usd_krw_data.columns = usd_krw_data.columns.get_level_values(0)
            return float(usd_krw_data["Close"].values.flatten()[-1])
    except Exception:
        pass
    return 1400.0  # 기본값

# 트레이딩뷰 로고 매핑 딕셔너리 (공통 사용)
tradingview_logos = {
    "AAPL": "apple", "MSFT": "microsoft", "GOOGL": "alphabet",             
    "AMD": "advanced-micro-devices", "NVDA": "nvidia", "TSLA": "tesla", 
    "AMZN": "amazon", "META": "meta-platforms", "NFLX": "netflix", 
    "INTC": "intel", "QCOM": "qualcomm", "ADBE": "adobe", 
    "CRM": "salesforce", "ORCL": "oracle", "SMCI": "super-micro-computer", 
    "AVGO": "broadcom", "TXN": "texas-instruments", "MU": "micron-technology", 
    "MRVL": "marvell-technology-inc", "CDNS": "cadence-design-systems"
}

# 개별 종목 데이터 수집 및 지표 계산 워커 함수 (스크리너용)
def process_single_ticker(ticker, exchange_rate):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

        close_series = pd.Series(data["Close"].values.flatten(), index=data.index)
        high_series = pd.Series(data["High"].values.flatten(), index=data.index)
        low_series = pd.Series(data["Low"].values.flatten(), index=data.index)

        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = float((100 - (100 / (1 + rs))).dropna().iloc[-1])

        TP = (high_series + low_series + close_series) / 3
        SMA_TP = TP.rolling(window=20).mean()
        MD = TP.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = float(((TP - SMA_TP) / (0.015 * MD + 1e-9)).dropna().iloc[-1])

        max_price = close_series.max()
        current_price_usd = close_series.iloc[-1]
        drawdown = ((current_price_usd - max_price) / max_price) * 100
        current_price_krw = current_price_usd * exchange_rate

        per, peg, market_cap_krw, operating_income_krw_str = np.nan, np.nan, np.nan, "N/A"
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            market_cap_usd = info.get("marketCap", np.nan)
            if market_cap_usd is not None and not pd.isna(market_cap_usd):
                market_cap_krw = (market_cap_usd * exchange_rate) / 100_000_000
            
            per = info.get("trailingPE")
            if per is None or pd.isna(per): per = info.get("forwardPE", np.nan)
            peg = info.get("pegRatio", np.nan)
            
            financials = ticker_obj.financials
            if "Operating Income" in financials.index:
                raw_income_usd = financials.loc["Operating Income"].iloc[0]
                if not pd.isna(raw_income_usd):
                    operating_income_krw_str = f"₩{(raw_income_usd * exchange_rate) / 100_000_000:,.0f}억원"
        except Exception: pass

        logo_keyword = tradingview_logos.get(ticker, ticker.lower())
        logo_url = f"https://s3-symbol-logo.tradingview.com/{logo_keyword}--big.svg"

        return {
            "로고": logo_url, "Ticker": str(ticker), "시가총액": market_cap_krw, 
            "작년 영업이익": operating_income_krw_str, "PER (배)": per, "PEG Ratio": peg,
            "RSI (14)": rsi, "CCI (20)": cci, "고점 대비 낙폭": float(drawdown), "현재가 (₩)": float(current_price_krw)
        }
    except Exception: return None

@st.cache_data(ttl=3600)
def run_fast_screener(selected_tickers, exchange_rate):
    screened_stocks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_single_ticker, ticker, exchange_rate): ticker for ticker in selected_tickers}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res is not None: screened_stocks.append(res)
    return screened_stocks


# ==========================================
# 🌟 Streamlit UI 레이아웃 및 탭 설정
# ==========================================
st.set_page_config(layout="wide", page_title="프로 퀀트 마스터 스크리너", initial_sidebar_state="collapsed")
current_rate = get_current_usd_krw()

# 사이드바 내부 부가 정보 배치
st.sidebar.title("ℹ️ 대시보드 정보")
st.sidebar.write(f"💵 **실시간 환율:** `{current_rate:,.2f} 원`")
st.sidebar.markdown("---")
st.sidebar.caption("💡 화면 왼쪽 위의 ☰ 버튼을 누르면 이 창이 닫히고 넓은 메인 화면을 볼 수 있습니다.")

# 메인 화면 상단 탭 생성
tab_screener, tab_backtest = st.tabs(["⚡ 실시간 마스터 스크리너", "📈 적립식 백테스팅"])

# ------------------------------------------
# 탭 1: 실시간 마스터 스크리너
# ------------------------------------------
with tab_screener:
    st.title("⚡ 초고속 과매도 우량주 마스터 스크리너")
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.write(f"⏱️ 마지막 업데이트: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 💵 **실시간 환율:** `{current_rate:,.2f} 원` 반영 중")
    with col2:
        if st.button("수동 새로고침", key="btn_refresh"):
            st.cache_data.clear()
            st.rerun()

    selected_tickers = st.multiselect("스크리닝할 종목을 선택하세요:", options=tickers, default=tickers, key="ms_tickers")

    if not selected_tickers:
        st.warning("최소한 하나의 종목을 선택해야 합니다.")
    else:
        with st.spinner("야후 파이낸스 고속 파이프라인 연결 중..."):
            results = run_fast_screener(selected_tickers, current_rate)

        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by="시가총액", ascending=False, na_position="last")
            
            st.metric(label="📊 수집된 우량주 개수", value=f"{len(results_df)} 개 종목")
            st.dataframe(
                results_df,
                column_config={
                    "로고": st.column_config.ImageColumn("아이콘", width="small"),
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "시가총액": st.column_config.NumberColumn("시가총액 (억원)", format="₩ %,.0f억원"),
                    "작년 영업이익": st.column_config.Column("작년 영업이익"),
                    "PER (배)": st.column_config.NumberColumn("PER (배)", format="%.2f"), 
                    "PEG Ratio": st.column_config.NumberColumn("PEG Ratio", format="%.2f"),
                    "RSI (14)": st.column_config.NumberColumn("RSI (14)", format="%.2f"),
                    "CCI (20)": st.column_config.NumberColumn("CCI (20)", format="%.2f"),
                    "고점 대비 낙폭": st.column_config.NumberColumn("고점 대비 낙폭", format="%.2f%%"),
                    "현재가 (₩)": st.column_config.NumberColumn("현재가 (원화)", format="₩ %,.0f원")
                },
                use_container_width=True, 
                hide_index=True
            )

# ------------------------------------------
# 탭 2: 적립식 백테스팅 (누적원금 실선화 반영)
# ------------------------------------------
with tab_backtest:
    st.title("📈 적립식 가치투자 복리 백테스팅 엔진")
    st.write("선택한 빅테크 종목을 과거 특정 시점부터 매달 꾸준히 모았을 때, 내 자산이 원화 기준으로 어떻게 불어났을지 시뮬레이션합니다.")
    
    st.markdown("---")
    
    set_col1, set_col2, set_col3 = st.columns(3)
    
    with set_col1:
        target_stock = st.selectbox("🎯 백테스팅할 종목 선택:", options=tickers, key="sb_target")
    with set_col2:
        period_choice = st.radio("📅 투자 기간 선택:", ["최근 1년", "최근 3년", "최근 5년"], index=1, horizontal=True, key="rd_period")
    with set_col3:
        monthly_budget = st.number_input("💰 매월 적립할 금액 (원):", min_value=10000, max_value=100000000, value=500000, step=50000, format="%d", key="ni_budget")

    years_map = {"최근 1년": 1, "최근 3년": 3, "최근 5년": 5}
    backtest_years = years_map[period_choice]
    start_date = datetime.now() - timedelta(days=backtest_years * 365)
    
    with st.spinner(f"📊 {target_stock}의 {backtest_years}년치 역사적 주가 정보 분석 중..."):
        df_history = yf.download(target_stock, start=start_date.strftime('%Y-%m-%d'), progress=False)
        
        if df_history.empty:
            st.error("주가 데이터를 가져오는 데 실패했습니다.")
        else:
            if isinstance(df_history.columns, pd.MultiIndex): 
                df_history.columns = df_history.columns.get_level_values(0)
            
            df_history['YearMonth'] = df_history.index.to_period('M')
            
            total_invested_krw = 0
            total_shares_owned = 0
            history_records = []
            
            last_month = None
            
            for date, row in df_history.iterrows():
                current_month = row['YearMonth']
                close_usd = float(row['Close'])
                close_krw = close_usd * current_rate
                
                if current_month != last_month:
                    total_invested_krw += monthly_budget
                    shares_bought = monthly_budget / close_krw
                    total_shares_owned += shares_bought
                    last_month = current_month
                
                current_value_krw = total_shares_owned * close_krw
                
                history_records.append({
                    "투자일": date,
                    "누적 투자 원금": total_invested_krw,
                    "누적 평가 금액": current_value_krw
                })
            
            final_share_price_usd = float(df_history['Close'].iloc[-1])
            final_share_price_krw = final_share_price_usd * current_rate
            final_eval_value_krw = total_shares_owned * final_share_price_krw
            total_profit_loss_pct = ((final_eval_value_krw - total_invested_krw) / total_invested_krw) * 100
            
            st.markdown("### 🏆 시뮬레이션 최종 성적표")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("총 납입 원금", f"₩{total_invested_krw:,.0f}원")
            with m_col2:
                st.metric("최종 평가 금액 (현재 가치)", f"₩{final_eval_value_krw:,.0f}원", 
                          delta=f"₩{final_eval_value_krw - total_invested_krw:+,.0f}원")
            with m_col3:
                st.metric("최종 누적 수익률", f"{total_profit_loss_pct:+.2f}%")
                
            # ----------------------------------------------------------------
            # 📈 Plotly 일별 디테일 트래킹 차트 설정 구역
            # ----------------------------------------------------------------
            st.markdown("### 📈 누적 자산 성장 그래프 (원금 vs 평가액 - 일별 상세 보기)")
            
            chart_df = pd.DataFrame(history_records)
            chart_df['날짜 표시'] = chart_df['투자일'].dt.strftime('%Y-%m-%d')
            chart_df['원금_포맷'] = chart_df['누적 투자 원금'].apply(lambda x: f"₩{x:,.0f}원")
            chart_df['평가액_포맷'] = chart_df['누적 평가 금액'].apply(lambda x: f"₩{x:,.0f}원")

            import plotly.graph_objects as go

            fig = go.Figure()

            # 🌟 변경 포인트: 누적 투자 원금 선 스타일을 dash='dash'에서 dash='solid'(실선)로 변경
            fig.add_trace(go.Scatter(
                x=chart_df['투자일'],
                y=chart_df['누적 투자 원금'],
                name='누적 투자 원금',
                mode='lines',
                line=dict(color='#FFA07A', width=2, dash='solid'),  # 실선으로 변경됨
                customdata=chart_df['원금_포맷'],
                hovertemplate="누적 투자 원금: %{customdata}<extra></extra>"
            ))

            # 누적 평가 금액 선 (초록 실선)
            fig.add_trace(go.Scatter(
                x=chart_df['투자일'],
                y=chart_df['누적 평가 금액'],
                name='누적 평가 금액',
                mode='lines',
                line=dict(color='#2ECC71', width=2.5),
                customdata=chart_df['평가액_포맷'],
                hovertemplate="누적 평가 금액: <b>%{customdata}</b><extra></extra>"
            ))

            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(
                    tickformat="%Y-%m-%d",
                    tickmode="auto",
                    nticks=12
                ),
                yaxis=dict(
                    tickformat=", .0f",
                    title="금액 (원)"
                ),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
            # ----------------------------------------------------------------
            
            logo_key = tradingview_logos.get(target_stock, target_stock.lower())
            st.sidebar.markdown("---")
            st.sidebar.image(f"https://s3-symbol-logo.tradingview.com/{logo_key}--big.svg", width=50)
            st.sidebar.write(f"테스팅 중인 종목: **{target_stock}**")

# ==========================================
# 공통 하단 용어 해설 가이드
# ==========================================
# ==========================================
# 공통 하단 용어 해설 가이드 (완성본)
# ==========================================
st.markdown("---")
st.markdown("""
### 📊 스크리너 핵심 퀀트 지표 완전 정복 Guide

1. **PER (Price-to-Earnings Ratio, 주가수익비율)**
   - **설명:** 현재 주가를 1주당 순이익(EPS)으로 나눈 값으로, **"기업이 벌어들이는 이익에 비해 주가가 몇 배나 비싸게 거래되고 있는가"**를 나타냅니다.
   - **퀀트 활용법:** 숫자가 낮을수록 기업이 내는 실적에 비해 주가가 저평가(저렴한 상태)되어 있음을 뜻합니다. 일반적으로 미국 빅테크 진영에서는 PER이 25 이하로 내려오면 밸류에이션 매력이 매우 높은 구간으로 진입했다고 평가합니다. *(최근 적자를 기록한 기업은 마이너스 PER이 되어 대시보드에 N/A 혹은 미래 예상치인 Forward PER로 대체 표시됩니다.)*

2. **PEG Ratio (Price/Earnings-to-Growth Ratio, 주가수익성장비율)**
   - **설명:** 현재의 PER을 기업의 향후 '이익 성장률'로 한 번 더 나눈 고도화된 가치 평가 지표입니다.
   - **퀀트 활용법:** 단순히 PER만 보면 성장하는 테크 기업들이 무조건 비싸 보일 수 있지만, 이 지표는 성장 속도 대비 주가가 싼지를 판별해 줍니다. 전설적인 투자자 피터 린치가 가장 신뢰한 지표로 유명하며, **보통 1 이하이면 성장성 대비 저평가**, 0.5 이하이면 극심한 저평가 상태(적극 매수 타이밍)로 해석합니다.

3. **RSI (Relative Strength Index, 상대강도지수)**
   - **설명:** 최근 14일간 주가의 상승 압력과 하락 압력 간의 상대적인 강도를 0~100 사이의 숫자로 나타낸 기술적 지표입니다.
   - **퀀트 활용법:** 주가가 단기적으로 과열되었는지 냉각되었는지 보여주는 대표적인 모멘텀 지표입니다. 일반적으로 **40 이하(특히 30 부근)는 '과매도(시장 소외 및 과도한 폭락)'** 구간으로 보며, 우량주를 억울하게 싼 가격에 주울 수 있는 역발상 분할 매수 타이밍이 됩니다. 반대로 70 이상은 단기 과열로 판단합니다.

4. **CCI (Commodity Channel Index, 상품채널지수)**
   - **설명:** 주가가 최근 20일 동안의 평균 가격(이동평균선)으로부터 얼마나 멀리 떨어져 있는지 변동성과 이격도를 측정하는 지표입니다.
   - **해석:** RSI보다 주가 움직임에 훨씬 민감하게 반응하여 빠르게 타점을 잡아줍니다. 일반적으로 **-100 이하로 떨어지면 주가가 평균 가격보다 비정상적으로 급락한 '극단적 과매도 상태'**를 뜻하므로, 단기 기술적 반등이나 진바닥 타점을 정교하게 노릴 때 유용합니다.

---

💡 **마스터 스크리너 활용 치트키:**
이 대시보드는 **"이익 성장성도 탄탄하고(낮은 PER/PEG), 체급도 거대한 대형 우량주인데, 최근 시장의 쏠림이나 일시적 악재로 인해 주가가 과도하게 두들겨 맞은(낮은 RSI/CCI 및 깊은 낙폭) 진흙 속의 진주"**를 포착하는 데 최적화되어 있습니다. 스크리너 탭과 백테스팅 탭을 오가며 이 지표들이 바닥을 칠 때의 위력을 직접 검증해 보세요!
""")
