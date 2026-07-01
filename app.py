import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Target 종목 리스트
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMD", "NVDA", "TSLA", "AMZN", "META", "NFLX", "INTC",
    "QCOM", "ADBE", "CRM", "ORCL", "SMCI", "AVGO", "TXN", "MU", "MRVL", "CDNS"
]

# 🌟 실시간 원/달러 환율을 가져오는 함수 (가장 최신 환율 반영)
def get_current_usd_krw():
    try:
        usd_krw_data = yf.download("USDKRW=X", period="1d", interval="1m", progress=False)
        if not usd_krw_data.empty:
            if isinstance(usd_krw_data.columns, pd.MultiIndex):
                usd_krw_data.columns = usd_krw_data.columns.get_level_values(0)
            return float(usd_krw_data["Close"].values.flatten()[-1])
    except Exception:
        pass
    return 1400.0  # 만약 환율 API 호출 실패 시 방어용 기본값 세팅

# 개별 종목 데이터 수집 및 지표 계산 워커 함수
def process_single_ticker(ticker, exchange_rate):
    try:
        # 안전하게 1년치 일봉 데이터 수집
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            return None
            
        # yfinance 최신 버전 MultiIndex 컬럼 깨부수기
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 1차원 Series화
        close_series = pd.Series(data["Close"].values.flatten(), index=data.index)
        high_series = pd.Series(data["High"].values.flatten(), index=data.index)
        low_series = pd.Series(data["Low"].values.flatten(), index=data.index)

        # RSI 계산 (14일)
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else np.nan

        # CCI 계산 (20일)
        TP = (high_series + low_series + close_series) / 3
        SMA_TP = TP.rolling(window=20).mean()
        MD = TP.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci_series = (TP - SMA_TP) / (0.015 * MD + 1e-9)
        cci = float(cci_series.dropna().iloc[-1]) if not cci_series.dropna().empty else np.nan

        # 고점 대비 낙폭 (MDD) 계산
        max_price = close_series.max()
        current_price_usd = close_series.iloc[-1]
        drawdown = ((current_price_usd - max_price) / max_price) * 100

        # 🌟 현재가 원화 환전
        current_price_krw = current_price_usd * exchange_rate

        # 야후 info 및 재무제표 API 호출
        per, peg, market_cap_krw, operating_income_krw_str = np.nan, np.nan, np.nan, "N/A"
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # 시가총액 달러 원본 가져와서 원화로 환전 후 '억원' 단위로 변환
            market_cap_usd = info.get("marketCap", np.nan)
            if market_cap_usd is not None and not pd.isna(market_cap_usd):
                # 시가총액(원) = 달러 * 환율 -> 억 원 단위로 절사 (/ 1억)
                market_cap_krw = (market_cap_usd * exchange_rate) / 100_000_000
            
            per = info.get("trailingPE", np.nan)
            peg = info.get("pegRatio", np.nan)
            
            # 작년 연간 영업이익 수집 및 원화 '억원' 단위 환전
            financials = ticker_obj.financials
            if "Operating Income" in financials.index:
                raw_income_usd = financials.loc["Operating Income"].iloc[0]
                if not pd.isna(raw_income_usd):
                    income_krw_mwon = (raw_income_usd * exchange_rate) / 100_000_000
                    operating_income_krw_str = f"₩{income_krw_mwon:,.0f}억원"
        except Exception:
            pass

        # 트레이딩뷰 로고 마스터 매핑
        tradingview_logos = {
            "AAPL": "apple", "MSFT": "microsoft", "GOOGL": "alphabet",             
            "AMD": "advanced-micro-devices", "NVDA": "nvidia", "TSLA": "tesla", 
            "AMZN": "amazon", "META": "meta-platforms", "NFLX": "netflix", 
            "INTC": "intel", "QCOM": "qualcomm", "ADBE": "adobe", 
            "CRM": "salesforce", "ORCL": "oracle", "SMCI": "super-micro-computer", 
            "AVGO": "broadcom", "TXN": "texas-instruments", "MU": "micron-technology", 
            "MRVL": "marvell-technology-inc", "CDNS": "cadence-design-systems"
        }
        
        logo_keyword = tradingview_logos.get(ticker, ticker.lower())
        logo_url = f"https://s3-symbol-logo.tradingview.com/{logo_keyword}--big.svg"

        return {
            "로고": logo_url,
            "Ticker": str(ticker),
            "시가총액": market_cap_krw,  # 정렬을 위해 원화 억 단위 정수형 데이터 주입
            "작년 영업이익": operating_income_krw_str,
            "PER (배)": float(per) if per is not None and not pd.isna(per) else "N/A",
            "PEG Ratio": float(peg) if peg is not None and not pd.isna(peg) else "N/A",
            "RSI (14)": rsi if not pd.isna(rsi) else 50.0,
            "CCI (20)": cci if not pd.isna(cci) else 0.0,
            "고점 대비 낙폭": float(drawdown),
            "현재가 (₩)": float(current_price_krw)
        }
    except Exception as e:
        print(f"❌ Error processing {ticker}: {str(e)}")
        return None

# 주식 스크리닝 함수 (멀티스레딩)
@st.cache_data(ttl=3600)
def run_fast_screener(selected_tickers, exchange_rate):
    screened_stocks = []
    
    progress_text = st.empty()
    progress_text.text("🚀 HTS급 병렬 알고리즘 엔진 가동 중...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_single_ticker, ticker, exchange_rate): ticker for ticker in selected_tickers}
        
        for future in as_completed(future_to_ticker):
            result = future.result()
            if result is not None:
                screened_stocks.append(result)
                
    progress_text.empty()
    return screened_stocks

# Streamlit UI 세팅
st.set_page_config(layout="wide", page_title="프로 퀀트 마스터 스크리너")
st.title("⚡ 부자되자")

# 🌟 먼저 실시간 환율을 동적으로 받아옵니다.
current_rate = get_current_usd_krw()

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.write(f"⏱️ 마지막 업데이트: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 💵 **실시간 원/달러 환율:** `{current_rate:,.2f} 원` 반영 중")
with col2:
    if st.button("수동 새로고침", help="캐시를 지우고 데이터를 새로 가져옵니다."):
        st.cache_data.clear()
        st.rerun()

st.write("시가총액이 높은 순서대로 원화 환산 퀀트 지표를 실시간 정렬하여 출력합니다.")

# 종목 선택
selected_tickers = st.multiselect(
    "스크리닝할 종목을 선택하세요:",
    options=tickers,
    default=tickers
)

if not selected_tickers:
    st.warning("최소한 하나의 종목을 선택해야 합니다.")
else:
    with st.spinner("야후 파이낸스 및 환율 허브 고속 연결 중..."):
        results = results = run_fast_screener(selected_tickers, current_rate)

    if results:
        results_df = pd.DataFrame(results)
        
        # 시가총액(원화 억 단위) 기준 내림차순 정렬
        results_df = results_df.sort_values(by="시가총액", ascending=False, na_position="last")
        
        st.metric(label="📊 수집된 우량주 개수", value=f"{len(results_df)} 개 종목")
        st.subheader("📝 대형주 순 정렬 원화 퀀트 대시보드")
        
        # 안전한 데이터프레임 빌드 및 포맷 세팅
        st.dataframe(
            results_df,
            column_config={
                "로고": st.column_config.ImageColumn("아이콘", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker"),
                "시가총액": st.column_config.NumberColumn("시가총액 (억원)", format="₩ %,.0f억원", help="실시간 환율 반영 원화 시가총액 (억 단위)"),
                "작년 영업이익": st.column_config.Column("작년 영업이익", help="가장 최근 연간 재무제표 기준 원화 환산 영업이익"),
                "PER (배)": st.column_config.Column("PER (배)"), 
                "PEG Ratio": st.column_config.Column("PEG Ratio"),
                "RSI (14)": st.column_config.NumberColumn("RSI (14)", format="%.2f"),
                "CCI (20)": st.column_config.NumberColumn("CCI (20)", format="%.2f"),
                "고점 대비 낙폭": st.column_config.NumberColumn("고점 대비 낙폭", format="%.2f%%"),
                "현재가 (₩)": st.column_config.NumberColumn("현재가 (원화)", format="₩ %,.0f원")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("데이터 수집에 실패했습니다. 야후 서버와의 일시적 통신 문제일 수 있으니 잠시 후 새로고침을 눌러보세요.")

# 대시보드 최하단 용어 해설 가이드 리빌딩
st.markdown("---")
st.markdown("""
### 📊 스크리너 용어 및 퀀트 지표 완전 정복 Guide (원화 환산 버전)
테이블에 표시되는 각 용어의 정의와 퀀트 투자 관점에서의 핵심 활용법입니다.

1. **Ticker (티커) / 현재가 (원화)**
   - 미국 주식 고유 식별 알파벳 약어와 실시간 환율을 반영하여 소수점 없이 원화(`₩`)로 정밀 변환한 1주당 주가입니다.

2. **시가총액 (억원)**
   - **설명:** 실시간 환율을 기반으로 기업의 총 가치를 대한민국 투자자에게 가장 친숙한 **'억 원(₩)'** 단위로 환산했습니다. 
   - **정렬 규칙:** 시가총액 억원 규모가 거대한 공룡 주도주(예: 3,000조 원이 넘는 마이크로소프트, 애플 등)가 최상단에 자동으로 정렬됩니다.

3. **작년 영업이익**
   - **설명:** 기업이 1년 동안 순수하게 주력 비즈니스를 통해 벌어들인 총 원화 기준 이익입니다. 기업의 기초 체력을 가장 정직하게 증명하는 재무 데이터로, 이 역시 보기 편하게 '억원' 단위로 표시됩니다.

4. **PER (주가수익비율) & PEG Ratio (주가수익성장비율)**
   - 비율 지표는 원화/달러화와 관계없이 동일하게 유지됩니다. 주가 대비 이익 체력(PER)과 기업의 미래 이익 성장세 대비 주가 수준(PEG)을 봅니다. **PEG가 1 이하인 기업**은 성장성 대비 매우 저렴한 상태입니다.

5. **RSI (14) & CCI (20) & 고점 대비 낙폭**
   - 단기 주가의 낙폭과 과매도(시장 소외) 강도를 잡아내는 필살 지표들입니다. 시가총액 서열이 높은 초우량주인데 유독 RSI가 40 이하로 내려가거나 1년 전 고점 대비 낙폭이 유독 깊다면 밸류에이션 매력도가 한층 극대화된 상태로 평가할 수 있습니다.
""")
