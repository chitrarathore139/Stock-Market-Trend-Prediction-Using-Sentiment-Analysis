import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def safe_download(ticker: str, start, end) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        return flatten_columns(df)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def get_stock_summary(ticker: str) -> dict:
    """Return a compact summary dict for chatbot responses."""
    df = safe_download(ticker, "2023-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    if df.empty:
        return {}
    latest   = df["Close"].iloc[-1]
    prev     = df["Close"].iloc[-2]
    change   = latest - prev
    pct      = (change / prev) * 100
    high52   = df["Close"].rolling(252).max().iloc[-1]
    low52    = df["Close"].rolling(252).min().iloc[-1]
    ma50     = df["Close"].rolling(50).mean().iloc[-1]
    ma200    = df["Close"].rolling(200).mean().iloc[-1]
    returns  = df["Close"].pct_change().dropna()
    ann_ret  = returns.mean() * 252 * 100
    ann_vol  = returns.std() * np.sqrt(252) * 100
    sharpe   = ann_ret / ann_vol if ann_vol != 0 else 0
    trend    = "bullish 📈" if ma50 > ma200 else "bearish 📉"
    return dict(ticker=ticker.upper(), latest=latest, change=change, pct=pct,
                high52=high52, low52=low52, ma50=ma50, ma200=ma200,
                ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, trend=trend, df=df)


def build_response(query: str) -> tuple[str, object]:
    """Return (text_response, optional_plotly_figure)."""
    q = query.lower()
    fig = None

    for ticker in ["aapl", "tsla"]:
        if ticker in q:
            s = get_stock_summary(ticker)
            if not s:
                return f"Could not fetch data for {ticker.upper()}.", None

            
            if any(w in q for w in ["price", "pricing", "data", "performance",
                                     "return", "stock data", "how is", "how has"]):
                text = (
                    f"### {s['ticker']} Pricing Summary\n"
                    f"- **Latest Close:** ${s['latest']:.2f}  "
                    f"({'▲' if s['change']>=0 else '▼'} ${abs(s['change']):.2f} / {s['pct']:+.2f}% today)\n"
                    f"- **52-Week High:** ${s['high52']:.2f}\n"
                    f"- **52-Week Low:** ${s['low52']:.2f}\n"
                    f"- **50-day MA:** ${s['ma50']:.2f}\n"
                    f"- **200-day MA:** ${s['ma200']:.2f}\n"
                    f"- **Annual Return:** {s['ann_ret']:.2f}%\n"
                    f"- **Annual Volatility:** {s['ann_vol']:.2f}%\n"
                    f"- **Sharpe Ratio:** {s['sharpe']:.4f}\n"
                    f"- **Trend Signal:** {s['trend']} (50MA vs 200MA)"
                )
                df_plot = s["df"].copy()
                ma50  = df_plot["Close"].rolling(50).mean()
                ma200 = df_plot["Close"].rolling(200).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"],
                                         name="Close", line=dict(color="#636EFA", width=1.5)))
                fig.add_trace(go.Scatter(x=df_plot.index, y=ma50,
                                         name="50-day MA", line=dict(color="#EF553B", width=1.2)))
                fig.add_trace(go.Scatter(x=df_plot.index, y=ma200,
                                         name="200-day MA", line=dict(color="#00CC96", width=1.2)))
                fig.update_layout(title=f"{s['ticker']} – Price & Moving Averages",
                                   hovermode="x unified",
                                   xaxis_title="Date", yaxis_title="Price (USD)",
                                   legend=dict(orientation="h", y=1.02))
                return text, fig

            
            if any(w in q for w in ["predict", "prediction", "forecast", "future",
                                     "lstm", "model", "trend"]):
                df_raw   = s["df"].copy().reset_index()
                close    = pd.DataFrame(df_raw["Close"])
                split    = int(len(close) * 0.70)
                train_df = close.iloc[:split]
                test_df  = close.iloc[split:]

                if len(test_df) < 100:
                    return (f"Not enough historical data to run LSTM predictions for "
                            f"{s['ticker']}. Try a ticker with more history."), None

                scaler   = MinMaxScaler(feature_range=(0, 1))
                scaler.fit_transform(train_df)
                past_100 = train_df.tail(100)
                final    = pd.concat([past_100, test_df], ignore_index=True)
                inp      = scaler.transform(final)
                x_t, y_t = [], []
                for i in range(100, inp.shape[0]):
                    x_t.append(inp[i-100:i])
                    y_t.append(inp[i, 0])
                x_t = np.array(x_t)
                y_t = np.array(y_t)

                # Check model file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, "stock_sentiment_model.pt.h5")
                if not os.path.exists(model_path) or load_model is None:
                    # Fall back to a simple rolling-mean forecast for demo
                    scale_f  = 1.0 / scaler.scale_[0]
                    y_actual = y_t * scale_f
                    # naive: smooth with rolling mean as "predicted"
                    y_naive  = pd.Series(y_actual).rolling(5, min_periods=1).mean().values
                    text = (
                        f"### {s['ticker']} Trend Prediction (Demo Mode)\n"
                        f"⚠️ The trained LSTM model file was not found. "
                        f"Showing a **rolling-average baseline** instead.\n\n"
                        f"- **Current Trend:** {s['trend']}\n"
                        f"- **50-day MA:** ${s['ma50']:.2f}\n"
                        f"- **200-day MA:** ${s['ma200']:.2f}\n"
                        f"- **Sharpe Ratio:** {s['sharpe']:.4f}\n\n"
                        f"Place `stock_sentiment_model.pt.h5` next to `streamlit_app.py` "
                        f"to enable full LSTM predictions."
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_actual, name="Actual Price",
                                             line=dict(color="#636EFA", width=1.5)))
                    fig.add_trace(go.Scatter(y=y_naive, name="Rolling-Mean Baseline",
                                             line=dict(color="#EF553B", width=1.5, dash="dot")))
                    fig.update_layout(title=f"{s['ticker']} – Actual vs Baseline",
                                       xaxis_title="Days (test set)",
                                       yaxis_title="Price (USD)",
                                       hovermode="x unified",
                                       legend=dict(orientation="h", y=1.02))
                    return text, fig

                model   = load_model(model_path)
                y_pred  = model.predict(x_t, verbose=0)
                sf      = 1.0 / scaler.scale_[0]
                y_pred  = y_pred.flatten() * sf
                y_t     = y_t * sf

                text = (
                    f"### {s['ticker']} LSTM Prediction\n"
                    f"- **Current Trend:** {s['trend']}\n"
                    f"- **Latest Close:** ${s['latest']:.2f}\n"
                    f"- **Sharpe Ratio:** {s['sharpe']:.4f}"
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_t, name="Actual Price",
                                         line=dict(color="#636EFA", width=1.5)))
                fig.add_trace(go.Scatter(y=y_pred, name="Predicted Price",
                                         line=dict(color="#EF553B", width=1.5, dash="dot")))
                fig.update_layout(title=f"{s['ticker']} – LSTM Prediction vs Actual",
                                   xaxis_title="Days (test set)",
                                   yaxis_title="Price (USD)",
                                   hovermode="x unified",
                                   legend=dict(orientation="h", y=1.02))
                return text, fig

            
            text = (
                f"### {s['ticker']} Quick Overview\n"
                f"- **Latest Close:** ${s['latest']:.2f} ({s['pct']:+.2f}% today)\n"
                f"- **52-Week High:** ${s['high52']:.2f}\n"
                f"- **52-Week Low:** ${s['low52']:.2f}\n"
                f"- **Trend Signal:** {s['trend']}"
            )
            return text, None

    return ("I can answer questions about **AAPL** and **TSLA**. "
            "Try one of the suggested questions below! 👇"), None


st.set_page_config(
    page_title="Sentiment-Aware Market Trend Tracker",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #a0aec0;
    margin-bottom: 1.2rem;
}
.badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-right: 8px;
    margin-bottom: 12px;
}
</style>

<div class="hero-title">📡 Sentiment-Aware Market Trend Tracker</div>
<div class="hero-sub">
    Tracks trending topics from social media · Analyzes public sentiment in real time ·
    Displays insights through interactive dashboards
</div>
<span class="badge">🐦 Twitter / X Sentiment</span>
<span class="badge">📈 Live Stock Data</span>
<span class="badge">🤖 LSTM Predictions</span>
<span class="badge">📊 Interactive Dashboards</span>
<hr style="border-color:#1e293b; margin-top:1rem; margin-bottom:1.5rem;">
""", unsafe_allow_html=True)

st.title("📊 Stock Explorer")

with st.sidebar:
    st.header("🔎 App 1 – Stock Explorer")
    ticker     = st.text_input("Ticker Symbol", placeholder="e.g. AAPL, TSLA")
    start_date = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp.today())

if ticker:
    if start_date >= end_date:
        st.warning("⚠️ Start date must be before end date.")
    else:
        data = safe_download(ticker, start_date, end_date)
        if data.empty:
            st.warning("No data found. Check the ticker symbol and date range.")
        else:
            fig = px.line(data, x=data.index, y="Close",
                          title=f"{ticker.upper()} – Closing Price",
                          labels={"Close": "Price (USD)", "Date": ""})
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, key="fig_main")

            tab_price, tab_pred = st.tabs(["📋 Pricing Data", "🔮 Predictions"])

            with tab_price:
                st.header("Price Movement")
                data2 = data.copy()
                data2["% Change"] = data2["Close"].pct_change() * 100
                data2.dropna(inplace=True)
                col1, col2, col3 = st.columns(3)
                annual_return = data2["% Change"].mean() * 252
                stdev  = data2["% Change"].std() * np.sqrt(252)
                sharpe = annual_return / stdev if stdev != 0 else 0
                col1.metric("Annual Return",             f"{annual_return:.2f}%")
                col2.metric("Annual Std Dev",            f"{stdev:.2f}%")
                col3.metric("Risk-Adjusted Return",      f"{sharpe:.4f}")
                st.dataframe(data2.tail(100), use_container_width=True)

            with tab_pred:
                st.header("Price Chart")
                st.plotly_chart(fig, use_container_width=True, key="fig_tab")
                st.info("ℹ️ LSTM predictions are shown in App 2 below.")
else:
    st.info("👈 Enter a ticker symbol in the sidebar to get started.")

st.divider()
st.title("🤖 Stock Trend Prediction (LSTM Model)")

col_left, col_right = st.columns([1, 2])
with col_left:
    user_input  = st.text_input("Enter Stock Ticker for Prediction", "AAPL")
    train_start = st.date_input("Training Start", value=pd.Timestamp("2011-02-01"), key="train_start")
    train_end   = st.date_input("Training End",   value=pd.Timestamp("2019-12-31"), key="train_end")

df = safe_download(user_input, train_start, train_end)
if df.empty:
    st.warning("Could not load data. Please check the ticker.")
    st.stop()

df = df.reset_index().dropna()
with col_right:
    st.subheader(f"Data Summary  ({train_start} → {train_end})")
    st.dataframe(df.describe(), use_container_width=True)

st.subheader("Closing Price with Moving Averages")
ma100  = df["Close"].rolling(100).mean()
ma200  = df["Close"].rolling(200).mean()
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["Close"],  name="Close",      line=dict(color="#636EFA", width=1)))
fig_ma.add_trace(go.Scatter(x=df["Date"], y=ma100,        name="100-day MA", line=dict(color="#EF553B", width=1.5)))
fig_ma.add_trace(go.Scatter(x=df["Date"], y=ma200,        name="200-day MA", line=dict(color="#00CC96", width=1.5)))
fig_ma.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Price (USD)",
                     legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig_ma, use_container_width=True, key="fig_ma")

st.subheader("Prediction vs Original Price")
data_training = pd.DataFrame(df["Close"][0: int(len(df) * 0.70)])
data_testing  = pd.DataFrame(df["Close"][int(len(df) * 0.70):])

if len(data_testing) < 100:
    st.warning("Not enough test data. Choose a wider date range.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(data_training)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "stock_sentiment_model.pt.h5")

if not os.path.exists(model_path):
    st.warning(
        "⚠️ **Model file not found** – showing rolling-average baseline instead.\n\n"
        f"Place `stock_sentiment_model.pt.h5` in:\n`{script_dir}`"
    )
    past_100 = data_training.tail(100)
    final_df = pd.concat([past_100, data_testing], ignore_index=True)
    inp      = scaler.transform(final_df)
    x_t, y_t = [], []
    for i in range(100, inp.shape[0]):
        x_t.append(inp[i-100:i])
        y_t.append(inp[i, 0])
    y_t      = np.array(y_t)
    sf       = 1.0 / scaler.scale_[0]
    y_actual = y_t * sf
    y_naive  = pd.Series(y_actual).rolling(5, min_periods=1).mean().values
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_actual, name="Actual Price",           line=dict(color="#636EFA", width=1.5)))
    fig_pred.add_trace(go.Scatter(y=y_naive,  name="Rolling-Mean Baseline",  line=dict(color="#EF553B", width=1.5, dash="dot")))
    fig_pred.update_layout(xaxis_title="Days (test set)", yaxis_title="Price (USD)",
                            hovermode="x unified", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_pred, use_container_width=True, key="fig_pred")
    st.stop()

if load_model is None:
    st.error("TensorFlow / Keras not installed. Run: `pip install tensorflow`")
    st.stop()

with st.spinner("Loading model and generating predictions…"):
    model    = load_model(model_path)
    past_100 = data_training.tail(100)
    final_df = pd.concat([past_100, data_testing], ignore_index=True)
    inp      = scaler.transform(final_df)
    x_t, y_t = [], []
    for i in range(100, inp.shape[0]):
        x_t.append(inp[i-100:i])
        y_t.append(inp[i, 0])
    x_t    = np.array(x_t)
    y_t    = np.array(y_t)
    y_pred = model.predict(x_t, verbose=0)
    sf     = 1.0 / scaler.scale_[0]
    y_pred = y_pred.flatten() * sf
    y_t    = y_t * sf

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(y=y_t,    name="Actual Price",    line=dict(color="#636EFA", width=1.5)))
fig_pred.add_trace(go.Scatter(y=y_pred, name="Predicted Price", line=dict(color="#EF553B", width=1.5, dash="dot")))
fig_pred.update_layout(xaxis_title="Days (test set)", yaxis_title="Price (USD)",
                        hovermode="x unified", legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig_pred, use_container_width=True, key="fig_pred")

st.divider()
st.title("💬 Stock Chatbot")
st.caption("Ask about AAPL or TSLA pricing data and predictions.")

# Suggested questions
SUGGESTIONS = [
    "📈 Show me AAPL pricing data",
    "🔮 What is the AAPL prediction?",
    "📈 Show me TSLA pricing data",
    "🔮 What is the TSLA prediction?",
]

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fig"):
            st.plotly_chart(msg["fig"], use_container_width=True,
                            key=f"chat_fig_{msg['id']}")

# Suggested buttons
st.markdown("**Suggested questions:**")
cols = st.columns(4)
for i, suggestion in enumerate(SUGGESTIONS):
    if cols[i].button(suggestion, key=f"sugg_{i}"):
        st.session_state.pending_query = suggestion

# Chat input
user_query = st.chat_input("Ask about AAPL or TSLA…")
if user_query:
    st.session_state.pending_query = user_query

# Process pending query
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = ""

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query, "fig": None, "id": len(st.session_state.messages)})

    # Generate response
    with st.spinner("Fetching data…"):
        response_text, response_fig = build_response(query)

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "fig": response_fig,
        "id": len(st.session_state.messages)
    })

    st.rerun()