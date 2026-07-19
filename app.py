import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# ============================================================
# Settings
# ============================================================
st.set_page_config(
    page_title="تحلیلگر روند و ریسک بیت‌کوین",
    page_icon="📈",
    layout="wide",
)

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

PERIOD_OPTIONS = {"6 ماه": 182, "1 سال": 365, "2 سال": 730, "3 سال": 1095}
SEQ_LEN_OPTIONS = {"کوتاه (10 روز)": 10, "متوسط (15 روز)": 15, "بلند (20 روز)": 20}
MC_DROPOUT_SAMPLES = 20

# ============================================================
# Data fetching (cached so re-running charts doesn't re-download)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bitcoin_from_yahoo(days_back: int) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(start=start_date, end=end_date, interval="1d", timeout=15)
    if hist.empty:
        raise RuntimeError("داده‌ای از Yahoo Finance دریافت نشد - اتصال اینترنت را بررسی کنید")
    df_new = pd.DataFrame({
        "Date": hist.index.strftime("%Y-%m-%d"),
        "Bitcoin": hist["Close"].values,
    }).reset_index(drop=True)
    return df_new


# ============================================================
# Core ML logic
#   - Predicts LOG-RETURNS (not raw price) to avoid the model
#     "cheating" by echoing yesterday's price.
#   - Scaler is fit ONLY on the training slice (no leakage).
#   - A naive baseline ("tomorrow = today") is always computed so
#     the MAPE can be judged against a trivial benchmark.
#   - Future uncertainty is estimated with MC-Dropout rollouts.
# ============================================================
def train_and_predict(df_snapshot, seq_len_local, days, keras_progress_cb=None):
    clean_df = df_snapshot.dropna(subset=["Bitcoin"]).reset_index(drop=True)
    prices_full = clean_df["Bitcoin"].values.astype(float)
    dates_full = clean_df["Date"].values

    if len(prices_full) < seq_len_local + 30:
        raise RuntimeError("داده کافی برای این طول دنباله وجود ندارد — بازه تاریخی بیشتری انتخاب کنید")

    log_prices = np.log(prices_full)
    returns = np.diff(log_prices)
    num_returns = len(returns)

    split_returns = int(0.8 * num_returns)
    split_returns = max(split_returns, seq_len_local + 5)
    split_returns = min(split_returns, num_returns - 5)
    if split_returns <= seq_len_local:
        raise RuntimeError("داده کافی برای ساخت مجموعه آموزش/آزمون در این طول دنباله وجود ندارد")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(returns[:split_returns].reshape(-1, 1))
    scaled_returns = scaler.transform(returns.reshape(-1, 1)).flatten()

    X, y = [], []
    for k in range(0, num_returns - seq_len_local):
        X.append(scaled_returns[k:k + seq_len_local])
        y.append(scaled_returns[k + seq_len_local])
    X = np.array(X).reshape(-1, seq_len_local, 1)
    y = np.array(y)

    train_end_k = split_returns - seq_len_local
    train_end_k = max(1, min(train_end_k, len(X) - 1))
    X_train, X_test = X[:train_end_k], X[train_end_k:]
    y_train, y_test = y[:train_end_k], y[train_end_k:]

    if len(X_test) == 0:
        raise RuntimeError("داده کافی برای ساخت مجموعه آزمون باقی نمانده است")

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(seq_len_local, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    callbacks = [early_stop]
    if keras_progress_cb is not None:
        callbacks.append(keras_progress_cb)

    model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.15,
              callbacks=callbacks, verbose=0)

    y_pred_scaled = model.predict(X_test, verbose=0)
    predicted_return = scaler.inverse_transform(y_pred_scaled).flatten()

    test_k = np.arange(train_end_k, len(X))
    predict_day_idx = test_k + seq_len_local + 1
    price_before = prices_full[test_k + seq_len_local]
    actual_next = prices_full[predict_day_idx]
    predicted_next = price_before * np.exp(predicted_return)
    naive_next = price_before

    nonzero = actual_next != 0
    mape = float(np.mean(np.abs((actual_next[nonzero] - predicted_next[nonzero]) / actual_next[nonzero])) * 100) if np.any(nonzero) else 0.0
    naive_mape = float(np.mean(np.abs((actual_next[nonzero] - naive_next[nonzero]) / actual_next[nonzero])) * 100) if np.any(nonzero) else 0.0

    actual_dir = np.sign(actual_next - price_before)
    pred_dir = np.sign(predicted_next - price_before)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

    test_dates = dates_full[predict_day_idx]

    mc_paths = np.zeros((MC_DROPOUT_SAMPLES, days))
    base_seq = scaled_returns[-seq_len_local:].reshape(1, seq_len_local, 1)
    for m in range(MC_DROPOUT_SAMPLES):
        seq_iter = base_seq.copy()
        price_iter = prices_full[-1]
        for d in range(days):
            pred_scaled = model(seq_iter, training=True).numpy()  # dropout stays ON -> stochastic
            pred_ret = scaler.inverse_transform(pred_scaled)[0, 0]
            price_iter = price_iter * np.exp(pred_ret)
            mc_paths[m, d] = price_iter
            new_val = pred_scaled.reshape(1, 1, 1)
            seq_iter = np.append(seq_iter[:, 1:, :], new_val, axis=1)

    future_mean = mc_paths.mean(axis=0)
    future_std = mc_paths.std(axis=0)

    return {
        "mape": mape, "naive_mape": naive_mape, "dir_acc": dir_acc,
        "test_dates": test_dates, "test_x": predict_day_idx,
        "test_actual": actual_next, "test_predicted": predicted_next, "test_naive": naive_next,
        "hist_prices": prices_full, "hist_dates": dates_full,
        "future_prices": future_mean, "future_std": future_std,
        "last_date": clean_df["Date"].iloc[-1], "days": days, "seq_len": seq_len_local,
    }


class StreamlitProgress(Callback):
    """Feeds Keras training progress into a Streamlit progress bar."""
    def __init__(self, bar, text, total_epochs=80):
        super().__init__()
        self.bar = bar
        self.text = text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        frac = min((epoch + 1) / self.total_epochs, 1.0)
        self.bar.progress(frac)
        loss = logs.get("loss", 0) if logs else 0
        self.text.caption(f"دوره آموزش {epoch + 1}/{self.total_epochs} — loss: {loss:.5f}")


# ============================================================
# UI — Sidebar
# ============================================================
st.title("📈 تحلیلگر روند و ریسک بیت‌کوین")
st.caption(
    "این ابزار برای **تحلیل روند تاریخی و کمک به تصمیم‌گیری** طراحی شده، نه پیش‌بینی قطعی. "
    "بازارهای مالی به‌شدت نویزدار و تحت تأثیر عوامل غیرقابل پیش‌بینی هستند؛ به همین دلیل هر پیش‌بینی "
    "همراه با بازه عدم‌قطعیت و مقایسه با یک مدل ساده پایه (baseline) نمایش داده می‌شود."
)

with st.sidebar:
    st.header("⚙️ تنظیمات")
    period_label = st.selectbox("بازه تاریخی داده", list(PERIOD_OPTIONS.keys()), index=2)
    seq_label = st.radio("طول دنباله ورودی مدل", list(SEQ_LEN_OPTIONS.keys()), index=0)
    days = st.slider("تعداد روزهای آینده برای تحلیل", min_value=1, max_value=50, value=10)

    st.divider()
    fetch_clicked = st.button("📡 دریافت داده از Yahoo Finance", use_container_width=True)
    run_clicked = st.button("🚀 اجرای تحلیل (آموزش مدل)", use_container_width=True, type="primary")

    with st.expander("ℹ️ توضیح معیارها برای داور"):
        st.markdown(
            "- **MAPE مدل**: میانگین درصد خطای پیش‌بینی مدل\n"
            "- **MAPE پایه (naive)**: خطای حالتی که فرض می‌کنیم قیمت فردا برابر امروز است — معیار مقایسه\n"
            "- **دقت جهت‌گیری**: درصد مواقعی که مدل جهت درست (صعود/نزول) را تشخیص داده\n"
            "- **باند عدم‌قطعیت (MC-Dropout)**: بازه‌ای که مدل نسبت به آن مطمئن نیست، با ۲۰ بار اجرای "
            "تصادفی مدل تخمین زده می‌شود\n"
            "- مدل روی **بازده لگاریتمی (log-return)** آموزش دیده، نه قیمت خام، تا مجبور شود واقعاً "
            "الگو یاد بگیرد نه فقط قیمت روز قبل را تکرار کند."
        )

# ============================================================
# Session state
# ============================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None

# ============================================================
# Fetch data
# ============================================================
if fetch_clicked:
    with st.spinner("در حال دریافت داده..."):
        try:
            st.session_state.df = fetch_bitcoin_from_yahoo(PERIOD_OPTIONS[period_label])
            st.session_state.result = None
            st.success(f"✅ {len(st.session_state.df)} روز داده دریافت شد")
        except Exception as e:
            st.error(f"❌ خطا در دریافت داده: {e}")

if st.session_state.df is not None:
    last_row = st.session_state.df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("آخرین قیمت (USD)", f"${last_row['Bitcoin']:,.0f}")
    c2.metric("تعداد روزهای داده", len(st.session_state.df))
    c3.metric("آخرین تاریخ", last_row["Date"])

# ============================================================
# Run analysis
# ============================================================
if run_clicked:
    if st.session_state.df is None:
        st.error("❌ ابتدا داده را دریافت کنید")
    else:
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        cb = StreamlitProgress(progress_bar, progress_text, total_epochs=80)
        try:
            with st.spinner("در حال آموزش مدل LSTM... (ممکن است ۱-۳ دقیقه طول بکشد)"):
                result = train_and_predict(
                    st.session_state.df, SEQ_LEN_OPTIONS[seq_label], days, keras_progress_cb=cb
                )
            st.session_state.result = result
            progress_bar.empty()
            progress_text.empty()
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.error(f"❌ خطا: {e}")

# ============================================================
# Results
# ============================================================
result = st.session_state.result
if result is not None:
    st.divider()
    st.subheader("📊 نتایج تحلیل")

    beat_baseline = result["mape"] < result["naive_mape"]
    m1, m2, m3 = st.columns(3)
    m1.metric("MAPE مدل", f"{result['mape']:.2f}%")
    m2.metric("MAPE پایه (naive)", f"{result['naive_mape']:.2f}%",
              delta=f"{result['naive_mape'] - result['mape']:.2f}%",
              delta_color="normal" if beat_baseline else "inverse")
    m3.metric("دقت جهت‌گیری", f"{result['dir_acc']:.1f}%")

    if beat_baseline:
        st.success("✅ مدل بهتر از مدل پایه ساده عمل کرده است")
    else:
        st.warning("⚠️ مدل نتوانسته بهتر از مدل پایه ساده (naive) عمل کند — این یک محدودیت شناخته‌شده در پیش‌بینی بازارهای مالی نویزدار است")

    # ---- Chart 1: test set actual vs predicted vs naive ----
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=result["test_dates"], y=result["test_actual"],
                               mode="lines+markers", name="واقعی", line=dict(color="#3B82F6")))
    fig1.add_trace(go.Scatter(x=result["test_dates"], y=result["test_predicted"],
                               mode="lines+markers", name="پیش‌بینی مدل", line=dict(color="#EF4444")))
    fig1.add_trace(go.Scatter(x=result["test_dates"], y=result["test_naive"],
                               mode="lines", name="پایه ساده (دیروز)", line=dict(color="gray", dash="dash")))
    fig1.update_layout(title="مجموعه آزمون: واقعی در برابر پیش‌بینی و پایه ساده",
                        xaxis_title="تاریخ", yaxis_title="قیمت (USD)",
                        template="plotly_dark", height=420)
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Chart 2: historical + future with uncertainty band ----
    last_date_obj = datetime.strptime(result["last_date"], "%Y-%m-%d")
    future_dates = [(last_date_obj + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(result["days"])]
    future = result["future_prices"]
    std = result["future_std"]
    upper = future + 1.96 * std
    lower = future - 1.96 * std

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(result["hist_dates"]), y=result["hist_prices"],
                               mode="lines", name="تاریخی", line=dict(color="#22D3EE")))
    fig2.add_trace(go.Scatter(x=future_dates, y=list(upper), mode="lines",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=future_dates, y=list(lower), mode="lines",
                               fill="tonexty", fillcolor="rgba(251,146,60,0.25)",
                               line=dict(width=0), name="بازه اطمینان ۹۵٪"))
    fig2.add_trace(go.Scatter(x=future_dates, y=list(future), mode="lines+markers",
                               name="میانگین پیش‌بینی", line=dict(color="#F87171")))
    fig2.update_layout(title=f"روند تاریخی و تحلیل {result['days']} روز آینده (عدم‌قطعیت MC-Dropout)",
                        xaxis_title="تاریخ", yaxis_title="قیمت (USD)",
                        template="plotly_dark", height=420)
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Table of future predictions ----
    st.subheader("📅 جدول تحلیل روزهای آینده")
    table_df = pd.DataFrame({
        "تاریخ": future_dates,
        "میانگین پیش‌بینی (USD)": [f"${p:,.0f}" for p in future],
        "بازه ۹۵٪ اطمینان": [f"${lo:,.0f} - ${hi:,.0f}" for lo, hi in zip(lower, upper)],
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ---- Download report ----
    report_lines = [
        "=" * 60,
        "گزارش تحلیل روند بیت‌کوین",
        "=" * 60,
        f"تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"تعداد روزهای تحلیل‌شده: {result['days']}",
        f"طول دنباله: {result['seq_len']}",
        f"MAPE مدل: {result['mape']:.2f}%",
        f"MAPE پایه: {result['naive_mape']:.2f}%",
        f"دقت جهت‌گیری: {result['dir_acc']:.1f}%",
        "",
        "پیش‌بینی‌ها:",
    ]
    for d, p, lo, hi in zip(future_dates, future, lower, upper):
        report_lines.append(f"{d}: ${p:,.0f}  (بازه: ${lo:,.0f} - ${hi:,.0f})")
    report_lines.append("")
    report_lines.append("توجه: مدل روی بازده لگاریتمی آموزش دیده؛ پایه ساده = 'فردا برابر امروز'.")
    report_text = "\n".join(report_lines)

    st.download_button(
        "💾 دانلود گزارش (TXT)",
        data=report_text.encode("utf-8"),
        file_name=f"btc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )
else:
    st.info("برای شروع، ابتدا داده را دریافت کنید و سپس تحلیل را اجرا کنید.")
