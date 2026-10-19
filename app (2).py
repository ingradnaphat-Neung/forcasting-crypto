# -*- coding: utf-8 -*-
"""
Crypto Day-Trade Analyzer & Forecaster (BTC / ETH / XRP)
- ดึงข้อมูลแท่งเทียน (OHLCV) จาก GitLab / โฟลเดอร์ในเครื่อง / yfinance
- คำนวณอินดิเคเตอร์สำหรับเดย์เทรด (EMA, RSI, MACD, Bollinger, ATR, VWAP)
- พยากรณ์ทิศทางและผลตอบแทน N แท่งข้างหน้าด้วย Gradient Boosting
- Backtest แบบ walk-forward (train อดีต -> test อนาคต) พร้อมค่าธรรมเนียม
- สร้างแผนเทรด: จุดเข้า / Stop-loss / Take-profit / ขนาดไม้ ตาม ATR

⚠️ เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
"""
import io
import os
import urllib.parse
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, roc_auc_score

st.set_page_config(page_title="Crypto Day Trade", page_icon="📈", layout="wide")

COINS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "XRP": "XRP-USD"}
INTERVALS = ["5m", "15m", "30m", "1h"]
INTERVAL_DELTA = {"5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
                  "30m": timedelta(minutes=30), "1h": timedelta(hours=1)}
PANDAS_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
YF_PERIOD = {"5m": "60d", "15m": "60d", "30m": "60d", "1h": "365d"}
BARS_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24}
TZ = "Asia/Bangkok"


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def get_secret(key: str, default: str = "") -> str:
    """อ่านค่าจาก st.secrets ก่อน ถ้าไม่มีค่อยอ่านจาก environment variable"""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """แปลง DataFrame ให้เป็นรูปแบบมาตรฐาน: index=เวลา(Asia/Bangkok), คอลัมน์ open/high/low/close/volume"""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in ["timestamp", "datetime", "date", "time", "open_time"] if c in df.columns), None)
    if tcol is None:
        raise ValueError("ไม่พบคอลัมน์เวลา (ต้องมี timestamp / datetime / date / time / open_time)")
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.max() > 1e11 else "s"
        t = pd.to_datetime(t, unit=unit, utc=True)
    else:
        t = pd.to_datetime(t, utc=True, errors="coerce")
    df.index = pd.DatetimeIndex(t)

    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"ไฟล์ขาดคอลัมน์: {missing}")
    df = df[need].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    df.index = df.index.tz_convert(TZ)
    df.index.name = "time"
    return df


def resample_if_needed(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """ถ้าข้อมูลละเอียดกว่า timeframe ที่เลือก (เช่นไฟล์ 1m แต่เลือก 15m) จะรวมแท่งให้อัตโนมัติ"""
    if len(df) < 3:
        return df
    step = df.index.to_series().diff().median()
    target = INTERVAL_DELTA[interval]
    if step < target:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df = df.resample(PANDAS_RULE[interval], label="left", closed="left").agg(agg)
        df = df.dropna(subset=["close"])
    elif step > target:
        st.warning(f"ข้อมูลมีความละเอียด {step} ซึ่งหยาบกว่า timeframe {interval} ที่เลือก — ใช้ตามข้อมูลจริง")
    return df


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_from_gitlab(base_url: str, project: str, branch: str, file_path: str, _token: str) -> pd.DataFrame:
    """ดึงไฟล์ CSV จาก GitLab ผ่าน REST API (รองรับ repo private ด้วย Personal/Project Access Token)"""
    project = project.strip()
    if not project:
        raise ValueError("ยังไม่ได้กรอก GitLab Project (เช่น my-group/crypto-daytrade)")
    pid = project if project.isdigit() else urllib.parse.quote(project, safe="")
    fp = urllib.parse.quote(file_path.strip().lstrip("/"), safe="")
    url = f"{base_url.rstrip('/')}/api/v4/projects/{pid}/repository/files/{fp}/raw"
    headers = {"PRIVATE-TOKEN": _token} if _token else {}
    r = requests.get(url, params={"ref": branch}, headers=headers, timeout=30)
    if r.status_code == 404:
        raise FileNotFoundError(f"ไม่พบไฟล์ '{file_path}' บน branch '{branch}' (หรือ token ไม่มีสิทธิ์อ่าน repo)")
    if r.status_code in (401, 403):
        raise PermissionError("GitLab ปฏิเสธการเข้าถึง — ตรวจสอบ token (ต้องมีสิทธิ์ read_repository / read_api)")
    r.raise_for_status()
    return normalize_ohlcv(pd.read_csv(io.StringIO(r.text)))


@st.cache_data(ttl=300, show_spinner=False)
def load_from_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    return normalize_ohlcv(pd.read_csv(path))


@st.cache_data(ttl=120, show_spinner=False)
def load_from_yfinance(ticker: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=YF_PERIOD[interval], interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"yfinance ไม่คืนข้อมูลสำหรับ {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return normalize_ohlcv(df.reset_index())


# ----------------------------------------------------------------------------
# Indicators & features
# ----------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]

    d["ema9"] = c.ewm(span=9, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["bb_mid"], d["bb_up"], d["bb_low"] = mid, mid + 2 * std, mid - 2 * std
    d["bb_pctb"] = (c - d["bb_low"]) / (d["bb_up"] - d["bb_low"])
    d["bb_width"] = (d["bb_up"] - d["bb_low"]) / mid

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # VWAP รีเซ็ตทุกวัน (ตามเวลาไทย) — นิยมใช้ในเดย์เทรด
    tp = (h + l + c) / 3
    day = d.index.normalize()
    d["vwap"] = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    d["vwap"] = d["vwap"].fillna(c)
    return d


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    logc = np.log(d["close"])
    r1 = logc.diff()
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}"] = logc.diff(k)
    f["vol_10"] = r1.rolling(10).std()
    f["vol_30"] = r1.rolling(30).std()
    f["rsi"] = d["rsi"] / 100
    f["macd_hist_n"] = d["macd_hist"] / d["close"]
    f["bb_pctb"] = d["bb_pctb"]
    f["bb_width"] = d["bb_width"]
    f["atr_pct"] = d["atr"] / d["close"]
    f["ema9_21"] = d["ema9"] / d["ema21"] - 1
    f["close_ema50"] = d["close"] / d["ema50"] - 1
    f["vwap_dist"] = d["close"] / d["vwap"] - 1
    vmean, vstd = d["volume"].rolling(48).mean(), d["volume"].rolling(48).std()
    f["vol_z"] = (d["volume"] - vmean) / vstd.replace(0, np.nan)
    f["range_pct"] = (d["high"] - d["low"]) / d["close"]
    f["body_pct"] = (d["close"] - d["open"]) / d["open"]
    hour = d.index.hour + d.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = d.index.dayofweek
    return f.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Model, backtest, analysis
# ----------------------------------------------------------------------------
def _models():
    common = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=42)
    return HistGradientBoostingClassifier(**common), HistGradientBoostingRegressor(**common)


def backtest(close, prob, fee, prob_long, prob_short, allow_short, bars_per_year):
    pos = pd.Series(0.0, index=prob.index)
    pos[prob >= prob_long] = 1.0
    if allow_short:
        pos[prob <= prob_short] = -1.0
    nxt = close.pct_change().shift(-1).reindex(prob.index).fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * nxt - turnover * fee
    eq = (1 + strat).cumprod()
    bh = (1 + nxt).cumprod()
    dd = eq / eq.cummax() - 1
    active = strat[pos != 0]
    sd = strat.std()
    return {
        "equity": pd.DataFrame({"กลยุทธ์ (โมเดล)": eq, "Buy & Hold": bh}),
        "total_return": float(eq.iloc[-1] - 1),
        "bh_return": float(bh.iloc[-1] - 1),
        "max_dd": float(dd.min()),
        "sharpe": float(strat.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0,
        "trades": int(((pos != 0) & (pos != pos.shift())).sum()),
        "win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "exposure": float((pos != 0).mean()),
    }


@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame, horizon: int, test_frac: float, fee: float,
                 prob_long: float, allow_short: bool, bars_per_day: int) -> dict:
    d = add_indicators(df)
    X = build_features(d)
    fut = np.log(d["close"].shift(-horizon) / d["close"])

    warm = 60  # ตัดช่วงอุ่นเครื่องของอินดิเคเตอร์
    idx = X.index[warm:]
    labeled = idx[fut.loc[idx].notna()]
    if len(labeled) < 500:
        raise ValueError(f"ข้อมูลน้อยเกินไป ({len(labeled)} แท่ง) ต้องมีอย่างน้อย ~560 แท่ง")

    split = int(len(labeled) * (1 - test_frac))
    train_idx = labeled[: split - horizon]  # เว้นช่องว่าง (purge) กันข้อมูลอนาคตรั่ว
    test_idx = labeled[split:]
    y_cls = (fut > 0).astype(int)

    clf, reg = _models()
    clf.fit(X.loc[train_idx], y_cls.loc[train_idx])
    reg.fit(X.loc[train_idx], fut.loc[train_idx])
    p_test = pd.Series(clf.predict_proba(X.loc[test_idx])[:, 1], index=test_idx)
    r_test = pd.Series(reg.predict(X.loc[test_idx]), index=test_idx)

    y_true = y_cls.loc[test_idx]
    try:
        auc = float(roc_auc_score(y_true, p_test))
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_true, (p_test >= 0.5).astype(int))),
        "precision": float(precision_score(y_true, (p_test >= prob_long).astype(int), zero_division=0)),
        "auc": auc,
        "base_rate": float(y_true.mean()),
        "mae_ret": float(mean_absolute_error(fut.loc[test_idx], r_test)),
        "dir_acc_reg": float((np.sign(r_test) == np.sign(fut.loc[test_idx])).mean()),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    bt = backtest(d["close"], p_test, fee, prob_long, 1 - prob_long, allow_short, bars_per_day * 365)

    # เทรนใหม่ด้วยข้อมูลทั้งหมด แล้วพยากรณ์แท่งล่าสุด
    clf_f, reg_f = _models()
    clf_f.fit(X.loc[labeled], y_cls.loc[labeled])
    reg_f.fit(X.loc[labeled], fut.loc[labeled])
    last_x = X.iloc[[-1]]
    p_last = float(clf_f.predict_proba(last_x)[0, 1])
    r_last = float(reg_f.predict(last_x)[0])

    recent = pd.DataFrame({
        "ราคาปิด": d["close"].loc[test_idx],
        "prob ขึ้น": p_test,
        "คาดการณ์ผลตอบแทน %": r_test * 100,
        "ผลตอบแทนจริง %": fut.loc[test_idx] * 100,
    }).tail(15)

    return {"d": d, "metrics": metrics, "bt": bt, "p_last": p_last, "r_last": r_last, "recent": recent}


def make_signal(p: float, prob_long: float, allow_short: bool) -> str:
    if p >= prob_long:
        return "LONG"
    if allow_short and p <= 1 - prob_long:
        return "SHORT"
    return "WAIT"


SIGNAL_LABEL = {"LONG": "🟢 LONG (ซื้อ)", "SHORT": "🔴 SHORT (ขาย)", "WAIT": "⚪ WAIT (รอ)"}


def trade_plan(d, signal, capital, risk_pct, sl_mult, tp_mult):
    price = float(d["close"].iloc[-1])
    atr = float(d["atr"].iloc[-1])
    if signal == "WAIT":
        return None
    side = 1 if signal == "LONG" else -1
    sl = price - side * sl_mult * atr
    tp = price + side * tp_mult * atr
    risk_amount = capital * risk_pct
    qty = risk_amount / abs(price - sl) if abs(price - sl) > 0 else 0
    qty = min(qty, capital / price)  # ไม่ใช้เลเวอเรจเกินทุน
    return {"entry": price, "sl": sl, "tp": tp, "qty": qty, "notional": qty * price,
            "risk": qty * abs(price - sl), "rr": tp_mult / sl_mult}


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def price_chart(d: pd.DataFrame, n: int, coin: str, forecast=None) -> go.Figure:
    x = d.tail(n)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.12, 0.16, 0.17])
    fig.add_trace(go.Candlestick(x=x.index, open=x["open"], high=x["high"], low=x["low"],
                                 close=x["close"], name=coin), row=1, col=1)
    for col, color in [("ema9", "#f59e0b"), ("ema21", "#3b82f6"), ("vwap", "#a855f7")]:
        fig.add_trace(go.Scatter(x=x.index, y=x[col], name=col.upper(),
                                 line=dict(width=1.2, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["bb_up"], name="BB", line=dict(width=0.8, color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["bb_low"], showlegend=False,
                             line=dict(width=0.8, color="gray", dash="dot")), row=1, col=1)
    if forecast is not None:
        fig.add_trace(go.Scatter(x=[x.index[-1], forecast["time"]], y=[x["close"].iloc[-1], forecast["price"]],
                                 mode="lines+markers", name="พยากรณ์",
                                 line=dict(color="#ef4444", dash="dash", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[forecast["time"], forecast["time"]], y=[forecast["low"], forecast["high"]],
                                 mode="lines", name="ช่วงคาดการณ์", line=dict(color="#ef4444", width=6),
                                 opacity=0.3), row=1, col=1)

    colors = np.where(x["close"] >= x["open"], "#22c55e", "#ef4444")
    fig.add_trace(go.Bar(x=x.index, y=x["volume"], marker_color=colors, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["rsi"], name="RSI", line=dict(color="#8b5cf6", width=1.2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    hist_colors = np.where(x["macd_hist"] >= 0, "#22c55e", "#ef4444")
    fig.add_trace(go.Bar(x=x.index, y=x["macd_hist"], marker_color=hist_colors, name="MACD hist", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["macd"], name="MACD", line=dict(color="#0ea5e9", width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["macd_signal"], name="Signal", line=dict(color="#f97316", width=1)), row=4, col=1)

    fig.update_layout(height=820, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_yaxes(title_text="ราคา", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    return fig


def equity_chart(eq: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in eq.columns:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="มูลค่าพอร์ต (เริ่ม = 1)",
                      legend=dict(orientation="h", y=1.1))
    return fig


def fmt_price(p: float) -> str:
    return f"{p:,.4f}" if p < 10 else f"{p:,.2f}"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
SOURCES = ["GitLab (API)", "โฟลเดอร์ในเครื่อง (repo ที่ clone)", "yfinance (ออนไลน์)"]
default_src = get_secret("DATA_SOURCE", "gitlab").lower()
default_idx = {"gitlab": 0, "local": 1, "yfinance": 2}.get(default_src, 0)

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    source = st.radio("แหล่งข้อมูล", SOURCES, index=default_idx)
    interval = st.selectbox("Timeframe (เดย์เทรด)", INTERVALS, index=1)
    path_tpl = "data/{coin}_{interval}.csv"
    if source == SOURCES[0]:
        gl_base = st.text_input("GitLab URL", get_secret("GITLAB_URL", "https://gitlab.com"))
        gl_project = st.text_input("Project (group/repo หรือ ID)", get_secret("GITLAB_PROJECT", ""))
        gl_branch = st.text_input("Branch", get_secret("GITLAB_BRANCH", "main"))
        path_tpl = st.text_input("Path ไฟล์ใน repo", path_tpl)
        gl_token = get_secret("GITLAB_TOKEN", "")
        st.caption("🔒 ใช้ token จาก secrets/ENV" if gl_token else "ไม่มี token — อ่านได้เฉพาะ repo public")
    elif source == SOURCES[1]:
        path_tpl = st.text_input("Path ไฟล์", path_tpl)

    st.subheader("🤖 โมเดล")
    horizon = st.slider("พยากรณ์ล่วงหน้า (จำนวนแท่ง)", 1, 12, 1)
    max_bars = st.slider("จำนวนแท่งที่ใช้เทรน", 1000, 20000, 6000, step=500)
    test_frac = st.slider("สัดส่วนข้อมูลทดสอบ", 0.1, 0.4, 0.2, 0.05)
    prob_long = st.slider("เกณฑ์เข้า Long (prob ขึ้น ≥)", 0.50, 0.80, 0.55, 0.01)
    allow_short = st.checkbox(f"อนุญาต Short (prob ขึ้น ≤ {1 - prob_long:.2f})", value=False)
    fee = st.number_input("ค่าธรรมเนียมต่อครั้ง (%)", 0.0, 1.0, 0.10, 0.01) / 100

    st.subheader("🛡️ บริหารความเสี่ยง")
    capital = st.number_input("เงินทุน (USD)", 10.0, 1e9, 1000.0, 100.0)
    risk_pct = st.slider("ความเสี่ยงต่อไม้ (% ของทุน)", 0.25, 5.0, 1.0, 0.25) / 100
    sl_mult = st.slider("Stop-loss (× ATR)", 0.5, 4.0, 1.5, 0.25)
    tp_mult = st.slider("Take-profit (× ATR)", 0.5, 6.0, 2.5, 0.25)

    st.subheader("📊 กราฟ")
    show_bars = st.slider("จำนวนแท่งที่แสดง", 100, 1000, 300, 50)
    if st.button("🔄 ดึงข้อมูลใหม่", type="primary"):
        st.cache_data.clear()
        st.rerun()


def load_coin(coin: str) -> pd.DataFrame:
    rel = path_tpl.format(coin=coin, interval=interval, ticker=COINS[coin])
    if source == SOURCES[0]:
        df = load_from_gitlab(gl_base, gl_project, gl_branch, rel, gl_token)
    elif source == SOURCES[1]:
        df = load_from_local(rel)
    else:
        df = load_from_yfinance(COINS[coin], interval)
    df = resample_if_needed(df, interval)
    return df.tail(max_bars + 100)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("📈 Crypto Day Trade — วิเคราะห์และพยากรณ์ BTC · ETH · XRP")
st.caption(f"Timeframe {interval} · พยากรณ์ {horizon} แท่งข้างหน้า · แหล่งข้อมูล: {source} · "
           "⚠️ เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")

results, errors = {}, {}
with st.spinner("กำลังโหลดข้อมูลและเทรนโมเดล..."):
    for coin in COINS:
        try:
            df = load_coin(coin)
            res = run_analysis(df, horizon, test_frac, fee, prob_long, allow_short, BARS_PER_DAY[interval])
            res["signal"] = make_signal(res["p_last"], prob_long, allow_short)
            results[coin] = res
        except Exception as e:  # แสดงข้อผิดพลาดรายเหรียญ แต่ไม่ทำให้ทั้งแอปล่ม
            errors[coin] = str(e)

for coin, msg in errors.items():
    st.error(f"{coin}: {msg}")
if not results:
    st.info("ยังไม่มีข้อมูล — ตรวจสอบการตั้งค่าแหล่งข้อมูลทางแถบด้านซ้าย")
    st.stop()

tabs = st.tabs(["🧭 ภาพรวม"] + [f"🪙 {c}" for c in results])

# ---------------- Overview ----------------
with tabs[0]:
    rows = []
    for coin, res in results.items():
        d, m, bt = res["d"], res["metrics"], res["bt"]
        last = d["close"].iloc[-1]
        back = BARS_PER_DAY[interval]
        chg = last / d["close"].iloc[-back - 1] - 1 if len(d) > back else np.nan
        rows.append({
            "เหรียญ": coin,
            "ราคาล่าสุด": fmt_price(last),
            "เปลี่ยนแปลง 24 ชม.": f"{chg:+.2%}",
            "RSI": round(float(d["rsi"].iloc[-1]), 1),
            "แนวโน้ม (EMA9/21)": "ขาขึ้น" if d["ema9"].iloc[-1] > d["ema21"].iloc[-1] else "ขาลง",
            "prob ขึ้น": f"{res['p_last']:.1%}",
            "คาดการณ์ผลตอบแทน": f"{res['r_last']:+.3%}",
            "สัญญาณ": SIGNAL_LABEL[res["signal"]],
            "Accuracy (test)": f"{m['accuracy']:.1%}",
            "กลยุทธ์ vs B&H": f"{bt['total_return']:+.1%} / {bt['bh_return']:+.1%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("เหรียญ"))
    any_d = next(iter(results.values()))["d"]
    st.caption(f"แท่งล่าสุด: {any_d.index[-1]:%Y-%m-%d %H:%M} (เวลาไทย)")

    norm = pd.DataFrame({c: r["d"]["close"] / r["d"]["close"].iloc[-show_bars:].iloc[0]
                         for c, r in results.items()}).tail(show_bars)
    fig = go.Figure([go.Scatter(x=norm.index, y=(norm[c] - 1) * 100, name=c) for c in norm.columns])
    fig.update_layout(height=360, title=f"เปรียบเทียบผลตอบแทน {show_bars} แท่งล่าสุด (%)",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig)

    st.markdown(
        "**วิธีอ่าน:** `prob ขึ้น` คือความน่าจะเป็นที่ราคาจะสูงขึ้นใน N แท่งข้างหน้า จากโมเดลที่เทรนด้วยข้อมูลทั้งหมด · "
        "สัญญาณจะเป็น LONG เมื่อ prob ≥ เกณฑ์ · ควรดู Accuracy/AUC ของชุดทดสอบด้วยว่าโมเดลดีกว่าการเดาสุ่มหรือไม่"
    )

# ---------------- Per-coin tabs ----------------
for tab, (coin, res) in zip(tabs[1:], results.items()):
    with tab:
        d, m, bt = res["d"], res["metrics"], res["bt"]
        last = float(d["close"].iloc[-1])
        atr = float(d["atr"].iloc[-1])
        f_price = last * np.exp(res["r_last"])
        band = atr * np.sqrt(horizon)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ราคาล่าสุด", fmt_price(last))
        c2.metric(f"ราคาคาดการณ์ (+{horizon} แท่ง)", fmt_price(f_price), f"{res['r_last']:+.3%}")
        c3.metric("prob ขึ้น", f"{res['p_last']:.1%}")
        c4.metric("RSI(14)", f"{d['rsi'].iloc[-1]:.1f}")
        c5.metric("ATR(14)", fmt_price(atr), f"{atr / last:.2%} ของราคา", delta_color="off")

        st.subheader(f"สัญญาณ: {SIGNAL_LABEL[res['signal']]}")
        plan = trade_plan(d, res["signal"], capital, risk_pct, sl_mult, tp_mult)
        if plan:
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("จุดเข้า", fmt_price(plan["entry"]))
            p2.metric("Stop-loss", fmt_price(plan["sl"]), f"{plan['sl'] / plan['entry'] - 1:+.2%}", delta_color="off")
            p3.metric("Take-profit", fmt_price(plan["tp"]), f"{plan['tp'] / plan['entry'] - 1:+.2%}", delta_color="off")
            p4.metric("ขนาดไม้", f"{plan['qty']:,.6f} {coin}", f"≈ ${plan['notional']:,.2f}", delta_color="off")
            p5.metric("Risk : Reward", f"1 : {plan['rr']:.2f}", f"เสี่ยง ${plan['risk']:,.2f}", delta_color="off")
        else:
            st.info("ยังไม่มีจุดเข้าที่ผ่านเกณฑ์ความน่าจะเป็น — รอสัญญาณถัดไป")

        forecast = {"time": d.index[-1] + INTERVAL_DELTA[interval] * horizon,
                    "price": f_price, "low": f_price - band, "high": f_price + band}
        st.plotly_chart(price_chart(d, show_bars, coin, forecast))

        st.subheader("🧪 ประสิทธิภาพโมเดล (ชุดทดสอบ — ข้อมูลที่โมเดลไม่เคยเห็น)")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{m['accuracy']:.1%}", f"base rate {m['base_rate']:.1%}", delta_color="off")
        k2.metric("Precision (Long)", f"{m['precision']:.1%}")
        k3.metric("ROC-AUC", f"{m['auc']:.3f}")
        k4.metric("ทิศทางถูก (regressor)", f"{m['dir_acc_reg']:.1%}")
        k5.metric("MAE ผลตอบแทน", f"{m['mae_ret'] * 100:.3f}%")
        st.caption(f"train {m['n_train']:,} แท่ง · test {m['n_test']:,} แท่ง · AUC ≈ 0.5 หมายถึงไม่ต่างจากการเดาสุ่ม")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("ผลตอบแทนกลยุทธ์", f"{bt['total_return']:+.2%}", f"B&H {bt['bh_return']:+.2%}", delta_color="off")
        b2.metric("Max Drawdown", f"{bt['max_dd']:.2%}")
        b3.metric("Sharpe (ต่อปี)", f"{bt['sharpe']:.2f}")
        b4.metric("จำนวนเทรด", f"{bt['trades']:,}")
        b5.metric("Win rate (ต่อแท่ง)", f"{bt['win_rate']:.1%}", f"ถือสถานะ {bt['exposure']:.0%} ของเวลา", delta_color="off")
        st.plotly_chart(equity_chart(bt["equity"]))

        with st.expander("ดูผลพยากรณ์ล่าสุดในชุดทดสอบ"):
            st.dataframe(res["recent"].style.format({
                "ราคาปิด": "{:,.4f}", "prob ขึ้น": "{:.1%}",
                "คาดการณ์ผลตอบแทน %": "{:+.3f}", "ผลตอบแทนจริง %": "{:+.3f}"}))

st.divider()
st.caption("โมเดลเรียนรู้จากข้อมูลในอดีตเท่านั้น ตลาดคริปโตผันผวนสูง ผลย้อนหลังไม่รับประกันอนาคต "
           "ควรใช้ร่วมกับการบริหารความเสี่ยงและวิจารณญาณของตนเอง")
