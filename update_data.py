# -*- coding: utf-8 -*-
"""
ดาวน์โหลดข้อมูลแท่งเทียน BTC / ETH / XRP แล้วบันทึกเป็น CSV ในโฟลเดอร์ data/
เพื่อ commit ขึ้น GitLab (ไฟล์จะถูกรวมกับของเดิม ทำให้ประวัติยาวขึ้นเรื่อย ๆ)

ใช้งาน:
    python update_data.py                    # ทุกเหรียญ, 5m 15m 30m 1h
    python update_data.py --intervals 15m 1h
"""
import argparse
import os

import pandas as pd
import yfinance as yf

COINS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "XRP": "XRP-USD"}
PERIOD = {"5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}


def fetch(ticker: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD[interval], interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    tcol = "datetime" if "datetime" in df.columns else "date"
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[tcol], utc=True),
        "open": df["open"], "high": df["high"], "low": df["low"],
        "close": df["close"], "volume": df["volume"],
    })
    return out.dropna(subset=["close"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--intervals", nargs="+", default=list(PERIOD))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for coin, ticker in COINS.items():
        for itv in args.intervals:
            path = os.path.join(args.out, f"{coin}_{itv}.csv")
            new = fetch(ticker, itv)
            if os.path.exists(path):
                old = pd.read_csv(path)
                old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
                new = pd.concat([old, new])
            new = new.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
            # ตัดแท่งสุดท้ายที่ยังไม่ปิดออก
            new = new.iloc[:-1]
            new.to_csv(path, index=False)
            print(f"✔ {path}: {len(new):,} แท่ง ({new['timestamp'].min()} → {new['timestamp'].max()})")


if __name__ == "__main__":
    main()
