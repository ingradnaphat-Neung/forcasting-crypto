# 📈 Crypto Day Trade — วิเคราะห์และพยากรณ์ BTC · ETH · XRP

แอป Streamlit สำหรับเดย์เทรด: ดึงข้อมูลแท่งเทียนจาก **GitLab** → คำนวณอินดิเคเตอร์ → พยากรณ์ด้วย ML → Backtest → แผนเทรด (Entry / SL / TP / ขนาดไม้) รันบน **Google Colab**

> ⚠️ เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน

## โครงสร้าง repo บน GitLab
```
crypto-daytrade/
├── app.py               # แอป Streamlit
├── update_data.py       # ดึงข้อมูลล่าสุดจาก yfinance → data/*.csv
├── requirements.txt
├── run_on_colab.ipynb   # เปิดใน Colab เพื่อรันแอป
└── data/
    ├── BTC_15m.csv      # ชื่อไฟล์: {coin}_{interval}.csv
    ├── ETH_15m.csv
    └── XRP_15m.csv
```

**รูปแบบ CSV:** `timestamp,open,high,low,close,volume`
- `timestamp` เป็นวันที่-เวลา (ISO) หรือ epoch วินาที/มิลลิวินาทีก็ได้ (เช่นไฟล์ export จาก Binance)
- ถ้าไฟล์ละเอียดกว่า timeframe ที่เลือก (เช่นไฟล์ 1m/5m แต่เลือก 15m) แอปจะรวมแท่งให้อัตโนมัติ

## เริ่มใช้งาน
1. สร้าง project บน GitLab แล้ว push ไฟล์ทั้งหมดขึ้นไป
2. ถ้ายังไม่มีข้อมูล: `python update_data.py` แล้ว commit โฟลเดอร์ `data/`
3. repo private → สร้าง Access Token (Settings → Access Tokens) สิทธิ์ `read_api`, `read_repository` (+ `write_repository` ถ้าจะ push ข้อมูลจาก Colab)
4. เปิด `run_on_colab.ipynb` ใน Colab (File → Upload notebook) → แก้ `GITLAB_PROJECT` → Run all → คลิกลิงก์ `trycloudflare.com`

## แหล่งข้อมูลในแอป (เลือกที่แถบซ้าย)
| โหมด | คำอธิบาย |
|---|---|
| GitLab (API) | อ่าน CSV ผ่าน GitLab REST API ทุกครั้งที่รีเฟรช (cache 5 นาที) |
| โฟลเดอร์ในเครื่อง | อ่านจาก `data/` ของ repo ที่ clone มา |
| yfinance | ดึงสดจาก Yahoo Finance (ไม่ต้องมีไฟล์) |

## สิ่งที่แอปทำ
- **อินดิเคเตอร์:** EMA 9/21/50, RSI 14, MACD, Bollinger Bands, ATR 14, VWAP (รีเซ็ตรายวันตามเวลาไทย)
- **โมเดล:** HistGradientBoosting — Classifier ทำนาย prob ราคาขึ้น และ Regressor ทำนาย % ผลตอบแทน N แท่งข้างหน้า
- **ตรวจสอบ:** แบ่ง train/test ตามเวลา (มีช่องว่างกันข้อมูลรั่ว) แสดง Accuracy, Precision, ROC-AUC, MAE
- **Backtest:** หักค่าธรรมเนียม เทียบกับ Buy & Hold พร้อม Max Drawdown, Sharpe, Win rate
- **แผนเทรด:** SL/TP ตามตัวคูณ ATR และคำนวณขนาดไม้จาก % ความเสี่ยงต่อไม้

## รันในเครื่อง
```bash
pip install -r requirements.txt
export GITLAB_PROJECT="group/crypto-daytrade" GITLAB_TOKEN="glpat-..."   # ถ้า repo private
streamlit run app.py
```
หรือใส่ค่าใน `.streamlit/secrets.toml` (`GITLAB_URL`, `GITLAB_PROJECT`, `GITLAB_BRANCH`, `GITLAB_TOKEN`, `DATA_SOURCE`) — **อย่า commit ไฟล์นี้ขึ้น repo**

## ข้อควรระวัง
- ราคาคริปโตระยะสั้นมี noise สูง ถ้า ROC-AUC ใกล้ 0.5 แปลว่าโมเดลไม่ต่างจากการเดาสุ่ม — อย่าใช้สัญญาณนั้นเทรดจริง
- Backtest ยังไม่รวม slippage และสมมติว่าเข้า/ออกที่ราคาปิด ผลจริงจะแย่กว่า
- ลิงก์ trycloudflare เปลี่ยนทุกครั้งที่รันใหม่ และหยุดเมื่อ Colab runtime หลุด
