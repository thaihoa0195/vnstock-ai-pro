import json
import requests
import datetime
from datetime import datetime, timedelta

def test_vnstock():
    print("Testing vnstock Market...")
    try:
        from vnstock import Market
        m = Market()
        # Test equity ohlcv
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        df = m.equity("FPT").ohlcv(start=start_date, end=end_date)
        print("Vnstock result shape:", df.shape if df is not None else None)
        if df is not None and not df.empty:
            print("Columns:", list(df.columns))
            print("First row:", df.iloc[0].to_dict())
            return True
    except Exception as e:
        print("Vnstock error:", e)
    return False

def test_tcbs():
    print("Testing TCBS direct API...")
    try:
        url = "https://apipubanalytics.tcbs.com.vn/stocks/v1/stock-bars/FPT/history?resolution=D&from=1700000000&to=1770000000"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=10)
        print("TCBS status:", res.status_code)
        if res.status_code == 200:
            data = res.json()
            bars = data.get("bars", [])
            print("TCBS bars count:", len(bars))
            if bars:
                print("First bar:", bars[0])
            return True
    except Exception as e:
        print("TCBS error:", e)
    return False

def test_ssi_or_cafef():
    print("Testing CafeF / SSI...")
    try:
        url = "https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:FPT~date:gte:2024-01-01&size=5"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        print("VNDirect status:", res.status_code)
        if res.status_code == 200:
            print("VNDirect data count:", len(res.json().get("data", [])))
            return True
    except Exception as e:
        print("VNDirect error:", e)
    return False

if __name__ == "__main__":
    vn_ok = test_vnstock()
    tcbs_ok = test_tcbs()
    vnd_ok = test_ssi_or_cafef()
    print("Summary:", {"vnstock": vn_ok, "tcbs": tcbs_ok, "vndirect": vnd_ok})
