from vnstock import Market, Reference, Fundamental
import datetime

print("Testing Market symbols or listing...")
try:
    ref = Reference()
    stocks = ref.equity.listing()
    print("Listing head:", stocks.head(3).to_dict(orient="records") if stocks is not None else None)
except Exception as e:
    print("Ref error:", e)

try:
    fund = Fundamental()
    profile = fund.equity("FPT").profile()
    print("Profile keys:", list(profile.keys()) if isinstance(profile, dict) else profile)
except Exception as e:
    print("Fund error:", e)

try:
    m = Market()
    vnindex = m.index("VNINDEX").ohlcv(start="2026-01-01", end="2026-10-15")
    print("VN-Index shape:", vnindex.shape if vnindex is not None else None)
    if vnindex is not None:
        print("VN-Index latest:", vnindex.iloc[-1].to_dict())
except Exception as e:
    print("VNINDEX error:", e)
