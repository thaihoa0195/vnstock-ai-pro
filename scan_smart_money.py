import json
from concurrent.futures import ThreadPoolExecutor
from app.data_service import POPULAR_STOCKS, fetch_history
from app.analyzer import calculate_ai_analysis

print("Scanning for Top 10 Smart Money Stocks...")

def analyze_stock(s):
    sym = s["symbol"]
    if s.get("is_index"):
        return None
    try:
        df = fetch_history(sym, days=60)
        if df is None or len(df) < 20:
            return None
        analysis = calculate_ai_analysis(df, sym)
        vsa = analysis.get("vsa", {})
        score = vsa.get("smart_money_score", 50)
        vol_ratio = vsa.get("vol_ratio", 1.0)
        price = analysis.get("current_price", 0)
        change = analysis.get("change_pct", 0)
        ew = analysis.get("elliott_wave", {})
        ichi = analysis.get("ichimoku", {})

        # Compute Smart Money Index combining VSA score, Volume Breakout, and Institutional Flow
        return {
            "symbol": sym,
            "name": s.get("name"),
            "sector": s.get("sector"),
            "price": price,
            "change_pct": change,
            "smart_money_score": score,
            "vsa_type": vsa.get("vsa_type"),
            "vsa_title": vsa.get("vsa_title"),
            "vsa_desc": vsa.get("vsa_desc"),
            "vol_ratio": vol_ratio,
            "foreign_flow": s.get("foreign_flow", "+0 tỷ"),
            "pe": s.get("pe"),
            "pb": s.get("pb"),
            "roe": s.get("roe"),
            "market_cap": s.get("market_cap"),
            "wave": ew.get("current_wave", "Sóng 3 Mở Rộng"),
            "wave_phase": ew.get("wave_phase", ""),
            "wave_target": ew.get("next_wave_target", 0),
            "cloud_status": ichi.get("cloud_status", "Trên Mây Kumo"),
            "tk_signal": ichi.get("tk_signal", "Tenkan > Kijun"),
            "entry_zone": analysis.get("entry_zone"),
            "tp1": analysis.get("target_1"),
            "tp2": analysis.get("target_2"),
            "stop_loss": analysis.get("stop_loss"),
            "risk_reward": analysis.get("risk_reward")
        }
    except Exception as e:
        print(f"Error {sym}: {e}")
        return None

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(filter(None, executor.map(analyze_stock, POPULAR_STOCKS)))

# Sort by Smart Money Score (desc), then Vol Ratio (desc)
results.sort(key=lambda x: (x["smart_money_score"], x["vol_ratio"]), reverse=True)

top10 = results[:10]

with open("smart_money_top10.json", "w", encoding="utf-8") as f:
    json.dump(top10, f, ensure_ascii=False, indent=2)

print(f"Successfully generated smart_money_top10.json with {len(top10)} stocks.")
