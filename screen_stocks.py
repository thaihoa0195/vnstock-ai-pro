"""
Targeted Screener for Top VN Stocks to find Top 5 Recommendations
"""
import sys
import os
import json
from app.data_service import POPULAR_STOCKS, fetch_history
from app.analyzer import calculate_ai_analysis
from app.forecaster import predict_daily_path

TOP_TARGETS = [
    {"symbol": "TCB", "name": "Techcombank", "sector": "Ngân hàng"},
    {"symbol": "SSI", "name": "Chứng khoán SSI", "sector": "Chứng khoán"},
    {"symbol": "HPG", "name": "Tập đoàn Hòa Phát", "sector": "Thép & Xây dựng"},
    {"symbol": "MWG", "name": "Thế Giới Di Động", "sector": "Bán lẻ"},
    {"symbol": "FPT", "name": "Tập đoàn FPT", "sector": "Công nghệ"},
    {"symbol": "DGC", "name": "Hóa chất Đức Giang", "sector": "Hóa chất công nghiệp"},
    {"symbol": "MBB", "name": "MBBank", "sector": "Ngân hàng"},
    {"symbol": "VCB", "name": "Vietcombank", "sector": "Ngân hàng"}
]

def run():
    results = []
    for s in TOP_TARGETS:
        sym = s["symbol"]
        try:
            df = fetch_history(sym, days=120)
            if df is None or len(df) < 20:
                continue

            analysis = calculate_ai_analysis(df, sym)
            forecast = predict_daily_path(df, days_ahead=10)

            curr = analysis.get("current_price", 0)
            tp1 = analysis.get("target_1", 0)
            tp2 = analysis.get("target_2", 0)
            sl = analysis.get("stop_loss", 0)

            upside1 = round(((tp1 - curr) / curr) * 100, 1) if curr else 0
            upside2 = round(((tp2 - curr) / curr) * 100, 1) if curr else 0
            downside = round(((curr - sl) / curr) * 100, 1) if curr else 0

            results.append({
                "symbol": sym,
                "name": s["name"],
                "sector": s["sector"],
                "current_price": curr,
                "score": analysis.get("score", 50),
                "rating": analysis.get("rating", "QUAN SÁT"),
                "entry_zone": analysis.get("entry_zone"),
                "tp1": tp1,
                "tp2": tp2,
                "stop_loss": sl,
                "upside_tp1": upside1,
                "upside_tp2": upside2,
                "downside": downside,
                "risk_reward": analysis.get("risk_reward", "1:2"),
                "win_probability": forecast.get("win_probability", 50),
                "trend_forecast": forecast.get("trend_forecast", "TÍCH LŨY"),
                "rsi": analysis.get("indicators", {}).get("rsi", 50),
                "signals": [sig["text"] for sig in analysis.get("signals", []) if sig["type"] == "BULLISH"],
                "action": analysis.get("action")
            })
        except Exception as e:
            print(f"Error {sym}: {e}")

    # Rank by score and win probability
    results.sort(key=lambda x: (x["score"], x["win_probability"], x["upside_tp1"]), reverse=True)

    with open("top_5_picks.json", "w", encoding="utf-8") as f:
        json.dump(results[:5], f, ensure_ascii=False, indent=2)

    print("TOP 5 PICKS GENERATED SUCCESSFULLY.")

if __name__ == "__main__":
    run()
