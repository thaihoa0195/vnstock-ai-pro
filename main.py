"""
FastAPI Server for Vietnam Stock AI Price Action Web Application
"""
import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.data_service import (
    POPULAR_STOCKS,
    fetch_history,
    get_stock_info,
    get_market_indices
)
from app.analyzer import calculate_ai_analysis, compute_indicators
from app.chat_service import generate_ai_chat_response
from app.forecaster import predict_daily_path

app = FastAPI(
    title="VN-Stock AI Pro - Phân Tích Đường Giá Chứng Khoán Việt Nam",
    description="Nền tảng phân tích kỹ thuật và hành động giá chứng khoán Việt Nam tích hợp AI thông minh",
    version="1.0.0"
)

# Ensure static & templates directory exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Variable to hold public live tunnel URL if active
PUBLIC_TUNNEL_URL: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    api_key: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def index_page():
    """
    Serve main application web interface
    """
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>VN-Stock AI Pro</h1>", status_code=200)

@app.get("/api/stocks")
async def list_stocks():
    """
    Return available Vietnamese stock list & sectors
    """
    return {"stocks": POPULAR_STOCKS}

@app.get("/api/market/indices")
async def market_indices():
    """
    Return VN-Index, VN30 status
    """
    return {"indices": get_market_indices()}

@app.get("/api/stock/{symbol}/history")
async def stock_history(symbol: str, days: int = Query(180, ge=30, le=500)):
    """
    Return OHLCV candlestick data for TradingView Lightweight Charts
    """
    symbol = symbol.upper().strip()
    df = fetch_history(symbol, days=days)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã {symbol}")

    # Compute indicators for overlay series (SMA20, SMA50, Bollinger Bands, EMA9, EMA21, RSI, MACD)
    df_ind = compute_indicators(df)

    candles = []
    volume_series = []
    sma20_series = []
    sma50_series = []
    ema9_series = []
    ema21_series = []
    bb_upper_series = []
    bb_lower_series = []
    rsi_series = []
    macd_series = []
    macd_signal_series = []
    macd_hist_series = []

    for _, row in df_ind.iterrows():
        t = str(row["time"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        v = float(row["volume"])
        is_up = c >= o

        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume_series.append({
            "time": t,
            "value": v,
            "color": "rgba(16, 185, 129, 0.6)" if is_up else "rgba(239, 68, 68, 0.6)"
        })

        if not row["sma_20"] != row["sma_20"]: # Not NaN
            sma20_series.append({"time": t, "value": round(float(row["sma_20"]), 2)})
        if not row["sma_50"] != row["sma_50"]:
            sma50_series.append({"time": t, "value": round(float(row["sma_50"]), 2)})
        if not row["ema_9"] != row["ema_9"]:
            ema9_series.append({"time": t, "value": round(float(row["ema_9"]), 2)})
        if not row["ema_21"] != row["ema_21"]:
            ema21_series.append({"time": t, "value": round(float(row["ema_21"]), 2)})
        if not row["bb_upper"] != row["bb_upper"]:
            bb_upper_series.append({"time": t, "value": round(float(row["bb_upper"]), 2)})
        if not row["bb_lower"] != row["bb_lower"]:
            bb_lower_series.append({"time": t, "value": round(float(row["bb_lower"]), 2)})
        if not row["rsi_14"] != row["rsi_14"]:
            rsi_series.append({"time": t, "value": round(float(row["rsi_14"]), 2)})
        if not row["macd"] != row["macd"]:
            macd_series.append({"time": t, "value": round(float(row["macd"]), 2)})
            macd_signal_series.append({"time": t, "value": round(float(row["macd_signal"]), 2)})
            hist_val = round(float(row["macd_hist"]), 2)
            macd_hist_series.append({
                "time": t,
                "value": hist_val,
                "color": "#10b981" if hist_val >= 0 else "#ef4444"
            })

    # Ichimoku series
    tenkan_series = []
    kijun_series = []
    senkou_a_series = []
    senkou_b_series = []
    for _, row in df_ind.iterrows():
        t = str(row["time"])
        if "tenkan_sen" in row and pd.notna(row["tenkan_sen"]):
            tenkan_series.append({"time": t, "value": round(float(row["tenkan_sen"]), 2)})
        if "kijun_sen" in row and pd.notna(row["kijun_sen"]):
            kijun_series.append({"time": t, "value": round(float(row["kijun_sen"]), 2)})
        if "senkou_span_a" in row and pd.notna(row["senkou_span_a"]):
            senkou_a_series.append({"time": t, "value": round(float(row["senkou_span_a"]), 2)})
        if "senkou_span_b" in row and pd.notna(row["senkou_span_b"]):
            senkou_b_series.append({"time": t, "value": round(float(row["senkou_span_b"]), 2)})

    info = get_stock_info(symbol)

    return {
        "symbol": symbol,
        "info": info,
        "candles": candles,
        "volume": volume_series,
        "indicators": {
            "sma20": sma20_series,
            "sma50": sma50_series,
            "ema9": ema9_series,
            "ema21": ema21_series,
            "bb_upper": bb_upper_series,
            "bb_lower": bb_lower_series,
            "rsi": rsi_series,
            "macd": macd_series,
            "macd_signal": macd_signal_series,
            "macd_hist": macd_hist_series,
            "tenkan": tenkan_series,
            "kijun": kijun_series,
            "senkou_a": senkou_a_series,
            "senkou_b": senkou_b_series
        }
    }

@app.get("/api/stock/{symbol}/analysis")
async def stock_analysis(symbol: str):
    """
    Generate AI Technical Analysis & Strategy Report
    """
    symbol = symbol.upper().strip()
    df = fetch_history(symbol, days=180)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Không thể phân tích mã {symbol}")

    analysis = calculate_ai_analysis(df, symbol)
    info = get_stock_info(symbol)
    analysis["info"] = info
    return analysis

@app.post("/api/stock/{symbol}/chat")
async def stock_chat(symbol: str, req: ChatRequest):
    """
    Interactive AI Q&A for stock price action
    """
    symbol = symbol.upper().strip()
    df = fetch_history(symbol, days=180)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu cổ phiếu")

    analysis = calculate_ai_analysis(df, symbol)
    answer = generate_ai_chat_response(
        symbol=symbol,
        question=req.question,
        analysis_data=analysis,
        api_key=req.api_key
    )
    return {"symbol": symbol, "answer": answer}

@app.get("/api/stock/{symbol}/forecast")
async def stock_forecast(symbol: str, days: int = Query(10, ge=3, le=20), model: str = Query("ensemble")):
    """
    Generate AI Daily Price Path Prediction (T+1 to T+10) using Deep Learning
    """
    symbol = symbol.upper().strip()
    df = fetch_history(symbol, days=120)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Không có dữ liệu lịch sử")
    forecast = predict_daily_path(df, days_ahead=days, model_type=model)
    forecast["symbol"] = symbol
    return forecast

@app.get("/api/top-picks")
async def get_top_picks():
    """
    Return AI-Screened Top 5 Stock Picks with Highest Profit Potential
    """
    picks_path = "top_5_picks.json"
    if os.path.exists(picks_path):
        try:
            with open(picks_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

@app.get("/api/tunnel-status")
async def get_tunnel_status():
    """
    Return current public online link if available
    """
    return {
        "online_link": PUBLIC_TUNNEL_URL,
        "is_online": PUBLIC_TUNNEL_URL is not None
    }

@app.post("/api/set-tunnel-url")
async def set_tunnel_url(payload: dict):
    global PUBLIC_TUNNEL_URL
    PUBLIC_TUNNEL_URL = payload.get("url")
    return {"status": "ok", "url": PUBLIC_TUNNEL_URL}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
