"""
AI Technical Analysis & Price Action Engine for Vietnam Stock Market
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from app.elliott_ichimoku import (
    compute_ichimoku_kinko_hyo,
    compute_elliott_wave_and_fibonacci,
    generate_multi_timeframe_investment_strategy
)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on OHLCV DataFrame
    Expects columns: ['time', 'open', 'high', 'low', 'close', 'volume']
    """
    df = df.copy()
    # Sort ascending by time
    df = df.sort_values(by="time").reset_index(drop=True)

    # 1. Moving Averages
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_100"] = df["close"].rolling(window=100).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()

    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    # 2. RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # 3. MACD (12, 26, 9)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 4. Bollinger Bands (20, 2)
    df["bb_middle"] = df["sma_20"]
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)) * 100

    # 5. ATR (Average True Range 14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean().fillna(true_range)

    # 6. Volume MA
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)

    # 7. Ichimoku Kinko Hyo (Mây Mochi / Ichimoku)
    p9_high = df["high"].rolling(9).max()
    p9_low = df["low"].rolling(9).min()
    df["tenkan_sen"] = (p9_high + p9_low) / 2.0

    p26_high = df["high"].rolling(26).max()
    p26_low = df["low"].rolling(26).min()
    df["kijun_sen"] = (p26_high + p26_low) / 2.0

    df["senkou_span_a"] = (df["tenkan_sen"] + df["kijun_sen"]) / 2.0
    p52_high = df["high"].rolling(52).max()
    p52_low = df["low"].rolling(52).min()
    df["senkou_span_b"] = (p52_high + p52_low) / 2.0

    return df

def detect_candlestick_patterns(row: pd.Series, prev_row: Optional[pd.Series] = None) -> List[Dict[str, str]]:
    """
    Detect classic Japanese candlestick patterns
    """
    patterns = []
    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]
    body = abs(c - o)
    candle_range = h - l if (h - l) > 0 else 0.0001
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c > o
    is_bearish = c < o

    # Doji
    if body <= (candle_range * 0.1):
        if lower_wick > (candle_range * 0.6) and upper_wick < (candle_range * 0.2):
            patterns.append({"name": "Dragonfly Doji (Doji Chuồn chuồn)", "signal": "BULLISH", "desc": "Tín hiệu đảo chiều tăng mạnh từ hỗ trợ"})
        elif upper_wick > (candle_range * 0.6) and lower_wick < (candle_range * 0.2):
            patterns.append({"name": "Gravestone Doji (Doji Bia mộ)", "signal": "BEARISH", "desc": "Tín hiệu đảo chiều giảm từ kháng cự"})
        else:
            patterns.append({"name": "Doji (Lưỡng lự)", "signal": "NEUTRAL", "desc": "Thị trường cân bằng, đang chờ bứt phá"})

    # Hammer / Inverted Hammer
    elif is_bullish and lower_wick >= (2 * body) and upper_wick <= (0.2 * body):
        patterns.append({"name": "Hammer (Nến Cây búa)", "signal": "BULLISH", "desc": "Phe mua bắt đáy thành công, đẩy lùi phe bán"})
    elif is_bearish and upper_wick >= (2 * body) and lower_wick <= (0.2 * body):
        patterns.append({"name": "Shooting Star (Nến Sao băng)", "signal": "BEARISH", "desc": "Phe bán chặn giá tại kháng cự, cảnh báo giảm"})
    elif is_bullish and upper_wick >= (2 * body) and lower_wick <= (0.2 * body):
        patterns.append({"name": "Inverted Hammer (Búa ngược)", "signal": "BULLISH", "desc": "Dấu hiệu nỗ lực bứt phá của phe mua"})

    # Marubozu (Cường lực)
    if body >= (candle_range * 0.85):
        if is_bullish:
            patterns.append({"name": "Bullish Marubozu (Nến Xanh Cường lực)", "signal": "BULLISH", "desc": "Lực mua áp đảo tuyệt đối từ mở đến đóng cửa"})
        else:
            patterns.append({"name": "Bearish Marubozu (Nến Đỏ Cường lực)", "signal": "BEARISH", "desc": "Lực bán tháo áp đảo tuyệt đối"})

    # Two candle patterns
    if prev_row is not None:
        p_o = prev_row["open"]
        p_c = prev_row["close"]
        p_is_bearish = p_c < p_o
        p_is_bullish = p_c > p_o

        # Bullish Engulfing
        if p_is_bearish and is_bullish and c >= p_o and o <= p_c:
            patterns.append({"name": "Bullish Engulfing (Nhấn chìm tăng)", "signal": "BULLISH", "desc": "Nến xanh bao trùm hoàn toàn nến đỏ trước đó"})
        # Bearish Engulfing
        elif p_is_bullish and is_bearish and c <= p_o and o >= p_c:
            patterns.append({"name": "Bearish Engulfing (Nhấn chìm giảm)", "signal": "BEARISH", "desc": "Nến đỏ bao trùm hoàn toàn nến xanh trước đó"})
        # Bullish Harami
        elif p_is_bearish and is_bullish and c < p_o and o > p_c:
            patterns.append({"name": "Bullish Harami (Mẹ bồng con tăng)", "signal": "BULLISH", "desc": "Đà bán đã chững lại, phe mua manh nha hồi phục"})

    return patterns

def calculate_ai_analysis(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Perform complete AI analysis, calculating score, signals, levels, and narrative report
    """
    if df is None or len(df) < 5:
        return {"error": "Không đủ dữ liệu lịch sử để phân tích"}

    df = compute_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    prev2 = df.iloc[-3] if len(df) > 2 else prev

    c = float(last["close"])
    o = float(last["open"])
    h = float(last["high"])
    l = float(last["low"])
    vol = float(last["volume"])

    # Key Support & Resistance (Swing high/low last 40 bars)
    recent_window = df.tail(min(len(df), 40))
    swing_high = float(recent_window["high"].max())
    swing_low = float(recent_window["low"].min())
    fib_diff = swing_high - swing_low
    fib_382 = swing_high - 0.382 * fib_diff
    fib_500 = swing_high - 0.500 * fib_diff
    fib_618 = swing_high - 0.618 * fib_diff

    # Pivot Point
    pivot = (h + l + c) / 3
    r1 = (2 * pivot) - l
    s1 = (2 * pivot) - h
    r2 = pivot + (h - l)
    s2 = pivot - (h - l)

    # Candlestick patterns
    candlestick_patterns = detect_candlestick_patterns(last, prev)

    # Indicator values
    rsi = float(last["rsi_14"]) if not np.isnan(last["rsi_14"]) else 50.0
    macd = float(last["macd"]) if not np.isnan(last["macd"]) else 0.0
    macd_signal = float(last["macd_signal"]) if not np.isnan(last["macd_signal"]) else 0.0
    macd_hist = float(last["macd_hist"]) if not np.isnan(last["macd_hist"]) else 0.0
    sma_20 = float(last["sma_20"]) if not np.isnan(last["sma_20"]) else c
    sma_50 = float(last["sma_50"]) if not np.isnan(last["sma_50"]) else c
    sma_200 = float(last["sma_200"]) if not np.isnan(last["sma_200"]) else c
    ema_9 = float(last["ema_9"]) if not np.isnan(last["ema_9"]) else c
    ema_21 = float(last["ema_21"]) if not np.isnan(last["ema_21"]) else c
    bb_upper = float(last["bb_upper"]) if not np.isnan(last["bb_upper"]) else c * 1.05
    bb_lower = float(last["bb_lower"]) if not np.isnan(last["bb_lower"]) else c * 0.95
    bb_width = float(last["bb_width"]) if not np.isnan(last["bb_width"]) else 5.0
    atr = float(last["atr_14"]) if not np.isnan(last["atr_14"]) else c * 0.02
    vol_sma20 = float(last["vol_sma_20"]) if not np.isnan(last["vol_sma_20"]) else vol
    vol_ratio = float(last["vol_ratio"]) if not np.isnan(last["vol_ratio"]) else 1.0

    # AI Multi-factor Scoring
    score = 50.0 # Neutral baseline
    signals_list = []

    # 1. Trend Factor (Up to +- 25 pts)
    if c > sma_20 and sma_20 > sma_50:
        score += 15
        signals_list.append({"type": "BULLISH", "text": "Giá nằm trên MA20 & MA50 (Xu hướng tăng ngắn - trung hạn vững)"})
    elif c > sma_20:
        score += 8
        signals_list.append({"type": "BULLISH", "text": "Giá vượt lên trên MA20 ngắn hạn"})
    elif c < sma_20 and sma_20 < sma_50:
        score -= 15
        signals_list.append({"type": "BEARISH", "text": "Giá nằm dưới MA20 & MA50 (Xu hướng giảm chi phối)"})
    else:
        score -= 6
        signals_list.append({"type": "NEUTRAL", "text": "Giá dao động quanh MA20, đang giằng co tích lũy"})

    if len(df) >= 120 and sma_200 > 0:
        if c > sma_200:
            score += 10
            signals_list.append({"type": "BULLISH", "text": "Giá trên MA200 (Thị trường Bullish dài hạn)"})
        else:
            score -= 10
            signals_list.append({"type": "BEARISH", "text": "Giá dưới MA200 (Thị trường Bearish dài hạn)"})

    # 2. Momentum Factor (Up to +- 20 pts)
    if rsi < 30:
        score += 12 # Oversold opportunity
        signals_list.append({"type": "BULLISH", "text": f"RSI({rsi:.1f}) rơi vào vùng Quá bán (Oversold < 30) - Tiềm năng bật đáy"})
    elif rsi > 70:
        score -= 8 # Overbought risk
        signals_list.append({"type": "BEARISH", "text": f"RSI({rsi:.1f}) chạm vùng Quá mua (Overbought > 70) - Áp lực chốt lời gia tăng"})
    elif 50 <= rsi <= 65:
        score += 8
        signals_list.append({"type": "BULLISH", "text": f"RSI({rsi:.1f}) trong vùng tăng trưởng bền vững (50 - 65)"})
    elif 35 <= rsi < 50:
        score -= 5
        signals_list.append({"type": "BEARISH", "text": f"RSI({rsi:.1f}) nằm dưới mốc 50, động lực tăng còn yếu"})

    # MACD Factor
    prev_macd = float(prev["macd"]) if not np.isnan(prev["macd"]) else 0.0
    prev_sig = float(prev["macd_signal"]) if not np.isnan(prev["macd_signal"]) else 0.0
    if prev_macd <= prev_sig and macd > macd_signal:
        score += 15
        signals_list.append({"type": "BULLISH", "text": "Tín hiệu MACD Golden Cross: MACD vừa cắt lên đường Tín hiệu"})
    elif macd > macd_signal:
        score += 6
        signals_list.append({"type": "BULLISH", "text": "Đường MACD duy trì trên Signal Line"})
    elif prev_macd >= prev_sig and macd < macd_signal:
        score -= 14
        signals_list.append({"type": "BEARISH", "text": "Tín hiệu MACD Death Cross: MACD vừa cắt xuống đường Tín hiệu"})
    else:
        score -= 6
        signals_list.append({"type": "BEARISH", "text": "Đường MACD nằm dưới Signal Line"})

    # 3. Volume & Flow Factor (Up to +- 15 pts)
    if vol_ratio >= 1.5 and c > prev["close"]:
        score += 12
        signals_list.append({"type": "BULLISH", "text": f"Khối lượng bùng nổ (+{(vol_ratio-1)*100:.0f}% so với TB20) kèm giá tăng (Dòng tiền lớn vào)"})
    elif vol_ratio >= 1.5 and c < prev["close"]:
        score -= 12
        signals_list.append({"type": "BEARISH", "text": f"Khối lượng bán đột biến (+{(vol_ratio-1)*100:.0f}% so với TB20) (Áp lực phân phối mạnh)"})
    elif vol_ratio < 0.7:
        signals_list.append({"type": "NEUTRAL", "text": "Thanh khoản thấp, thị trường thận trọng chờ tín hiệu"})

    # 4. Bollinger Bands & Volatility Factor
    if c >= bb_upper:
        score -= 5
        signals_list.append({"type": "BEARISH", "text": "Giá chạm dải trên Bollinger Band, cẩn trọng nhịp chỉnh kỹ thuật"})
    elif c <= bb_lower:
        score += 8
        signals_list.append({"type": "BULLISH", "text": "Giá chạm dải dưới Bollinger Band, xác suất bật nảy kỹ thuật cao"})
    elif bb_width < 6.0:
        signals_list.append({"type": "NEUTRAL", "text": f"Bollinger Band co thắt cực hẹp ({bb_width:.1f}%) - Chuẩn bị biến động bùng nổ"})

    # 5. Candlestick patterns addition
    for p in candlestick_patterns:
        if p["signal"] == "BULLISH":
            score += 6
            signals_list.append({"type": "BULLISH", "text": f"Mẫu hình nến: {p['name']} - {p['desc']}"})
        elif p["signal"] == "BEARISH":
            score -= 6
            signals_list.append({"type": "BEARISH", "text": f"Mẫu hình nến: {p['name']} - {p['desc']}"})

    # Clamp Score between 0 and 100
    final_score = max(5.0, min(95.0, round(score, 1)))

    # Overall Rating & Recommendation
    if final_score >= 75:
        rating = "MUA MẠNH"
        rating_color = "#10b981" # Green
        action = "Canh mua chủ động theo đà tăng hoặc tích lũy vị thế tại vùng hỗ trợ"
    elif final_score >= 60:
        rating = "MUA"
        rating_color = "#34d399"
        action = "Có thể mở vị thế mua thăm dò, giải ngân từng phần quanh vùng hỗ trợ"
    elif final_score >= 45:
        rating = "QUAN SÁT / NẮM GIỮ"
        rating_color = "#f59e0b" # Amber
        action = "Chưa có điểm mua an toàn rõ nét. Ưu tiên quan sát phản ứng giá hoặc nắm giữ chờ chốt"
    elif final_score >= 30:
        rating = "BÁN / HẠ TỶ TRỌNG"
        rating_color = "#f87171"
        action = "Hạ tỷ trọng cổ phiếu, hạ margin, không mua đuổi khi có nhịp hồi kỹ thuật"
    else:
        rating = "BÁN MẠNH"
        rating_color = "#ef4444" # Red
        action = "Thoát vị thế hoặc đứng ngoài, xu hướng giảm giá đang kiểm soát hoàn toàn"

    # Dynamic Trading Targets & Stop Loss
    stop_loss = max(round(l - 1.2 * atr, 2), round(s1 * 0.985, 2))
    # Ensure stop loss is below current price
    if stop_loss >= c:
        stop_loss = round(c * 0.94, 2)

    risk = c - stop_loss
    tp1 = round(c + 1.5 * risk, 2)
    tp2 = round(c + 2.5 * risk, 2)
    entry_zone = f"{round(max(s1, c * 0.98), 2):,} - {round(c, 2):,}"
    rr_ratio = round((tp1 - c) / risk, 1) if risk > 0 else 1.5

    change_pct = round(((c - prev["close"]) / prev["close"]) * 100, 2) if prev["close"] else 0.0

    # 7. Ichimoku & Elliott Wave Analysis
    ichimoku = compute_ichimoku_kinko_hyo(df)
    elliott = compute_elliott_wave_and_fibonacci(df)
    strategy = generate_multi_timeframe_investment_strategy(symbol, c, ichimoku, elliott)

    # Append Ichimoku & Elliott to signals
    if ichimoku:
        signals_list.append({
            "type": "BULLISH" if ichimoku.get("cloud_status") == "BULLISH" else ("BEARISH" if ichimoku.get("cloud_status") == "BEARISH" else "NEUTRAL"),
            "text": f"Mây Ichimoku: {ichimoku.get('cloud_position', '')} - {ichimoku.get('tk_signal', '')}"
        })
    if elliott:
        signals_list.append({
            "type": "BULLISH",
            "text": f"Sóng Elliott: {elliott.get('current_wave', '')} (Mục tiêu sóng: {elliott.get('next_wave_target', c):,.2f} VNĐ)"
        })

    # AI Report Generation
    report = generate_vietnamese_ai_report(
        symbol=symbol,
        score=final_score,
        rating=rating,
        price=c,
        change_pct=change_pct,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        sma_20=sma_20,
        sma_50=sma_50,
        vol_ratio=vol_ratio,
        swing_high=swing_high,
        swing_low=swing_low,
        s1=s1,
        r1=r1,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        patterns=candlestick_patterns,
        action=action,
        ichimoku=ichimoku,
        elliott=elliott,
        strategy=strategy
    )

    return {
        "symbol": symbol,
        "current_price": c,
        "open": o,
        "high": h,
        "low": l,
        "volume": vol,
        "score": final_score,
        "rating": rating,
        "rating_color": rating_color,
        "action": action,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "target_1": tp1,
        "target_2": tp2,
        "risk_reward": f"1:{rr_ratio}",
        "indicators": {
            "rsi": round(rsi, 2),
            "macd": round(macd, 2),
            "macd_signal": round(macd_signal, 2),
            "macd_hist": round(macd_hist, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "ema_9": round(ema_9, 2),
            "ema_21": round(ema_21, 2),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_width": round(bb_width, 2),
            "atr_14": round(atr, 2),
            "vol_ratio": round(vol_ratio, 2)
        },
        "levels": {
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "fib_382": round(fib_382, 2),
            "fib_500": round(fib_500, 2),
            "fib_618": round(fib_618, 2)
        },
        "ichimoku": ichimoku,
        "elliott_wave": elliott,
        "investment_strategy": strategy,
        "patterns": candlestick_patterns,
        "signals": signals_list,
        "ai_report": report
    }

def generate_vietnamese_ai_report(
    symbol: str, score: float, rating: str, price: float, change_pct: float,
    rsi: float, macd: float, macd_signal: float, sma_20: float, sma_50: float,
    vol_ratio: float, swing_high: float, swing_low: float,
    s1: float, r1: float, stop_loss: float, tp1: float, tp2: float,
    patterns: List[Dict[str, str]], action: str,
    ichimoku: Optional[Dict[str, Any]] = None,
    elliott: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None
) -> str:
    """
    Produce structured, in-depth analytical report in Vietnamese
    """
    sign_str = f"+{change_pct}%" if change_pct >= 0 else f"{change_pct}%"
    trend_state = "TĂNG TRƯỞNG" if price > sma_20 > sma_50 else ("ĐIỀU CHỈNH / TÍCH LŨY" if price >= sma_50 else "SUY YẾU / GIẢM GIÁ")

    pattern_texts = ", ".join([p["name"] for p in patterns]) if patterns else "Chưa xuất hiện mẫu hình nến đặc biệt"

    money_flow = "DÒNG TIỀN VÀO MẠNH (Vol vượt trội)" if vol_ratio > 1.3 and change_pct > 0 else (
        "ÁP LỰC BÁN ÁP ĐẢO (Vol lớn khi giá giảm)" if vol_ratio > 1.3 and change_pct < 0 else "Thanh khoản trung bình, lực mua bán ở mức cân bằng"
    )

    report_lines = [
        f"### 📊 BÁO CÁO PHÂN TÍCH HÀNH ĐỘNG GIÁ AI: MÃ {symbol}",
        f"**Đánh giá tổng thể**: **{rating}** (Điểm kỹ thuật AI: **{score}/100**)",
        f"**Giá hiện tại**: `{price:,.2f} VNĐ` ({sign_str}) | **Trạng thái**: `{trend_state}`\n",
        "#### 1. Cấu trúc Xu hướng & Hành động Giá (Price Action)",
        f"- Cổ phiếu {symbol} đang giao dịch trong biên độ dao động gần nhất từ đáy `{swing_low:,.2f}` đến đỉnh `{swing_high:,.2f}`.",
        f"- So sánh với các đường trung bình động: Giá hiện tại đang {'nằm TRÊN' if price >= sma_20 else 'nằm DƯỚI'} đường MA20 (`{sma_20:,.2f}`) và {'TRÊN' if price >= sma_50 else 'DƯỚI'} MA50 (`{sma_50:,.2f}`).",
        f"- Mẫu hình nến ghi nhận gần nhất: **{pattern_texts}**.\n",
        "#### 2. Phân tích Tín hiệu Kỹ thuật & Động lượng",
        f"- **Chỉ báo RSI (14)**: Đạt mức `{rsi:.1f}`. {'Cảnh báo quá mua, áp lực rung lắc cao.' if rsi > 70 else ('Vùng quá bán, mở ra xác suất phục hồi kỹ thuật.' if rsi < 30 else 'Động lượng nằm trong khoảng ổn định, phe mua và bán đang tìm điểm cân bằng.')}",
        f"- **Chỉ báo MACD**: Đường MACD (`{macd:.2f}`) đang {'NẰM TRÊN' if macd > macd_signal else 'NẰM DƯỚI'} đường Signal (`{macd_signal:.2f}`). {'Xung lực tăng đang có ưu thế.' if macd > macd_signal else 'Áp lực điều chỉnh ngắn hạn vẫn còn.'}",
        f"- **Động thái Dòng tiền**: {money_flow} (Tỷ lệ khối lượng so với TB 20 phiên: `{vol_ratio:.2f}x`).\n",
    ]

    if ichimoku and elliott:
        report_lines.extend([
            "#### 3. Phân Tích Sóng Elliott & Mây Ichimoku Kinko Hyo (Mây Mochi)",
            f"- **Vị thế Sóng Elliott**: **{elliott.get('current_wave', 'N/A')}**.",
            f"  - *Nhận định cấu trúc sóng*: {elliott.get('wave_description', '')}",
            f"  - *Mục tiêu sóng tiếp theo*: `{elliott.get('next_wave_target', price):,.2f} VNĐ` | *Hỗ trợ then chốt sóng*: `{elliott.get('key_wave_support', swing_low):,.2f} VNĐ`.",
            f"  - *Các mốc Fibonacci mở rộng*: Fibo 1.272: `{elliott.get('fib_ext_127', 0):,.2f}` | Fibo 1.618: `{elliott.get('fib_ext_161', 0):,.2f}` | Fibo 2.618: `{elliott.get('fib_ext_261', 0):,.2f}`.",
            f"- **Trạng thái Mây Ichimoku**: **{ichimoku.get('cloud_position', 'N/A')}** ({ichimoku.get('cloud_color', '')}).",
            f"  - *Tín hiệu Tenkan/Kijun*: **{ichimoku.get('tk_signal', '')}** (Tenkan: `{ichimoku.get('tenkan_sen', 0):,.2f}` | Kijun: `{ichimoku.get('kijun_sen', 0):,.2f}`).",
            f"  - *Biên độ dải mây Kumo*: Dải trên `{ichimoku.get('cloud_top', 0):,.2f}` | Dải dưới `{ichimoku.get('cloud_bottom', 0):,.2f}` (Độ dày mây: `{ichimoku.get('cloud_thickness_pct', 0)}%`).",
            f"  - *Đánh giá Kumo*: {ichimoku.get('cloud_description', '')}\n"
        ])

    if strategy:
        w = strategy.get("weekly", {})
        m = strategy.get("monthly", {})
        y = strategy.get("yearly", {})
        report_lines.extend([
            "#### 4. Chiến Lược Đầu Tư Đa Khung Thời Gian (Tuần - Tháng - Năm)",
            f"- 🗓️ **THEO TUẦN (1 - 4 Tuần)**: Mục tiêu `{w.get('target_price', 0):,.2f} VNĐ` (+{w.get('upside_pct', 0)}%) | Dừng lỗ: `{w.get('stop_loss', 0):,.2f}` | R:R `{w.get('risk_reward', '')}`.",
            f"  - *Chiến thuật*: {w.get('catalyst', '')}",
            f"- 📅 **THEO THÁNG (1 - 6 Tháng)**: Mục tiêu `{m.get('target_price', 0):,.2f} VNĐ` (+{m.get('upside_pct', 0)}%) | Dừng lỗ: `{m.get('stop_loss', 0):,.2f}` | R:R `{m.get('risk_reward', '')}`.",
            f"  - *Chiến thuật*: {m.get('catalyst', '')}",
            f"- 📈 **THEO NĂM (1 - 3 Năm)**: Mục tiêu `{y.get('target_price', 0):,.2f} VNĐ` (+{y.get('upside_pct', 0)}%) | Tỷ lệ R:R `{y.get('risk_reward', '')}`.",
            f"  - *Chiến lược*: {y.get('catalyst', '')}\n"
        ])

    report_lines.extend([
        "#### 5. Khuyến Nghị Hành Động Tổng Thể",
        f"- **Hành động đề xuất**: **{action}**.",
        f"- **Vùng gom mua tối ưu**: Quanh `{max(s1, price * 0.98):,.2f}` - `{price:,.2f} VNĐ`",
        f"- **Mục tiêu chốt lời (TP1 / TP2)**: `{tp1:,.2f}` / `{tp2:,.2f} VNĐ`",
        f"- **Điểm dừng lỗ (Stop Loss)**: `{stop_loss:,.2f} VNĐ`"
    ])

    return "\n".join(report_lines)
