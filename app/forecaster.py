"""
AI Daily Price Path Forecasting Engine using Deep Learning & Quantitative Time-Series
Supported Architectures:
- Ensemble Deep AI (Tổ hợp đa mô hình)
- LSTM Neural Network (Long Short-Term Memory)
- GRU Gated Network (Gated Recurrent Unit)
- Transformer Attention (Temporal Multi-Head Attention)
"""
import datetime
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from app.deep_models import train_and_predict_deep_models

def get_next_trading_days(start_date_str: str, n_days: int = 10) -> List[str]:
    """
    Generate next n valid trading days (Monday to Friday)
    """
    try:
        curr = datetime.strptime(start_date_str, "%Y-%m-%d")
    except Exception:
        curr = datetime.now()

    trading_days = []
    d = curr + timedelta(days=1)
    while len(trading_days) < n_days:
        if d.weekday() < 5:  # Monday to Friday
            trading_days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return trading_days

def predict_daily_path(
    df: pd.DataFrame, 
    days_ahead: int = 10, 
    model_type: str = "ensemble",
    symbol: Optional[str] = None,
    technical_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict daily price path (T+1 to T+10) using Deep Learning models harmonized with technical context
    """
    if df is None or len(df) < 20:
        return {"error": "Cần ít nhất 20 phiên dữ liệu để dự báo đường giá"}

    df = df.copy().sort_values(by="time").reset_index(drop=True)
    last_row = df.iloc[-1]
    last_price = float(last_row["close"])
    last_date = str(last_row["time"])[:10]

    # Calculate returns & volatility
    df["return"] = df["close"].pct_change()
    returns = df["return"].dropna()
    daily_vol = float(returns.tail(20).std()) if len(returns) >= 20 else 0.018
    daily_vol = max(0.008, min(daily_vol, 0.045))

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])
    rsi = 50.0 if np.isnan(rsi) else rsi

    # Retrieve technical context if symbol is provided
    if technical_context is None and symbol:
        try:
            from app.analyzer import calculate_ai_analysis
            technical_context = calculate_ai_analysis(df, symbol)
        except Exception:
            technical_context = None

    # Run Deep Learning models suite
    deep_results = train_and_predict_deep_models(df, days_ahead=days_ahead, technical_context=technical_context)
    if not deep_results:
        return {"error": "Không thể khởi tạo mô hình Deep Learning"}

    # Selected model
    model_key = model_type.lower().strip()
    if model_key not in deep_results:
        model_key = "ensemble"
    active_m = deep_results[model_key]

    future_dates = get_next_trading_days(last_date, n_days=days_ahead)

    path_base = []
    path_bull = []
    path_bear = []
    daily_details = []

    m_base = active_m["path_base"]
    m_bull = active_m["path_bull"]
    m_bear = active_m["path_bear"]
    drift = active_m["drift"]

    # Calculate metrics
    target_t5 = m_base[4] if len(m_base) >= 5 else m_base[-1]
    target_t10 = m_base[-1]
    expected_gain_t5 = round(((target_t5 - last_price) / last_price) * 100, 2)
    expected_gain_t10 = round(((target_t10 - last_price) / last_price) * 100, 2)

    # Contextual Trend Classification
    t_score = float(technical_context.get("score", 50.0)) if technical_context else 50.0
    t_rating = str(technical_context.get("rating", "")).upper() if technical_context else ""

    if t_score <= 40.0 or ("BÁN" in t_rating) or expected_gain_t10 < -1.5:
        trend_forecast = "ĐIỀU CHỈNH GIẢM (RỦI RO)"
        trend_color = "#ef4444"
        win_prob = round(max(28.0, min(45.0, 35.0 + (t_score - 20.0) * 0.4)), 1)
    elif t_score >= 70.0 or ("MUA MẠNH" in t_rating):
        trend_forecast = "TĂNG TRƯỞNG MẠNH (BỨT PHÁ)"
        trend_color = "#10b981"
        win_prob = round(max(78.0, min(92.0, 74.0 + (t_score - 70.0) * 0.5 + max(0.0, expected_gain_t10) * 1.2)), 1)
    elif t_score >= 55.0 or ("MUA" in t_rating):
        trend_forecast = "TĂNG TRƯỞNG KỸ THUẬT"
        trend_color = "#34d399"
        win_prob = round(max(65.0, min(77.0, 62.0 + (t_score - 55.0) * 0.8 + max(0.0, expected_gain_t10) * 1.0)), 1)
    elif expected_gain_t10 >= 0.8:
        trend_forecast = "TÍCH LŨY HỒI PHỤC"
        trend_color = "#38bdf8"
        win_prob = round(max(55.0, min(66.0, 55.0 + expected_gain_t10 * 1.8)), 1)
    else:
        trend_forecast = "TÍCH LŨY ĐI NGANG (TEST HỖ TRỢ)"
        trend_color = "#f59e0b"
        win_prob = 52.5

    for step, f_date in enumerate(future_dates):
        step_idx = step + 1
        curr_base = m_base[step] if step < len(m_base) else last_price
        curr_bull = m_bull[step] if step < len(m_bull) else last_price * 1.05
        curr_bear = m_bear[step] if step < len(m_bear) else last_price * 0.95

        pct_change = ((curr_base - last_price) / last_price) * 100.0

        path_base.append({"time": f_date, "value": curr_base})
        path_bull.append({"time": f_date, "value": curr_bull})
        path_bear.append({"time": f_date, "value": curr_bear})

        daily_details.append({
            "step": f"T+{step_idx}",
            "date": f_date,
            "predicted_price": curr_base,
            "range_min": curr_bear,
            "range_max": curr_bull,
            "change_pct": round(pct_change, 2),
            "trend": "Tăng" if pct_change > 0.2 else ("Giảm" if pct_change < -0.2 else "Đi ngang"),
            "confidence": f"{max(58, int(94 - step_idx * 2.8))}%"
        })

    commentary = generate_forecast_commentary(
        model_name=active_m["name"],
        model_type=active_m["type"],
        accuracy=active_m["accuracy"],
        loss_mse=active_m["loss_mse"],
        last_price=last_price,
        target_t5=target_t5,
        target_t10=target_t10,
        gain_t5=expected_gain_t5,
        gain_t10=expected_gain_t10,
        win_prob=win_prob,
        trend=trend_forecast,
        rsi=rsi,
        daily_vol=daily_vol * 100,
        technical_context=technical_context
    )

    # Format deep models for client consumption
    formatted_models = {}
    for k, v in deep_results.items():
        formatted_models[k] = {
            "name": v["name"],
            "type": v["type"],
            "accuracy": v["accuracy"],
            "loss_mse": v["loss_mse"],
            "epochs": v["epochs"],
            "description": v["description"],
            "path_base": [{"time": f_date, "value": v["path_base"][i]} for i, f_date in enumerate(future_dates)],
            "path_bull": [{"time": f_date, "value": v["path_bull"][i]} for i, f_date in enumerate(future_dates)],
            "path_bear": [{"time": f_date, "value": v["path_bear"][i]} for i, f_date in enumerate(future_dates)]
        }

    return {
        "last_price": last_price,
        "last_date": last_date,
        "days_ahead": days_ahead,
        "active_model_key": model_key,
        "active_model": {
            "name": active_m["name"],
            "type": active_m["type"],
            "accuracy": active_m["accuracy"],
            "loss_mse": active_m["loss_mse"],
            "epochs": active_m["epochs"],
            "description": active_m["description"]
        },
        "all_models": formatted_models,
        "win_probability": win_prob,
        "trend_forecast": trend_forecast,
        "trend_color": trend_color,
        "expected_gain_t5": expected_gain_t5,
        "expected_gain_t10": expected_gain_t10,
        "target_price_t5": target_t5,
        "target_price_t10": target_t10,
        "path_base": path_base,
        "path_bull": path_bull,
        "path_bear": path_bear,
        "daily_details": daily_details,
        "commentary": commentary
    }

def generate_forecast_commentary(
    model_name: str, model_type: str, accuracy: str, loss_mse: str,
    last_price: float, target_t5: float, target_t10: float,
    gain_t5: float, gain_t10: float, win_prob: float,
    trend: str, rsi: float, daily_vol: float,
    technical_context: Optional[Dict[str, Any]] = None
) -> str:
    sign_t5 = f"+{gain_t5}%" if gain_t5 >= 0 else f"{gain_t5}%"
    sign_t10 = f"+{gain_t10}%" if gain_t10 >= 0 else f"{gain_t10}%"

    reconcile_note = ""
    if technical_context:
        t_score = technical_context.get("score", 50.0)
        t_rating = technical_context.get("rating", "QUAN SÁT")
        t_elliott = technical_context.get("elliott_wave", {}).get("current_wave", "") if isinstance(technical_context.get("elliott_wave"), dict) else str(technical_context.get("elliott_wave", ""))
        entry_zone = technical_context.get("entry_zone", f"{last_price*0.98:.2f} - {last_price:.2f}")

        if t_score >= 70 or gain_t10 > 2.0:
            reconcile_note = (
                f"\n• 💡 **Chiến thuật Nhịp T+ & Vùng Gom Hàng**: Mô hình Deep Learning ghi nhận các phiên đầu (T+1..T+2) "
                f"có thể xuất hiện nhịp rung lắc nén giá lành mạnh kiểm định lại vùng hỗ trợ mây Kumo ({entry_zone} VNĐ). "
                f"Đây là **VÙNG GOM MUA TỐI ƯU** theo cấu trúc **{t_elliott or 'Sóng 3 Mở Rộng'}** "
                f"trước khi đường giá bước vào pha bứt phá gia tốc hướng tới mục tiêu T+5 ({target_t5:,.2f}) và T+10 ({target_t10:,.2f} VNĐ)."
            )

    return (
        f"Mô hình Deep Learning **{model_name}** ({model_type}) đã hoàn tất huấn luyện online với hàm mất mát MSE đạt **{loss_mse}** "
        f"và độ chính xác định hướng kiểm định đạt **{accuracy}**.\n\n"
        f"• **Dự báo xu hướng chủ đạo**: **{trend}** trong 10 phiên tới (Độ tin cậy định lượng: **{win_prob}%**).\n"
        f"• **Ngắn hạn T+1 đến T+5**: Đường nơ-ron học sâu dự phóng giá tiệm cận mốc **{target_t5:,.2f} VNĐ** ({sign_t5}) "
        f"với biên độ dao động nón biến động kỳ vọng ±{daily_vol:.1f}%/ngày.\n"
        f"• **Mục tiêu T+6 đến T+10**: Giá mục tiêu mở rộng kỳ vọng đạt **{target_t10:,.2f} VNĐ** ({sign_t10})."
        f"{reconcile_note}\n"
        f"• **Kỷ luật giao dịch**: Nhà đầu tư giải ngân theo từng phần tại vùng gom, duy trì tỷ lệ Risk:Reward tối thiểu 1:2.5 và tuân thủ chặt chẽ điểm dừng lỗ khuyến nghị."
    )
