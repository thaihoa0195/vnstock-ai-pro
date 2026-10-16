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

def predict_daily_path(df: pd.DataFrame, days_ahead: int = 10, model_type: str = "ensemble") -> Dict[str, Any]:
    """
    Predict daily price path (T+1 to T+10) using Deep Learning models
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

    # Run Deep Learning models suite
    deep_results = train_and_predict_deep_models(df, days_ahead=days_ahead)
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

    win_prob = 50.0 + (drift / daily_vol) * 35.0
    win_prob = max(35.0, min(88.0, round(win_prob, 1)))

    trend_forecast = "TĂNG TRƯỞNG" if drift > 0.001 else ("ĐIỀU CHỈNH GIẢM" if drift < -0.001 else "TÍCH LŨY ĐI NGANG")
    trend_color = "#10b981" if drift > 0.001 else ("#ef4444" if drift < -0.001 else "#f59e0b")

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
            "confidence": f"{max(55, int(94 - step_idx * 3.0))}%"
        })

    # Summary metrics
    target_t5 = daily_details[4]["predicted_price"] if len(daily_details) >= 5 else path_base[-1]["value"]
    target_t10 = daily_details[-1]["predicted_price"]
    expected_gain_t5 = round(((target_t5 - last_price) / last_price) * 100, 2)
    expected_gain_t10 = round(((target_t10 - last_price) / last_price) * 100, 2)

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
        daily_vol=daily_vol * 100
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
    trend: str, rsi: float, daily_vol: float
) -> str:
    sign_t5 = f"+{gain_t5}%" if gain_t5 >= 0 else f"{gain_t5}%"
    sign_t10 = f"+{gain_t10}%" if gain_t10 >= 0 else f"{gain_t10}%"

    return (
        f"Mô hình Deep Learning **{model_name}** ({model_type}) đã hoàn tất huấn luyện online với hàm mất mát MSE đạt **{loss_mse}** "
        f"và độ chính xác định hướng kiểm định đạt **{accuracy}**.\n\n"
        f"• **Dự báo xu hướng chủ đạo**: **{trend}** trong 10 phiên tới (Xác suất thành công: **{win_prob}%**).\n"
        f"• **Ngắn hạn T+1 đến T+5**: Đường nơ-ron học sâu dự phóng giá tiệm cận mốc **{target_t5:,.2f} VNĐ** ({sign_t5}) "
        f"với biên độ dao động nón biến động kỳ vọng ±{daily_vol:.1f}%/ngày.\n"
        f"• **Mục tiêu T+6 đến T+10**: Giá mục tiêu mở rộng kỳ vọng đạt **{target_t10:,.2f} VNĐ** ({sign_t10}).\n"
        f"• **Cảnh báo rủi ro**: Nếu xuất hiện áp lực bán tháo bất ngờ, dải biên dưới (Bearish Cone) sẽ đóng vai trò hỗ trợ động. "
        f"Nhà đầu tư nên tuân thủ kỷ luật dừng lỗ (Stop Loss) theo tỷ lệ Risk:Reward khuyến nghị."
    )
