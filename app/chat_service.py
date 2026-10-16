"""
AI Stock Assistant Q&A Engine
"""
import requests
from typing import Dict, Any, Optional

SYSTEM_PROMPT = """Bạn là Chuyên gia Cao cấp Phân tích Kỹ thuật & Price Action Chứng khoán Việt Nam (VN-Stock AI Pro).
Nhiệm vụ của bạn là giải đáp chi tiết, khách quan, chuyên nghiệp các câu hỏi của nhà đầu tư về hành động giá, vùng cung cầu, quản trị rủi ro, điểm mua/bán tối ưu dựa trên dữ liệu kỹ thuật thực tế được cung cấp.

Quy tắc phân tích:
1. Luôn đề cao QUẢN TRỊ RỦI RO (Stop Loss, tỷ lệ R:R tối thiểu 1:1.5 hoặc 1:2).
2. Kết hợp đa khung: Xu hướng (MA20, MA50, MA200), Động lượng (RSI, MACD), Dòng tiền (Khối lượng Vol so với TB20), và Mẫu hình nến Price Action.
3. Luôn đưa ra các mốc giá rõ ràng (vùng mua, chốt lời TP1, TP2, cắt lỗ SL) thay vì nói chung chung.
4. Trả lời bằng tiếng Việt gãy gọn, mạch lạc, dùng các bullet point dễ đọc.
"""

def generate_ai_chat_response(
    symbol: str,
    question: str,
    analysis_data: Dict[str, Any],
    api_key: Optional[str] = None
) -> str:
    """
    Generate contextual AI response using Gemini API or built-in Expert Rule-based Engine
    """
    if api_key and api_key.strip():
        # Call Google Gemini API
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
            context = f"""Dữ liệu kỹ thuật hiện tại của cổ phiếu {symbol}:
- Giá hiện tại: {analysis_data.get('current_price')} VNĐ
- Điểm đánh giá AI: {analysis_data.get('score')}/100 ({analysis_data.get('rating')})
- Khuyến nghị: {analysis_data.get('action')}
- Vùng mua: {analysis_data.get('entry_zone')}
- Cắt lỗ (SL): {analysis_data.get('stop_loss')}
- Chốt lời 1 (TP1): {analysis_data.get('target_1')}
- Chốt lời 2 (TP2): {analysis_data.get('target_2')}
- Tỷ lệ R/R: {analysis_data.get('risk_reward')}
- RSI: {analysis_data.get('indicators', {}).get('rsi')}
- MACD: {analysis_data.get('indicators', {}).get('macd')} (Signal: {analysis_data.get('indicators', {}).get('macd_signal')})
- Khối lượng so với TB20: {analysis_data.get('indicators', {}).get('vol_ratio')}x
- Hỗ trợ S1/S2: {analysis_data.get('levels', {}).get('support_1')} / {analysis_data.get('levels', {}).get('support_2')}
- Kháng cự R1/R2: {analysis_data.get('levels', {}).get('resistance_1')} / {analysis_data.get('levels', {}).get('resistance_2')}
- Mẫu hình nến: {[p.get('name') for p in analysis_data.get('patterns', [])]}

Câu hỏi của người dùng: "{question}"
"""
            payload = {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": context}]}]
            }
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                result = res.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            print(f"Gemini API call failed: {e}, falling back to built-in AI engine")

    # Built-in High Intelligence Price Action Engine
    q = question.lower()
    c = analysis_data.get("current_price", 0)
    score = analysis_data.get("score", 50)
    rating = analysis_data.get("rating", "QUAN SÁT")
    sl = analysis_data.get("stop_loss", 0)
    tp1 = analysis_data.get("target_1", 0)
    tp2 = analysis_data.get("target_2", 0)
    entry = analysis_data.get("entry_zone", f"{c}")
    indicators = analysis_data.get("indicators", {})
    rsi = indicators.get("rsi", 50)
    vol_ratio = indicators.get("vol_ratio", 1.0)
    levels = analysis_data.get("levels", {})
    s1 = levels.get("support_1", c * 0.97)
    r1 = levels.get("resistance_1", c * 1.05)
    patterns = analysis_data.get("patterns", [])
    pattern_names = [p.get("name") for p in patterns] if patterns else []

    if any(k in q for k in ["mua", "vào giá", "điểm vào", "entry", "gom"]):
        if score >= 60:
            return (
                f"### 🎯 Khuyến nghị Vị thế Mua cho {symbol}:\n"
                f"- **Đánh giá AI**: **{rating}** (Điểm kỹ thuật `{score}/100`). Trạng thái kỹ thuật đang ủng hộ phe Mua.\n"
                f"- **Vùng mua an toàn**: **{entry} VNĐ** (Nên giải ngân từng phần quanh hỗ trợ S1 `{s1:,.2f}`).\n"
                f"- **Mục tiêu chốt lời**: TP1 tại **{tp1:,.2f}** (+{((tp1-c)/c)*100:.1f}%), TP2 tại **{tp2:,.2f}** (+{((tp2-c)/c)*100:.1f}%).\n"
                f"- **Điểm dừng lỗ (SL)**: Dứt khoát đóng vị thế nếu thủng **{sl:,.2f}** (-{((c-sl)/c)*100:.1f}%).\n"
                f"- **Chiến lược**: Không nên fomo mua đuổi giá xanh mạnh, hãy kiên nhẫn chờ các nhịp test lại hỗ trợ trong phiên."
            )
        else:
            return (
                f"### ⚠️ Khuyến nghị Thận trọng về Điểm Mua {symbol}:\n"
                f"- **Đánh giá AI**: **{rating}** (Điểm kỹ thuật chỉ đạt `{score}/100`).\n"
                f"- Cổ phiếu hiện chưa hình thành nền giá an toàn. RSI ở mức `{rsi:.1f}`, động lực tăng ngắn hạn còn yếu.\n"
                f"- **Lời khuyên**: Nhà đầu tư **KHÔNG NÊN VỘI MỞ VỊ THẾ MUA MỚI**. Hãy kiên nhẫn quan sát phản ứng giá tại vùng hỗ trợ cứng `{s1:,.2f}`.\n"
                f"- Chỉ cân nhắc tham gia khi xuất hiện nến rút chân xác nhận đảo chiều và thanh khoản cạn kiệt."
            )

    elif any(k in q for k in ["bán", "cắt lỗ", "chốt lời", "thoát", "giữ hay bán", "sl", "tp"]):
        return (
            f"### 🛡️ Chiến lược Chốt lời & Cắt lỗ cho {symbol}:\n"
            f"- **Giá hiện tại**: `{c:,.2f} VNĐ`\n"
            f"- **Vùng Cắt lỗ (Stop Loss)**: **{sl:,.2f} VNĐ**. Nếu nến ngày đóng cửa thủng mốc này, nên hạ tỷ trọng để tránh kẹt vốn sâu.\n"
            f"- **Vùng Chốt lời từng phần (Take Profit)**:\n"
            f"  + **Mục tiêu 1 (TP1)**: `{tp1:,.2f}` (Kháng cự gần, nên hiện thực hóa 30% - 50% lợi nhuận).\n"
            f"  + **Mục tiêu 2 (TP2)**: `{tp2:,.2f}` (Đỉnh sóng cũ / Kháng cự mạnh R2).\n"
            f"- **Khuyến nghị vị thế**: {analysis_data.get('action')}."
        )

    elif any(k in q for k in ["kháng cự", "hỗ trợ", "cản", "đáy", "đỉnh"]):
        return (
            f"### 📐 Các Mốc Cung - Cầu Trọng Yếu Của {symbol}:\n"
            f"- **Kháng cự R1 (Cản gần)**: `{r1:,.2f} VNĐ`\n"
            f"- **Kháng cự R2 (Cản mạnh)**: `{levels.get('resistance_2', r1 * 1.05):,.2f} VNĐ`\n"
            f"- **Hỗ trợ S1 (Đệm giá ngắn hạn)**: `{s1:,.2f} VNĐ`\n"
            f"- **Hỗ trợ S2 (Hỗ trợ cứng đáy cũ)**: `{levels.get('support_2', s1 * 0.95):,.2f} VNĐ`\n"
            f"- **Fibonacci Retracement 50%**: `{levels.get('fib_500', c):,.2f} VNĐ`\n"
            f"💡 *Nhận định*: Vùng `{s1:,.2f}` đóng vai trò là mốc phòng thủ quan trọng nhất của phe bò trong ngắn hạn."
        )

    elif any(k in q for k in ["dòng tiền", "vol", "thanh khoản", "lái", "gom hàng", "smart money"]):
        flow_eval = "Đang có dòng tiền lớn nhập cuộc tích cực!" if vol_ratio > 1.3 else ("Thanh khoản cạn kiệt, phe bán không còn nhiều lực." if vol_ratio < 0.7 else "Thanh khoản ở mức trung bình, dòng tiền thận trọng tích lũy.")
        return (
            f"### 🌊 Phân tích Dòng tiền & Khối lượng (Volume Flow) {symbol}:\n"
            f"- **Tỷ lệ Vol hiện tại / TB 20 phiên**: `{vol_ratio:.2f}x`.\n"
            f"- **Đánh giá Dòng tiền**: **{flow_eval}**\n"
            f"- **Hành vi Giá & Khối lượng**: {'Giá tăng kèm vol đột biến cho thấy lực hấp thụ cung cực tốt.' if vol_ratio > 1.2 and score >= 60 else 'Khối lượng giao dịch thấp cho thấy sự giằng co, cung cầu đang chờ đợi tín hiệu bứt phá rõ nét.'}\n"
            f"- **Chiến lược theo dòng tiền**: Bám sát đường MA20, gia tăng tỷ trọng khi có phiên bứt phá vượt cản `{r1:,.2f}` với thanh khoản bùng nổ."
        )

    # General fallback answer
    patterns_str = ", ".join(pattern_names) if pattern_names else "Không có mẫu hình đột biến"
    return (
        f"### 🤖 Báo Cáo Phân Tích Tổng Quan Cho {symbol}:\n"
        f"- **Đánh giá AI**: **{rating}** | Điểm số: **{score}/100**\n"
        f"- **Giá hiện tại**: `{c:,.2f} VNĐ`\n"
        f"- **Chỉ báo cốt lõi**:\n"
        f"  + RSI (14): `{rsi:.1f}` ({'Quá bán' if rsi < 30 else ('Quá mua' if rsi > 70 else 'Cân bằng')})\n"
        f"  + MACD: `{'MACD > Signal (Tích cực)' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'MACD < Signal (Cần theo dõi)'}`\n"
        f"  + Mẫu hình nến: **{patterns_str}**\n"
        f"- **Kế hoạch Giao dịch**:\n"
        f"  + Vùng mua: **{entry}**\n"
        f"  + Chốt lời kỳ vọng: **{tp1:,.2f}** - **{tp2:,.2f}**\n"
        f"  + Ngưỡng cắt lỗ: **{sl:,.2f}**\n"
        f"- *Gợi ý thêm*: Bạn có thể hỏi cụ thể hơn như *'vùng mua an toàn'*, *'các mốc hỗ trợ kháng cự'*, hoặc *'phân tích dòng tiền'*"
    )
