# 📈 VN-Stock AI Pro - Hệ Thống Phân Tích & Dự Báo Đường Giá Chứng Khoán Việt Nam 24/7

> Nền tảng phân tích kỹ thuật hành động giá (Price Action) và dự báo đường giá (Price Path Prediction T+1 đến T+10) tích hợp AI dành riêng cho thị trường chứng khoán Việt Nam (HOSE, HNX, UPCOM).

---

## 🌟 Điểm Nổi Bật

- **Biểu đồ thời gian thực TradingView Lightweight Charts**: Nến Nhật chuyên nghiệp, hiển thị khối lượng, SMA 20, SMA 50, Bollinger Bands, RSI(14), MACD.
- **AI Đa Nhân Tố (Multi-factor AI Scoring)**: Chấm điểm kỹ thuật 0–100, xác định vùng gom hàng (Entry Zone), mục tiêu chốt lời (TP1, TP2), điểm cắt lỗ (Stop Loss) và tỷ lệ Lời/Lỗ (Risk:Reward).
- **Dự Báo Đường Giá T+1 đến T+10 (AI Price Path Projection)**: Mô phỏng nón biến động (Volatility Cone) với 3 kịch bản: Cơ sở (Base Case), Lạc quan (Bull Case +1.65σ), Bi quan (Bear Case -1.65σ) vẽ trực tiếp lên biểu đồ.
- **Top 5 Cổ Phiếu Tiềm Năng**: Tự động lọc các mã có dòng tiền vào mạnh, xác suất sinh lời cao nhất.
- **Trợ Lý AI 1-kèm-1**: Tư vấn chiến lược, vùng kháng cự/hỗ trợ, giải đáp mã cổ phiếu ngay lập tức.
- **Không Cần Bật Máy Tính 24/7**: Ứng dụng chạy trên Cloud độc lập, máy tính cá nhân tắt vẫn online bình thường.

---

## 🚀 Hướng Dẫn Triển Khai Chạy Online 24/7 (Miễn Phí)

### Cách 1: Triển khai 1-Click lên Render.com (Khuyên dùng - Ổn định nhất)
1. Đăng ký tài khoản miễn phí tại [render.com](https://render.com) (Đăng nhập bằng tài khoản GitHub).
2. Nhấn nút **"New +"** ở góc trên chọn **"Web Service"**.
3. Chọn Repository GitHub của bạn.
4. Render sẽ tự động đọc tệp `render.yaml` và cấu hình:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Nhấn **"Deploy Web Service"**.
6. Sau 2-3 phút, bạn sẽ có link vĩnh viễn dạng: `https://ten-ung-dung.onrender.com` chạy 24/24 kể cả khi tắt máy tính!

---

### Cách 2: Triển khai lên Hugging Face Spaces (Chạy 24/7 vĩnh viễn)
1. Truy cập [huggingface.co/spaces](https://huggingface.co/spaces) và tạo Space mới.
2. Chọn **Space SDK: Docker** (hoặc Blank).
3. Đẩy code lên Space hoặc kết nối GitHub.
4. Hugging Face sẽ tự động build từ `Dockerfile` và cấp link HTTPS 24/7.

---

## 💻 Hướng Dẫn Chạy Cục Bộ (Local PC)

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Khởi chạy máy chủ
python main.py
```
Mở trình duyệt truy cập: `http://localhost:8000`
