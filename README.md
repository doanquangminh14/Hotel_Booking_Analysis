# 🏨 Hotel Booking Analysis

Dự án phân tích dữ liệu đặt phòng khách sạn (**Hotel Booking Demand Dataset**).

---

## 📁 Cấu trúc thư mục & Notebooks

| File | Mô tả nội dung |
| :--- | :--- |
| [`eda.ipynb`](eda.ipynb) | **Khai phá Dữ liệu & Trả lời Bài toán Kinh doanh (EDA & Business Insights)**: Phân tích trực quan hóa chuyên sâu về tỷ lệ hủy phòng, biến động giá phòng ADR, hiệu quả doanh thu theo kênh và phân khúc, chân dung hành vi khách hàng cùng các đề xuất chiến lược tối ưu vận hành. |
| [`understanding.ipynb`](understanding.ipynb) | **Khám phá & Hiểu sâu dữ liệu (EDA ban đầu)**: Kiểm tra cấu trúc 32 cột dữ liệu, thống kê mô tả, kiểm tra sức khỏe dữ liệu (Missing values, Duplicates, Anomalies). |
| [`cleaning.ipynb`](cleaning.ipynb) | **Quy trình làm sạch dữ liệu (Data Cleaning & Preprocessing)**: Xử lý dòng trùng lặp, xử lý missing values (`company`, `agent`, `country`, `children`), loại bỏ đơn 0 khách, chuẩn hóa ngoại lai ADR, chuyển đổi kiểu dữ liệu (`int64`, `datetime`) và tạo các đặc trưng mới (`total_stay`, `total_guests`, `is_room_changed`, `is_family`, `total_cost`). Dữ liệu sạch được lưu trong biến bộ nhớ `df_clean`. |
| `Data/hotel_bookings.csv` | File dữ liệu gốc chứa 119,390 bản ghi đặt phòng khách sạn. |