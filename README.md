# 🏨 Hotel Booking Analysis

Dự án phân tích dữ liệu đặt phòng khách sạn (**Hotel Booking Demand Dataset**).

---

## 📁 Cấu trúc thư mục & Notebooks

| File | Mô tả nội dung |
| :--- | :--- |
| [`understanding.ipynb`](understanding.ipynb) | **Khám phá & Hiểu sâu dữ liệu (EDA)**: Kiểm tra cấu trúc 32 cột dữ liệu, thống kê mô tả, kiểm tra sức khỏe dữ liệu (Missing values, Duplicates, Anomalies), phân tích tỷ lệ hủy phòng, tính mùa vụ, thời gian đặt trước (lead time), giá phòng trung bình (ADR), phân khúc thị trường và phân bố địa lý khách hàng. |
| [`cleaning.ipynb`](cleaning.ipynb) | **Quy trình làm sạch dữ liệu (Data Cleaning & Preprocessing)**: Xử lý dòng trùng lặp, xử lý missing values (`company`, `agent`, `country`, `children`), loại bỏ đơn 0 khách, chuẩn hóa ngoại lai ADR, chuyển đổi kiểu dữ liệu (`int64`, `datetime`) và tạo các đặc trưng mới (`total_stay`, `total_guests`, `is_room_changed`, `is_family`, `total_cost`). Dữ liệu sạch được lưu trong biến bộ nhớ `df_clean`. |
| `Data/hotel_bookings.csv` | File dữ liệu gốc chứa 119,390 bản ghi đặt phòng khách sạn. |