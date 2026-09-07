# 🏨 Hotel Booking Analysis

Dự án phân tích dữ liệu đặt phòng khách sạn (**Hotel Booking Demand Dataset**).

---

## 📁 Cấu trúc thư mục & Notebooks

| File | Mô tả nội dung |
| :--- | :--- |
| [`feature_engineering.ipynb`](notebook/feature_engineering.ipynb) | **Tiền xử lý & Kỹ thuật tạo đặc trưng (Feature Engineering Pipeline)**: Làm sạch dữ liệu chuẩn hóa, tạo 16 đặc trưng phái sinh chuyên sâu (Thời gian, Mùa vụ, Đoàn khách, Hành vi, Tài chính) và xuất dữ liệu `hotel_bookings_featured.csv` sẵn sàng cho Machine Learning. |
| [`customer_clustering.ipynb`](notebook/customer_clustering.ipynb) | **Phân khúc khách hàng bằng Machine Learning (Customer Segmentation via Clustering)**: Tiền xử lý dữ liệu, giảm chiều PCA, tối ưu hóa số cụm bằng Elbow & Silhouette, huấn luyện K-Means ($k=4$), phân tích chân dung khách hàng (Personas) và đề xuất chiến lược kinh doanh thực tế. |
| [`eda.ipynb`](notebook/eda.ipynb) | **Khai phá Dữ liệu & Trả lời Bài toán Kinh doanh (EDA & Business Insights)**: Phân tích trực quan hóa chuyên sâu về tỷ lệ hủy phòng, biến động giá phòng ADR, hiệu quả doanh thu theo kênh và phân khúc, chân dung hành vi khách hàng cùng các đề xuất chiến lược tối ưu vận hành. |
| [`understanding.ipynb`](notebook/understanding.ipynb) | **Khám phá & Hiểu sâu dữ liệu (EDA ban đầu)**: Kiểm tra cấu trúc 32 cột dữ liệu, thống kê mô tả, kiểm tra sức khỏe dữ liệu (Missing values, Duplicates, Anomalies). |
| [`cleaning.ipynb`](notebook/cleaning.ipynb) | **Quy trình làm sạch dữ liệu (Data Cleaning & Preprocessing)**: Xử lý dòng trùng lặp, xử lý missing values (`company`, `agent`, `country`, `children`), loại bỏ đơn 0 khách, chuẩn hóa ngoại lai ADR, chuyển đổi kiểu dữ liệu (`int64`, `datetime`) và tạo các đặc trưng mới (`total_stay`, `total_guests`, `is_room_changed`, `is_family`, `total_cost`). Dữ liệu sạch được lưu trong biến bộ nhớ `df_clean`. |
| `Data/hotel_bookings.csv` | File dữ liệu gốc chứa 119,390 bản ghi đặt phòng khách sạn. |
| `Data/hotel_bookings_featured.csv` | File dữ liệu đã qua tiền xử lý & Feature Engineering (87,223 dòng, 48 cột) sẵn sàng cho Machine Learning. |