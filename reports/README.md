# 📊 HỆ THỐNG BÁO CÁO PHÂN TÍCH & MACHINE LEARNING (PROJECT REPORTS HUB)

Thư mục `reports/` được chia thành các phần báo cáo chuyên biệt, mỗi báo cáo đi kèm thư mục hình ảnh trực quan hóa độ phân giải cao và tài liệu phân tích kinh doanh riêng biệt:

---

## 📁 CẤU TRÚC BÁO CÁO

| Thư mục Báo cáo | File báo cáo chi tiết | Script tạo biểu đồ | Mô tả nội dung |
| :--- | :--- | :--- | :--- |
| [`reports/eda/`](eda/) | [`reports/eda/README.md`](eda/README.md) | [`src/eda.py`](../src/eda.py) | **Khai phá Dữ liệu & Insights Kinh doanh**: Phân tích tỷ lệ hủy phòng (theo kênh, thời gian đặt trước, chính sách cọc), mùa vụ & giá phòng ADR, cơ cấu nhóm khách hàng và thị trường quốc tế (kèm 8 biểu đồ tại `eda/figures/`). |
| [`reports/ml/`](ml/) | [`reports/ml/README.md`](ml/README.md) | [`src/ml.py`](../src/ml.py) | **Mô hình Machine Learning Phân khúc Khách hàng**: Đánh giá số cụm tối ưu ($k=4$), phân tích giảm chiều PCA, định danh 4 nhóm Personas qua Radar Chart và hướng dẫn tích hợp thực tế (kèm 5 biểu đồ tại `ml/figures/`). |

---

## 🚀 HƯỚNG DẪN TẠO LẠI TẤT CẢ BIỂU ĐỒ

Bạn có thể chạy các script Python độc lập để tự động khởi tạo lại toàn bộ hình ảnh trong từng thư mục báo cáo:

1. **Khởi tạo lại biểu đồ phân tích kinh doanh (EDA)**:
   ```bash
   python src/eda.py
   ```
   *Hình ảnh sẽ được lưu tại:* `reports/eda/figures/`

2. **Khởi tạo lại biểu đồ mô hình Machine Learning**:
   ```bash
   python src/ml.py
   ```
   *Hình ảnh sẽ được lưu tại:* `reports/ml/figures/`
