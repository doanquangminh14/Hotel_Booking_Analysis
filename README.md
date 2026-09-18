# Hotel Booking Analysis & Customer Segmentation Machine Learning

Dự án phân tích dữ liệu đặt phòng khách sạn (**Hotel Booking Demand Dataset**) kết hợp xây dựng mô hình **Machine Learning (Học không giám sát - Phân cụm K-Means & PCA)** nhằm tối ưu hóa doanh thu, quản trị rủi ro hủy phòng và cá nhân hóa trải nghiệm khách hàng.

---

## TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

Tập dữ liệu chứa **119,390 bản ghi** đặt phòng từ 2 loại hình khách sạn (**City Hotel** và **Resort Hotel**). Dự án được triển khai theo quy trình chuẩn của một bài toán Khoa học Dữ liệu thực tế:

1. **Khám phá & Kiểm tra chất lượng dữ liệu (Data Understanding & EDA)**.
2. **Tiền xử lý & Chuẩn hóa dữ liệu (Data Cleaning & Quality Assurance)**.
3. **Kỹ thuật tạo đặc trưng chuyên sâu (Feature Engineering Pipeline)**: Tạo 16 biến phái sinh đa chiều về thời gian, cơ cấu đoàn, tài chính và lịch sử tương tác.
4. **Khai phá quy luật kinh doanh & Trực quan hóa (Business Insights & Visualizations)**: Trả lời 8 bài toán trọng tâm về tỷ lệ hủy phòng, mùa vụ giá ADR, cơ cấu đoàn khách và thị trường quốc tế.
5. **Xây dựng mô hình Machine Learning Phân khúc Khách hàng (Customer Segmentation via K-Means & PCA)**:
   - Đánh giá lựa chọn số cụm tối ưu ($k=4$) qua 4 phương pháp kiểm định: Elbow/Inertia, Silhouette Score, Davies-Bouldin Index và Calinski-Harabasz Index.
   - Giảm chiều không gian bằng PCA (3 thành phần chính đầu tiên giải thích 50.6% tổng phương sai dữ liệu).
   - Phân tích ma trận trọng số đặc trưng (PCA Loadings Matrix) để diễn giải rõ ràng ý nghĩa từng trục biến thiên.
   - Định danh 4 nhóm chân dung khách hàng (Personas) qua Radar Chart và phân tích chỉ số đóng góp doanh thu (Revenue KPIs).
   - Thiết lập kiến trúc triển khai mô hình chấm điểm thời gian thực (Real-time Scoring Flow) và cơ chế giám sát độ lệch phân phối dữ liệu (Data Drift Monitoring).
6. **Đóng gói mã nguồn theo dạng Pipeline Module (`src/`) và hệ thống báo cáo chuyên nghiệp (`reports/`)**.

---

## KẾT QUẢ & PHÁT HIỆN CHÍNH (KEY HIGHLIGHTS)

* **Rủi ro Hủy phòng**: Tỷ lệ hủy phòng trung bình là **27.5%**. City Hotel có tỷ lệ hủy cao hơn Resort Hotel. Thời gian đặt trước (`lead_time`) tỷ lệ thuận với xác suất hủy (đạt gần **40%** (39.7%) đối với các đơn đặt trước trên 6 tháng so với **8.4%** ở nhóm đặt sát giờ 0-7 ngày).
* **Doanh thu & Mùa vụ**: Resort Hotel bùng nổ doanh thu vào mùa hè tháng 7-8 (ADR >150 EUR/đêm) nhưng suy giảm mạnh vào mùa đông. City Hotel duy trì giá phòng ổn định (~100-120 EUR/đêm) quanh năm.
* **Hành vi Khách hàng**: Khách đi theo **Gia đình (Family)** có mức chi tiêu phòng (ADR) cao nhất (~150 EUR). Khách có từ **1-2 yêu cầu đặc biệt** có tỷ lệ hủy phòng giảm xuống dưới 18% (so với 32.8% ở khách không có yêu cầu).
* **4 Phân khúc Khách hàng Machine Learning (K-Means $k=4$)**:
  * **Cluster 0: Gia đình nghỉ dưỡng cao cấp (9.3%)** — 100% có trẻ em, ADR cao nhất (~149.8 EUR), nhu cầu bãi xe cao nhất (18.8%), số yêu cầu đặc biệt cao nhất (1.12).
  * **Cluster 1: Cặp đôi tiêu chuẩn (55.3%)** — Nhóm khách chủ lực chiếm 48.5% tổng doanh thu, đi 2 người lớn, lưu trú 2.7 đêm, đặt chủ yếu qua OTA.
  * **Cluster 2: Khách nghỉ dưỡng dài ngày (15.8%)** — Lưu trú dài nhất (~7.9 đêm), đặt trước xa (~139.1 ngày), giá trị đơn đặt trung bình cao nhất (774.2 EUR/đơn).
  * **Cluster 3: Khách công tác & Khách quen (19.5%)** — Đi 1 mình (Solo), đặt gấp (`lead_time` ~36.2 ngày), tỷ lệ khách quen quay lại cao nhất (**16.5%**), chi tiêu bình quân đầu người cao nhất (77.1 EUR/người/đêm).

---

## CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Hotel_Booking_Analysis/
├── Data/
│   ├── hotel_bookings.csv                 # Dữ liệu gốc (119,390 dòng, 32 cột)
│   ├── hotel_bookings_cleaned.csv         # Dữ liệu đã làm sạch (87,223 dòng, 33 cột)
│   └── hotel_bookings_featured.csv        # Dữ liệu Feature Engineering cho ML (87,223 dòng, 53 cột)
│
├── notebook/
│   ├── understanding.ipynb                # Khám phá cấu trúc & kiểm tra chất lượng dữ liệu
│   ├── cleaning.ipynb                     # Quy trình làm sạch dữ liệu chi tiết
│   ├── eda.ipynb                          # Khai phá dữ liệu & trả lời bài toán kinh doanh
│   ├── feature_engineering.ipynb          # Tạo 16 đặc trưng phái sinh & phân tích tương quan
│   └── customer_clustering.ipynb          # Xây dựng mô hình Phân cụm khách hàng Machine Learning
│
├── src/
│   ├── __init__.py                        # Khởi tạo package Python
│   ├── extract.py                         # Module trích xuất và nạp dữ liệu tự động
│   ├── transform.py                       # Module làm sạch & Feature Engineering (ETL Pipeline)
│   ├── eda.py                             # Module tự động xuất 8 biểu đồ phân tích kinh doanh
│   └── ml.py                              # Module huấn luyện K-Means, PCA & xuất 7 biểu đồ ML và bảng Profile
│
├── reports/
│   ├── README.md                          # Mục lục tổng quan hệ thống báo cáo
│   ├── eda/
│   │   ├── README.md                      # Báo cáo chuyên sâu Khai phá Dữ liệu & 8 bài toán kinh doanh
│   │   └── figures/                       # 8 biểu đồ phân tích EDA (300 DPI)
│   └── ml/
│       ├── README.md                      # Báo cáo chuyên sâu Mô hình Machine Learning & 4 Personas
│       ├── cluster_profiles.csv           # Bảng thông số thống kê chi tiết từng cụm
│       └── figures/                       # 7 biểu đồ đánh giá mô hình ML (300 DPI)
│
├── requirements.txt                       # Danh mục thư viện và phiên bản phụ thuộc
├── .gitignore                             # Cấu hình bỏ qua các file rác và cache
└── README.md                              # Tài liệu tổng quan dự án
```

---

## DANH MỤC NOTEBOOKS & BÁO CÁO CHI TIẾT

### 1. Hệ thống Jupyter Notebooks (`notebook/`)
| Notebook | Mô tả nội dung |
| :--- | :--- |
| [`understanding.ipynb`](notebook/understanding.ipynb) | **Khám phá & Hiểu sâu dữ liệu**: Thống kê mô tả 32 biến, phân tích tỷ lệ Missing values, Duplicate và Anomalies. |
| [`cleaning.ipynb`](notebook/cleaning.ipynb) | **Làm sạch dữ liệu**: Xử lý 31,994 dòng trùng lặp, điền missing, lọc đơn 0 khách, chuẩn hóa ADR ngoại lai và ép kiểu dữ liệu. |
| [`feature_engineering.ipynb`](notebook/feature_engineering.ipynb) | **Kỹ thuật tạo đặc trưng**: Tạo 16 features mới (Thời gian, Mùa vụ, Đoàn khách, Lịch sử, Tài chính) và ma trận tương quan. |
| [`eda.ipynb`](notebook/eda.ipynb) | **Phân tích Khám phá EDA**: Trực quan hóa tỷ lệ hủy, mùa vụ giá phòng ADR, hiệu quả kênh phân phối và hành vi lưu trú. |
| [`customer_clustering.ipynb`](notebook/customer_clustering.ipynb) | **Machine Learning Phân cụm**: Tiền xử lý dữ liệu, tìm $k$ tối ưu bằng Elbow/Silhouette, giảm chiều PCA và định danh Personas. |

### 2. Hệ thống Báo cáo Chuyên sâu (`reports/`)
| Báo cáo | Mô tả nội dung |
| :--- | :--- |
| [**Báo cáo Khai phá Dữ liệu (EDA Report)**](reports/eda/README.md) | Diễn giải toàn diện 8 biểu đồ kinh doanh, phân tích rủi ro hủy phòng, chiến lược định giá mùa vụ và ma trận hành động. |
| [**Báo cáo Machine Learning (ML Report)**](reports/ml/README.md) | Quy trình phân cụm, đánh giá $k=4$, ma trận trọng số PCA Loadings, Radar Chart 4 Personas, chỉ số đóng góp doanh thu và kiến trúc triển khai thực tế (kèm 7 biểu đồ tại `reports/ml/figures/` và `cluster_profiles.csv`). |
| [**Mục lục Báo cáo (Reports Hub)**](reports/README.md) | Trang điều hướng trung tâm liên kết tất cả các báo cáo và biểu đồ trong dự án. |

---

## HƯỚNG DẪN CÀI ĐẶT & THỰC THI (HOW TO RUN)

### Bước 1: Clone repository và cài đặt thư viện
```bash
git clone https://github.com/doanquangminh14/Hotel_Booking_Analysis.git
cd Hotel_Booking_Analysis
pip install -r requirements.txt
```

### Bước 2: Chạy Pipeline Trích xuất & Chuyển đổi Dữ liệu (ETL)
```bash
# Trích xuất và kiểm tra dữ liệu
python src/extract.py

# Làm sạch, tạo feature và xuất file hotel_bookings_featured.csv
python src/transform.py
```

### Bước 3: Tự động khởi tạo Biểu đồ & Báo cáo
```bash
# Tạo 8 biểu đồ phân tích kinh doanh (lưu tại reports/eda/figures/)
python src/eda.py

# Huấn luyện mô hình ML và tạo 7 biểu đồ phân cụm (lưu tại reports/ml/figures/)
python src/ml.py
```

---

## CÔNG NGHỆ & THƯ VIỆN SỬ DỤNG (TECH STACK)

* **Ngôn ngữ**: Python 3.11+
* **Xử lý Dữ liệu**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn` (K-Means, PCA, StandardScaler, Metrics)
* **Trực quan hóa**: `matplotlib`, `seaborn`
* **Môi trường**: Jupyter Notebook, VS Code, Git / GitHub