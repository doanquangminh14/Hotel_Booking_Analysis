# BÁO CÁO MÔ HÌNH MACHINE LEARNING: PHÂN KHÚC KHÁCH HÀNG (CUSTOMER SEGMENTATION REPORT)

Báo cáo này trình bày chi tiết quy trình xây dựng, đánh giá và ứng dụng thực tiễn của mô hình **Học không giám sát (Machine Learning - Phân cụm K-Means & PCA)** nhằm phân nhóm và định danh các chân dung khách hàng (Customer Personas) trong kinh doanh khách sạn.

Tất cả các biểu đồ phân tích trong báo cáo được tự động khởi tạo bởi module [`src/ml.py`](../../src/ml.py) và lưu trữ tại thư mục [`reports/ml/figures/`](figures/). Dữ liệu thống kê chi tiết từng đặc trưng được lưu tại [`cluster_profiles.csv`](cluster_profiles.csv).

---

## 1. TỔNG QUAN BÀI TOÁN & MỤC TIÊU KINH DOANH

### 1.1 Bối cảnh và Thách thức
Trong quản trị doanh thu khách sạn (Revenue Management), việc áp dụng một chính sách giá và tiếp thị đại trà cho tất cả khách hàng dẫn đến lãng phí ngân sách tiếp thị, tỷ lệ hủy phòng cao và bỏ lỡ cơ hội gia tăng doanh thu trên mỗi phòng sẵn có (RevPAR).

### 1.2 Mục tiêu ứng dụng Machine Learning
1. **Tự động nhận diện phân khúc**: Phân nhóm khách hàng dựa trên hành vi lưu trú, thói quen đặt trước, quy mô đoàn và khả năng chi trả.
2. **Cá nhân hóa trải nghiệm & tiếp thị**: Thiết kế các gói sản phẩm, ưu đãi và dịch vụ bổ trợ phù hợp cho từng nhóm đối tượng cụ thể.
3. **Tối ưu hóa giá linh hoạt (Dynamic Pricing)**: Cung cấp chính sách giá, điều kiện hủy và số đêm lưu trú tối thiểu phù hợp theo từng phân khúc.
4. **Xây dựng chương trình khách hàng thân thiết (Loyalty Program)**: Giữ chân khách hàng tiềm năng và chuyển đổi khách đặt qua đại lý OTA sang đặt trực tiếp (Direct Booking).

### 1.3 Tập dữ liệu huấn luyện
* **Tập dữ liệu**: Lọc từ dữ liệu đã qua tiền xử lý và tạo đặc trưng, chỉ giữ lại các đơn đặt phòng **hoàn tất lưu trú thành công (`is_canceled == 0`)** với quy mô **63,219 bản ghi**.
* **Lý do lựa chọn**: Các đơn hoàn tất lưu trú phản ánh chính xác nhất hành vi tiêu dùng, thói quen sinh hoạt và doanh thu thực tế mà khách hàng mang lại cho khách sạn.

---

## 2. HỆ THỐNG ĐẶC TRƯNG & TIỀN XỬ LÝ DỮ LIỆU

Mô hình sử dụng **14 đặc trưng hành vi đa chiều** được trích xuất tự động qua pipeline [`src/transform.py`](../../src/transform.py):

| Nhóm đặc trưng | Tên biến | Ý nghĩa phân tích thực tế |
| :--- | :--- | :--- |
| **Thời gian & Đặt trước** | `lead_time`<br>`total_stay`<br>`stays_in_weekend_nights`<br>`stays_in_week_nights`<br>`weekend_stay_ratio` | Thời gian lên kế hoạch trước ngày nhận phòng, tổng số đêm lưu trú, số đêm ở ngày thường so với cuối tuần và tỷ trọng cuối tuần. |
| **Cơ cấu đoàn khách** | `total_guests`<br>`is_family`<br>`is_solo_traveler` | Quy mô nhóm khách, gia đình có trẻ em/trẻ sơ sinh hay khách đi công tác một mình. |
| **Tài chính & Chi tiêu** | `adr`<br>`adr_per_person` | Giá phòng trung bình mỗi đêm (ADR) và mức chi trả trung bình trên mỗi đầu người. |
| **Tương tác & Mức độ gắn kết** | `total_of_special_requests`<br>`required_car_parking_spaces`<br>`booking_changes`<br>`is_repeated_guest` | Số lượng yêu cầu đặc biệt (thể hiện mức độ quan tâm đến chuyến đi), nhu cầu bãi đỗ xe ô tô, số lần đổi lịch và lịch sử khách quay lại. |

### Quy trình tiền xử lý:
1. **Xử lý giá trị ngoại lai (Outlier Capping / Winsorization)**: Giới hạn các biến có giá trị phân tán lớn ở phân vị 99% (`lead_time`, `total_stay`, `adr`, `adr_per_person`, `booking_changes`) để tránh làm lệch tâm cụm.
2. **Chuẩn hóa dữ liệu (StandardScaler)**: Đưa toàn bộ 14 biến về cùng thang đo chuẩn (trung bình bằng 0, phương sai bằng 1) giúp thuật toán phân cụm đánh giá công bằng mức độ ảnh hưởng của từng đặc trưng.

---

## 3. ĐÁNH GIÁ & LỰA CHỌN SỐ CỤM TỐI ƯU (k = 4)

![Optimal k Evaluation](figures/01_optimal_k_evaluation.png)

Chúng ta đánh giá thực nghiệm số lượng cụm $k$ trong khoảng từ 2 đến 8 thông qua 4 tiêu chí đánh giá mô hình:

1. **Phương pháp Điểm uốn (Elbow Method / Inertia)**:
   * Tổng khoảng cách nội cụm giảm mạnh khi tăng từ $k=2$ lên $k=4$, sau đó tốc độ giảm chậm dần từ $k=5$ trở đi.
   * Điểm uốn xuất hiện rõ nét nhất tại vùng **$k = 4$**.

2. **Hệ số Silhouette (Silhouette Score)**:
   * Đo lường độ gắn kết giữa các điểm trong cùng một cụm và độ phân tách với các cụm khác.
   * Hệ số đạt mức cân bằng ổn định tại **$k = 4$** trên tập mẫu kiểm chứng.

3. **Chỉ số Davies-Bouldin (Giá trị càng nhỏ càng tốt)**:
   * Đo lường mức độ tương đồng giữa các cụm. Chỉ số giảm xuống mức tối ưu tại **$k = 4$** (~1.64), thể hiện các cụm có ranh giới phân tách rõ ràng.

4. **Chỉ số Calinski-Harabasz (Giá trị càng lớn càng tốt)**:
   * Đo lường tỷ lệ phân tán giữa các cụm so với nội cụm. Đạt giá trị vượt trội (>10,400 điểm) tại **$k = 4$**.

**Kết luận**: Lựa chọn **$k = 4$** mang lại chất lượng phân nhóm tốt nhất, đồng thời phản ánh trọn vẹn 4 nhóm hành vi kinh doanh rõ rệt trong thực tế.

---

## 4. PHÂN TÍCH GIẢM CHIỀU PCA & Ý NGHĨA TRỌNG SỐ ĐẶC TRƯNG

### 4.1 Tỷ lệ phương sai giải thích (Explained Variance)

![PCA Explained Variance](figures/02_pca_explained_variance.png)

* **Thành phần chính 1 (PC1)**: Giải thích **22.4%** phương sai dữ liệu.
* **Thành phần chính 2 (PC2)**: Giải thích **15.7%** phương sai dữ liệu.
* **Thành phần chính 3 (PC3)**: Giải thích **12.1%** phương sai dữ liệu.
* **Tổng tích lũy 3 PC đầu tiên**: Giải thích **50.2%** tổng biến thiên của 14 đặc trưng ban đầu (đạt ngưỡng phân tích trên 50%), giúp biểu diễn không gian phân cụm một cách tin cậy.

### 4.2 Ma trận trọng số đặc trưng (PCA Feature Loadings)

![PCA Feature Loadings](figures/06_pca_feature_loadings.png)

Ma trận trọng số giúp giải thích trực quan các trục phân bố:
* **Trục PC1 (Thời lượng & Kế hoạch)**: Đóng góp dương lớn nhất từ tổng số đêm ở (`total_stay`), số đêm trong tuần (`stays_in_week_nights`), đêm cuối tuần và thời gian đặt trước (`lead_time`). Trục này đại diện cho xu hướng nghỉ dưỡng dài ngày.
* **Trục PC2 (Đoàn khách & Chi tiêu)**: Đóng góp dương mạnh từ số lượng khách (`total_guests`), giá phòng (`adr`), biến gia đình (`is_family`) và đóng góp âm từ khách đi một mình (`is_solo_traveler`). Trục này phân hóa giữa khách gia đình chi tiêu cao và khách đi công tác đơn lẻ.
* **Trục PC3 (Cuối tuần & Yêu cầu)**: Phản ánh tỷ trọng đêm nghỉ cuối tuần và số lượng yêu cầu dịch vụ đặc biệt.

---

## 5. TRỰC QUAN HÓA KHÔNG GIAN PHÂN CỤM 2D PCA

![PCA 2D Clusters](figures/03_pca_2d_clusters.png)

Trên không gian 2 chiều PCA cùng các tâm cụm đại diện (đánh dấu X màu đen):
* **Cụm 0 (Màu đỏ - Gia đình cao cấp)**: Nằm ở phía trên (PC2 cao), tách biệt hoàn toàn nhờ quy mô đoàn lớn và giá phòng cao.
* **Cụm 1 (Màu xanh dương - Cặp đôi tiêu chuẩn)**: Tập trung ở trung tâm, đại diện cho tệp khách hàng phổ thông đông đảo nhất.
* **Cụm 2 (Màu xanh lá - Nghỉ dưỡng dài ngày)**: Trải dài về bên phải (PC1 cao), đại diện cho các kỳ nghỉ dài ngày.
* **Cụm 3 (Màu cam - Khách công tác & Khách quen)**: Nằm ở góc dưới bên trái (PC2 âm, PC1 thấp), đại diện cho khách đi một mình, đặt phòng gấp và lưu trú ngắn ngày.

---

## 6. BẢNG THỐNG KÊ CHI TIẾT 4 PHÂN KHÚC KHÁCH HÀNG

![Cluster Feature Distributions](figures/04_cluster_feature_distributions.png)

### Bảng so sánh các chỉ số trung bình giữa 4 Phân khúc:

| Chỉ số hành vi & Vận hành | Cụm 0: Gia đình cao cấp | Cụm 1: Cặp đôi tiêu chuẩn | Cụm 2: Nghỉ dưỡng dài ngày | Cụm 3: Công tác & Khách quen |
| :--- | :---: | :---: | :---: | :---: |
| **Quy mô đơn đặt (Số lượng / Tỷ lệ)** | 5,892 (9.3%) | **34,987 (55.3%)** | 10,004 (15.8%) | 12,336 (19.5%) |
| **Thời gian đặt trước (`lead_time`)** | 74.6 ngày | 61.4 ngày | **139.1 ngày** | 36.2 ngày |
| **Tổng số đêm lưu trú (`total_stay`)** | 3.5 đêm | 2.7 đêm | **7.9 đêm** | 2.1 đêm |
| **Số đêm cuối tuần / trong tuần** | 1.0 / 2.5 đêm | 0.7 / 2.0 đêm | **2.3 / 5.6 đêm** | 0.5 / 1.6 đêm |
| **Số khách trung bình (`total_guests`)** | **3.36 người** | 2.09 người | 2.01 người | 1.00 người |
| **Tỷ lệ có trẻ em (`is_family`)** | **99.8%** | 0.0% | 1.1% | 0.0% |
| **Giá phòng TB mỗi đêm (`adr`)** | **149.8 EUR** | 104.2 EUR | 98.4 EUR | 77.2 EUR |
| **Chi tiêu trên mỗi đầu người** | 44.5 EUR | 49.9 EUR | 49.3 EUR | **77.1 EUR** |
| **Nhu cầu bãi đỗ xe ô tô** | **18.8%** | 11.8% | 9.5% | 9.5% |
| **Số yêu cầu đặc biệt trung bình** | **1.12** | 0.82 | 0.76 | 0.43 |
| **Số lần thay đổi thông tin đặt phòng** | **0.49** | 0.22 | 0.42 | 0.38 |
| **Tỷ lệ khách quen quay lại** | 1.5% | 2.4% | 1.3% | **16.5%** |

---

## 7. CHỈ SỐ DOANH THU & HIỆU QUẢ KINH DOANH (BUSINESS KPIS)

![Cluster Business Metrics](figures/07_cluster_business_metrics.png)

1. **Tỷ trọng đóng góp doanh thu (Revenue Contribution)**:
   * **Cụm 1 (Cặp đôi tiêu chuẩn)**: Đóng góp lớn nhất vào tổng doanh thu phòng (**43.4%**) nhờ số lượng đặt phòng áp đảo.
   * **Cụm 2 (Nghỉ dưỡng dài ngày)**: Đóng góp **33.7%** tổng doanh thu dù chỉ chiếm 16.1% lượt đặt, nhờ thời gian lưu trú vượt trội (gần 8 đêm/đơn).
   * **Cụm 0 (Gia đình cao cấp)**: Đóng góp **13.7%** doanh thu với mức giá phòng trung bình cao nhất.
   * **Cụm 3 (Khách công tác)**: Đóng góp **9.2%** doanh thu lưu trú.

2. **Giá trị đơn đặt phòng trung bình (Average Booking Value)**:
   * **Cụm 2 (Nghỉ dưỡng dài ngày)**: Đạt **762.4 EUR / đơn đặt** (cao nhất toàn khách sạn).
   * **Cụm 0 (Gia đình cao cấp)**: Đạt **536.5 EUR / đơn đặt**.
   * **Cụm 1 (Cặp đôi tiêu chuẩn)**: Đạt **285.9 EUR / đơn đặt**.
   * **Cụm 3 (Khách công tác)**: Đạt **170.7 EUR / đơn đặt**.

3. **Tỷ lệ phụ thuộc kênh đại lý trực tuyến (Online TA Share)**:
   * **Cụm 0 (Gia đình cao cấp)**: **63.7%** đơn đặt qua Online TA.
   * **Cụm 1 (Cặp đôi tiêu chuẩn)**: **61.6%** đơn đặt qua Online TA.
   * **Cụm 2 (Nghỉ dưỡng dài ngày)**: **41.7%** đơn đặt qua Online TA.
   * **Cụm 3 (Khách công tác)**: **31.4%** đơn đặt qua Online TA (có tỷ trọng đặt trực tiếp và kênh doanh nghiệp cao nhất).

---

## 8. RADAR CHART & CHIẾN LƯỢC HÀNH ĐỘNG CHO 4 PERSONAS

![Radar Personas](figures/05_radar_personas.png)

---

### PHÂN KHÚC 0: GIA ĐÌNH NGHỈ DƯỠNG CAO CẤP (HIGH-VALUE VACATION FAMILIES)
* **Đặc điểm cốt lõi**:
  * 100% đoàn có trẻ nhỏ / trẻ sơ sinh, quy mô trung bình 3.36 người.
  * Giá phòng ADR cao nhất toàn khách sạn (**149.8 EUR/đêm**).
  * Nhu cầu bãi đỗ xe cao nhất (**18.8%**), số lượng yêu cầu đặc biệt cao nhất (**1.12 yêu cầu/đơn**).
* **Nhu cầu & Kỳ vọng**:
  * Cần không gian phòng rộng, phòng thông nhau (Connecting Rooms), môi trường an toàn và tiện nghi cho trẻ nhỏ.
  * Ưu tiên việc đỗ xe thuận tiện, thực đơn dinh dưỡng riêng và các hoạt động vui chơi giải trí cho bé.
* **Chiến lược tiếp thị & Vận hành**:
  * **Sản phẩm**: Thiết kế các gói Family Suite kết hợp vé tham quan công viên nước, khu giải trí trẻ em.
  * **Chính sách giá**: Miễn phí giường phụ (Extra bed) và bữa sáng cho trẻ em dưới 6 tuổi.
  * **Dịch vụ gia tăng**: Cung cấp dịch vụ trông trẻ (Babysitting), quà tặng chào mừng cho bé khi nhận phòng.

---

### PHÂN KHÚC 1: CẶP ĐÔI TIÊU CHUẨN (MID-TIER STANDARD COUPLES)
* **Đặc điểm cốt lõi**:
  * Chiếm quy mô lớn nhất (**55.3%** tổng lượng khách).
  * Đi theo cặp đôi (2 người lớn, không có trẻ em), lưu trú trung bình 2.7 đêm.
  * Thời gian đặt trước vừa phải (61.4 ngày), kênh đặt chủ yếu qua các nền tảng OTA.
* **Nhu cầu & Kỳ vọng**:
  * Nhạy cảm với mức giá phòng, quan tâm đến các đánh giá trực tuyến và vị trí thuận tiện để trải nghiệm ẩm thực, tham quan.
* **Chiến lược tiếp thị & Vận hành**:
  * **Chuyển đổi kênh (OTA sang Direct Booking)**: Tặng voucher giảm giá 10% cho lần đặt tiếp theo trên website chính thức hoặc tặng đồ uống chào mừng khi đặt trực tiếp.
  * **Gói trải nghiệm**: Cung cấp gói kỷ niệm ngày cưới/trăng mật (Romance Package) bao gồm bữa tối lãng mạn, rượu vang và trang trí phòng.

---

### PHÂN KHÚC 2: KHÁCH NGHỈ DƯỠNG DÀI NGÀY (LONG-STAY HOLIDAY PLANNERS)
* **Đặc điểm cốt lõi**:
  * Thời gian lưu trú dài nhất (**7.9 đêm**), lên kế hoạch trước rất xa (**139.1 ngày**).
  * Mang lại giá trị trung bình trên mỗi đơn đặt cao nhất (**774.2 EUR/đơn**).
  * Tập trung chủ yếu vào mùa hè tại các Resort Hotel.
* **Nhu cầu & Kỳ vọng**:
  * Cần các tiện nghi sinh hoạt dài ngày: Dịch vụ giặt ủi, ẩm thực đa dạng thay đổi theo ngày, các tour du lịch trải nghiệm địa phương.
* **Chiến lược tiếp thị & Vận hành**:
  * **Giá phòng lũy tiến**: Giảm giá 15% cho các đêm lưu trú từ đêm thứ 5 trở đi.
  * **Gói ẩm thực trọn gói**: Bán kèm gói Full-Board / Half-Board với thực đơn phong phú.
  * **Cam kết giữ phòng**: Cung cấp dịch vụ xe đưa đón sân bay 2 chiều miễn phí để tăng tính gắn kết, giảm thiểu rủi ro hủy phòng sát ngày.

---

### PHÂN KHÚC 3: KHÁCH CÔNG TÁC & KHÁCH QUEN (SOLO & BUSINESS LOYAL GUESTS)
* **Đặc điểm cốt lõi**:
  * Khách đi một mình (100% Solo traveler), lưu trú ngắn ngày (2.1 đêm trong tuần).
  * Thời gian đặt phòng rất gấp (`lead_time` trung bình 36.2 ngày).
  * Tỷ lệ khách quen quay lại cao vượt trội (**16.5%**, cao gấp 10 lần các nhóm khác).
  * Mức chi tiêu bình quân trên mỗi đầu người cao nhất (**77.1 EUR/người/đêm**).
* **Nhu cầu & Kỳ vọng**:
  * Cần sự nhanh chóng, thuận tiện: Thủ tục nhận/trả phòng nhanh, Wi-Fi tốc độ cao, không gian làm việc yên tĩnh, hỗ trợ xuất hóa đơn công ty.
* **Chiến lược tiếp thị & Vận hành**:
  * **Chương trình hội viên doanh nghiệp**: Tích lũy điểm thưởng nâng hạng phòng tự động, hỗ trợ nhận phòng sớm (Early check-in) và trả phòng muộn (Late check-out) linh hoạt.
  * **Ký kết hợp đồng đối tác**: Thiết lập bảng giá hợp đồng doanh nghiệp cố định (Corporate Contract Rates) quanh năm.

---

## 9. KIẾN TRÚC TRIỂN KHAI VÀO HỆ THỐNG THỰC TẾ (MLOPS & SERVING)

### 9.1 Quy trình chấm điểm thời gian thực (Scoring Pipeline)
Khi một đơn đặt phòng mới được tạo trên hệ thống Quản lý Khách sạn (PMS / CRM):

1. **Thu thập dữ liệu đơn đặt**: Tiếp nhận thông tin về ngày đến, số đêm, cơ cấu khách, giá phòng và các yêu cầu đi kèm.
2. **Trích xuất đặc trưng**: Module [`src/transform.py`](../../src/transform.py) tự động tính toán 14 biến hành vi và chuẩn hóa qua bộ `StandardScaler` đã huấn luyện.
3. **Phân loại phân khúc**: Mô hình K-Means gán nhãn cụm (Cluster ID) và định danh Persona tương ứng.
4. **Kích hoạt hành động tự động**:
   - Hệ thống CRM gửi email xác nhận kèm các gợi ý tiện ích phù hợp với nhóm khách.
   - Bộ phận Lễ tân và Buồng phòng chuẩn bị trước phòng nghỉ theo tiêu chuẩn của từng Persona.
   - Hệ thống Quản trị Doanh thu đưa ra đề xuất nâng cấp hạng phòng (Upsell) phù hợp.

### 9.2 Giám sát và Tái huấn luyện mô hình (Monitoring & Retraining)
* **Giám sát độ lệch dữ liệu (Data Drift)**: Định kỳ hàng tháng theo dõi chỉ số ổn định phân phối (Population Stability Index - PSI) trên 14 đặc trưng để phát hiện sớm sự thay đổi trong hành vi đặt phòng.
* **Lịch tái huấn luyện**: Cập nhật lại mô hình định kỳ mỗi quý hoặc trước các đợt cao điểm mùa du lịch nhằm thích ứng với biến động thị trường.
