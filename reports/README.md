# HE THONG BAO CAO PHAN TICH & MACHINE LEARNING (PROJECT REPORTS HUB)

Thu muc `reports/` duoc to chuc thanh cac phan bao cao chuyen biet, moi bao cao di kem thu muc hinh anh truc quan hoa do phan giai cao (300 DPI) va tai lieu phan tich kinh doanh rieng biet:

---

## CAU TRUC BAO CAO

| Thu muc Bao cao | File bao cao chi tiet | Script tao bieu do | Mo ta noi dung |
| :--- | :--- | :--- | :--- |
| [`reports/eda/`](eda/) | [`reports/eda/README.md`](eda/README.md) | [`src/eda.py`](../src/eda.py) | **Khai pha Du lieu & Insights Kinh doanh**: Phan tich ty le huy phong (theo kenh, thoi gian dat truoc, chinh sach coc), mua vu & gia phong ADR, co cau nhom khach hang va thi truong quoc te (kem 8 bieu do tai `eda/figures/`). |
| [`reports/ml/`](ml/) | [`reports/ml/README.md`](ml/README.md) | [`src/ml.py`](../src/ml.py) | **Mo hinh Machine Learning Phan khuc Khach hang**: Danh gia so cum toi uu ($k=4$), giam chieu khong gian PCA, ma tran trong so dac trung, dinh danh 4 Personas, chi so dong gop doanh thu va kien truc MLOps (kem 7 bieu do tai `ml/figures/` va `cluster_profiles.csv`). |

---

## HUONG DAN TAO LAI TAT CA BIEU DO

Ban co the chay cac script Python doc lap de tu dong khoi tao lai toan bo hinh anh trong tung thu muc bao cao:

1. **Khoi tao lai bieu do phan tich kinh doanh (EDA)**:
   ```bash
   python src/eda.py
   ```
   *Hinh anh se duoc luu tai:* `reports/eda/figures/`

2. **Khoi tao lai bieu do mo hinh Machine Learning**:
   ```bash
   python src/ml.py
   ```
   *Hinh anh va bang thong ke se duoc luu tai:* `reports/ml/figures/` va `reports/ml/cluster_profiles.csv`
