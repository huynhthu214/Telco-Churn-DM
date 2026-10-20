# Telco Customer Churn Prediction

Đồ án cuối kỳ học phần Data Mining (IT701220), Track B.

Dự án xây dựng và đánh giá các mô hình học máy nhằm dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông (*Customer Churn*). Ba thuật toán được sử dụng gồm K-Nearest Neighbors (KNN), Decision Tree và Naive Bayes. Quy trình thực hiện được tổ chức theo phương pháp CRISP-DM.

## 1. Dữ liệu

- **Nguồn dữ liệu:** [Telco Customer Churn trên Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Quy mô:** 7.043 dòng và 21 cột.
- **Tỷ lệ khách hàng rời mạng:** Khoảng 26,5%.
- **Đặc điểm:** Đây là bộ dữ liệu mẫu mô phỏng hoạt động của một công ty viễn thông giả định, thuộc nhóm dữ liệu Telco Customer Churn được phân phối rộng rãi cùng các tài liệu mẫu của IBM; không phải dữ liệu vận hành thực tế của một doanh nghiệp cụ thể.

## 2. Cấu trúc thư mục

```text
project-root/
├── data/                   # Dữ liệu đầu vào
├── notebook/               # Các notebook phân tích và mô hình hóa
│   ├── 01_EDA_TelcoChurn.ipynb
│   ├── 01b_Imbalance_Experiment.ipynb
│   └── 02_Modeling_Models.ipynb
├── figures/                # Biểu đồ sử dụng trong báo cáo
├── requirements.txt        # Các thư viện Python cần thiết
└── README.md
```

*Lưu ý: Cập nhật cây thư mục trên theo cấu trúc thực tế của repository, bao gồm thư mục dashboard và các file đầu ra nếu có.*

## 3. Cài đặt môi trường

**Yêu cầu:**
- Python: [điền phiên bản đã kiểm thử].
- Các thư viện được liệt kê trong `requirements.txt`.

Cài đặt thư viện bằng lệnh:

```bash
pip install -r requirements.txt
```

Nếu sử dụng Google Colab, tải notebook và dữ liệu lên môi trường Colab, sau đó thực hiện các bước cài đặt được hướng dẫn trong notebook.

## 4. Hướng dẫn thực hiện

Thực hiện các notebook theo thứ tự sau:

1. **`01_EDA_TelcoChurn.ipynb`**: Khám phá dữ liệu (EDA), xử lý dữ liệu, bao gồm cột `TotalCharges`, chia tập train/test và tạo `preprocessing_pipeline.pkl` nếu notebook có triển khai bước này.
2. **`01b_Imbalance_Experiment.ipynb`**: Thực nghiệm và so sánh các phương pháp xử lý mất cân bằng lớp, bao gồm không xử lý, sử dụng `class_weight` và SMOTE.
3. **`02_Modeling_Models.ipynb`**: Huấn luyện, tinh chỉnh và đánh giá ba mô hình KNN, Decision Tree và Naive Bayes; thực hiện các phân tích mở rộng được trình bày trong báo cáo.

**Lưu ý khi chạy lại:**
- Thực thi các notebook theo đúng thứ tự và kiểm tra các đường dẫn dữ liệu đầu vào.
- Bảo đảm những file được tạo ở bước trước đã tồn tại trước khi chạy bước tiếp theo.
- Nếu sử dụng SMOTE, chỉ áp dụng trên tập huấn luyện để tránh rò rỉ dữ liệu (*data leakage*).
- Nếu có sử dụng dashboard, bổ sung hướng dẫn khởi chạy và các file đầu vào cần thiết tại đây.

## 5. Thành viên và phân công

| Thành viên | Nhiệm vụ chính |
|---|---|
| Kim Khánh | Business Understanding, tổng hợp báo cáo |
| Minh Thư | Data & Preprocessing |
| Minh Tuấn | KNN, Decision Tree và đánh giá mô hình |
| Thúy Vi | Naive Bayes và Dashboard |

---