# Customer Churn Analytics — Telco Customer Churn

Đồ án cuối kỳ Data Mining (Track B: Applied Project). Dự đoán khách hàng rời bỏ dịch vụ viễn thông bằng 3 thuật toán **KNN**, **Decision Tree**, **Naive Bayes**, kèm dashboard demo bằng Streamlit.

> Dữ liệu là bộ mẫu IBM Telco Customer Churn. Các con số, ngưỡng và giả định chi phí chỉ nhằm minh họa phương pháp, không áp dụng trực tiếp cho doanh nghiệp thực tế.

## Cấu trúc thư mục

```
Telco-Churn-DM/
├── app.py                  # Dashboard Streamlit (3 tab)
├── test_app.py             # Kiểm thử tự động (pytest + streamlit AppTest)
├── config.json             # Ngưỡng, giả định chi phí, cutoff phân khúc
├── requirements.txt        # Phụ thuộc (đã pin phiên bản)
├── README.md
├── .streamlit/
│   └── config.toml         # Màu chủ đạo của giao diện
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dữ liệu gốc (tự đặt vào)
│   └── ...                 # pipeline, .npy, y_*.csv do notebook 01 tạo ra
├── models/                 # do notebook 02 tạo ra
│   ├── knn.pkl, dt.pkl, nb.pkl
│   ├── locked_thresholds.pkl
│   └── locked_net_benefit_thresholds.pkl
├── figures/                # hình xuất từ notebook
└── notebooks/
    ├── 01_EDA_TelcoChurn.ipynb
    ├── 01b_Imbalance_Experiment.ipynb
    └── 02_Modeling_Models.ipynb
```

`app.py` tự tìm `data/` và `models/` cạnh nó hoặc ở thư mục cha.

## Cài đặt

Cần Python 3.12. Các file `.pkl` được lưu bằng scikit-learn 1.7.2 nên **phải đúng phiên bản** trong `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt môi trường ảo, chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi thử lại.

## Tái lập kết quả (nếu chưa có `models/`)

Chạy các notebook theo thứ tự, từ trên xuống dưới (Restart & Run All):

1. `01_EDA_TelcoChurn.ipynb` — làm sạch, tiền xử lý, chia train/test, lưu pipeline vào `data/`.
2. `01b_Imbalance_Experiment.ipynb` — thí nghiệm xử lý mất cân bằng lớp (không bắt buộc cho dashboard).
3. `02_Modeling_Models.ipynb` — tuning, cross-validation, ngưỡng, đánh giá test, lưu mô hình vào `models/`.

## Chạy dashboard

```powershell
python -m streamlit run app.py
```

Trình duyệt mở tại `http://localhost:8501`. (Dùng `python -m streamlit` để tránh lỗi PowerShell không nhận lệnh `streamlit`.)

| Tab                      | Nội dung                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Tổng quan**            | Tỉ lệ churn, nhận định chính, tỉ lệ churn theo hợp đồng / Internet / tenure / cước tháng, so sánh 3 mô hình trên tập test                  |
| **Dự đoán khách hàng**   | Nhập thông tin một khách hàng, nhận điểm rủi ro của 3 mô hình, khuyến nghị can thiệp, phân khúc Risk × Value, đường đi trong Decision Tree |
| **Quyết định can thiệp** | So sánh 3 kịch bản (không can thiệp / đại trà / theo mô hình) theo lợi ích ròng kỳ vọng, điểm hòa vốn                                      |

Mô hình nạp bằng `st.cache_resource`, dữ liệu và điểm test bằng `st.cache_data`, nên chỉ tính một lần cho mỗi phiên chạy.

## Cấu hình (`config.json`)

| Nhóm                        | Ý nghĩa                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `cost_sensitive_thresholds` | Ngưỡng tối thiểu hóa chi phí 5·FN + FP: KNN 0,20 / DT 0,25 / NB 0,89                                                 |
| `net_benefit_thresholds`    | Ngưỡng tối đa hóa Net Benefit: KNN 0,39 / DT 0,48 / NB 0,94. **Dashboard dùng bộ này**                               |
| `business_assumptions`      | `C` = chi phí liên hệ mỗi khách (10), `s` = tỉ lệ giữ chân thành công (0,30), `h` = giá trị giữ được mỗi khách (100) |
| `risk_value_segmentation`   | Cutoff rủi ro 0,659505 (phân vị 70% OOF của DT) và cutoff giá trị 70,50 (median MonthlyCharges)                      |
| `notebook_test_net_benefit` | Kết quả tham chiếu trên tập test: 2960 / 3090 / 2620                                                                 |

Công thức: `Net Benefit = TP × h × s − (TP + FP) × C`.

## Kiểm thử

```powershell
python -m pytest test_app.py -v
```

Bộ test kiểm tra: (1) `config.json` hợp lệ và khớp hằng số trong `app.py` và các file `.pkl`; (2) Net Benefit trên tập test tái lập đúng 2960 / 3090 / 2620; (3) dashboard có đủ 3 tab, dự đoán chạy được với khách mẫu, khách không dùng Internet/điện thoại và khách mới (tenure = 0), và tab quyết định chạy được với từng mô hình. Test cần `data/` và `models/` sẽ tự bỏ qua nếu thiếu file.

## Hạn chế cần lưu ý

- Chi phí, giá trị và tỉ lệ thành công (`C`, `h`, `s`) là giả định minh họa. Kéo thanh trượt trong tab 3 không tối ưu lại ngưỡng, vì ngưỡng đã khóa từ dữ liệu training.
- Điểm rủi ro chưa được hiệu chỉnh xác suất (calibration). Decision Tree sâu 3 chỉ cho 8 mức điểm, Naive Bayes cho điểm rất cực đoan. Dùng điểm để xếp hạng, không đọc như xác suất chính xác.
- Decision Tree bỏ sót toàn bộ khách churn có hợp đồng 1–2 năm trên tập test (45/45). Cần rà soát thủ công nhóm này.
- `MonthlyCharges` chỉ là proxy cho giá trị khách hàng, không phải CLV.
- CV sau tuning không phải nested CV nên chỉ số có thể hơi lạc quan.

## Xử lý sự cố

| Triệu chứng                           | Cách xử lý                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `streamlit` không được nhận diện      | Dùng `python -m streamlit run app.py`                                              |
| "Không tìm thấy thư mục dự án"        | Kiểm tra `models/dt.pkl` và `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` có tồn tại |
| Lỗi hoặc cảnh báo khi nạp `.pkl`      | `python -m pip show scikit-learn` phải là 1.7.2; cài lại theo `requirements.txt`   |
| Mục "Kiểm tra tái lập" ở tab 3 hiện ✗ | Sai phiên bản thư viện hoặc `models/` được tạo từ lần chạy khác                    |

## Nhóm thực hiện: Nhóm 2

1. Đặng Ngọc Kim Khánh - 52300208
2. Phan Thuý Vi - 52300084
3. Huỳnh Minh Thư - 52300068
4. Nguyễn Minh Tuấn - 52300079
