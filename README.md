## ⚽ Dự đoán tỉ số bóng đá EPL bằng XGBoost (Streamlit)

Ứng dụng web dự đoán tỉ số (bàn thắng đội nhà/đội khách) cho các trận đấu EPL.

### Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Đảm bảo dữ liệu nằm ở vị trí: `./dataset/data.csv`.

### Huấn luyện mô hình (qua Notebook)

1. Mở notebook (nếu có file):

```bash
jupyter notebook eda_and_train.ipynb
```

2. Chạy lần lượt các cell để:
   - EDA sơ bộ dữ liệu
   - Tách mùa giải theo thời gian: Train (2013-07-01 → 2023-06-30), Val (2023-07-01 → 2024-06-30), Test (2024-07-01 → 2025-06-30)
   - Huấn luyện 2 mô hình XGBoost Regression (home/away)
   - Lưu mô hình vào `./models/home_model.pkl`, `./models/away_model.pkl`
   - Lưu metric vào `./models/metrics.json`

### Chạy ứng dụng web

```bash
streamlit run app.py
```

### Cách dùng trong UI

- Chọn “Đội nhà” và “Đội khách” (2 đội không được trùng nhau).
- Bấm “Dự đoán tỉ số” để xem dự đoán.
- Bên phải hiển thị metric tham khảo (Val 2023-2024, Test 2024-2025) nếu có `metrics.json`.
