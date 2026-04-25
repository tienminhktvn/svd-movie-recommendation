# 🎬 SVD Movie Recommender – Interactive Demo

Demo tương tác minh họa thuật toán **Singular Value Decomposition (SVD)** ứng dụng trong hệ thống gợi ý phim.

## 📁 Cấu trúc thư mục

```
demo/
├── backend/          # Flask API (Python)
│   ├── app.py
│   └── requirements.txt
├── frontend/         # React + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── ExplainerSection.jsx
│   │   │   ├── RatingTable.jsx
│   │   │   ├── PredictButton.jsx
│   │   │   ├── ResultChart.jsx
│   │   │   └── Footer.jsx
│   │   ├── App.jsx
│   │   └── index.css
│   └── ...
└── README.md
```

## 🚀 Hướng dẫn chạy

> **Yêu cầu:** Python 3.10+, Node.js 18+, npm 9+

### Bước 1 – Chuẩn bị (chạy một lần)

**Backend** – Cài thư viện Python (dùng virtualenv có sẵn của project):

```powershell
# Từ thư mục gốc SVD/
.\venv\Scripts\pip install flask flask-cors
```

**Frontend** – Cài npm packages:

```powershell
cd demo\frontend
npm install
```

---

### Bước 2 – Khởi động Backend (Flask)

```powershell
# Từ thư mục gốc SVD/
.\venv\Scripts\python.exe demo\backend\app.py
```

Server chạy tại: **http://localhost:5000**

Kiểm tra: `http://localhost:5000/api/health` → `{"status": "ok"}`

---

### Bước 3 – Khởi động Frontend (Vite)

```powershell
cd demo\frontend
npm run dev
```

Mở trình duyệt tại: **http://localhost:5173**

---

## 🎯 Cách sử dụng Demo

1. Trang web hiển thị **bảng 5 User × 7 Phim**.
2. Cột xanh (5 phim đầu) = Ratings thực tế đã biết (**Anchor Movies**).
3. Cột tím (2 phim cuối) = Ô trống `???` (**Test Movies** — bị ẩn).
4. Nhấn nút **"🚀 Chạy Mô Phỏng Dự Đoán (SVD)"**.
5. Backend dùng **Pseudo-inverse** + **Dot Product** để dự đoán.
6. Bảng cập nhật với `Đoán | Thực tế | Sai số` mỗi ô.
7. Biểu đồ cột ghép hiện bên dưới để trực quan hoá **Error Gap**.

---

## ⚙️ Thuật toán SVD

```
A = U · S · Vᵀ   (Singular Value Decomposition)

Bước 1: A_k = U[:, :k]     # Ma trận đặc trưng phim (k chiều tiềm ẩn)
Bước 2: A_k · x = b        # Giải hệ với Pseudo-inverse
         x = pinv(A_k) · b  # Vector sở thích người dùng
Bước 3: rating = A_k[movie] · x + mean[movie]   # Dự đoán + de-normalize
```

**Dữ liệu:** MovieLens Latest Small (100k ratings, 610 users, 9,742 movies)
