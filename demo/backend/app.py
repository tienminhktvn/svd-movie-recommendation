"""
SVD Movie Recommender Demo - Backend API
Flask server cung cap du lieu demo va thuc hien du doan rating bang SVD.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path

# Fix UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ──────────────────────────────────────────────
# Đường dẫn gốc của project (thư mục SVD)
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # …/SVD
DATA_DIR  = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

app = Flask(__name__)
CORS(app)   # Cho phép frontend (Vite dev server) gọi API

# ──────────────────────────────────────────────
# MOVIE IDs (tra từ movies.csv)
# ──────────────────────────────────────────────
MOVIE_CONFIG = [
    # anchor movies (5 phim mồi)
    {"movieId": 1,      "name": "Toy Story",                       "year": 1995, "role": "anchor", "emoji": "🤠"},
    {"movieId": 5349,   "name": "Spider-Man",                      "year": 2002, "role": "anchor", "emoji": "🕷️"},
    {"movieId": 79091,  "name": "Despicable Me",                   "year": 2010, "role": "anchor", "emoji": "🍌"},
    {"movieId": 4896,   "name": "Harry Potter",                    "year": 2001, "role": "anchor", "emoji": "⚡"},
    {"movieId": 6539,   "name": "Pirates of the Caribbean",        "year": 2003, "role": "anchor", "emoji": "🏴‍☠️"},
    # test movies (2 phim kiểm chứng)
    {"movieId": 6377,   "name": "Finding Nemo",                    "year": 2003, "role": "test",   "emoji": "🐠"},
    {"movieId": 5618,   "name": "Spirited Away",                   "year": 2001, "role": "test",   "emoji": "🐉"},
]

ANCHOR_IDS = [m["movieId"] for m in MOVIE_CONFIG if m["role"] == "anchor"]
TEST_IDS   = [m["movieId"] for m in MOVIE_CONFIG if m["role"] == "test"]
ALL_IDS    = ANCHOR_IDS + TEST_IDS

# ──────────────────────────────────────────────
# Load dữ liệu một lần khi khởi động server
# ──────────────────────────────────────────────
print("[INFO] Dang tai du lieu CSV...")
ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
movies_df  = pd.read_csv(DATA_DIR / "movies.csv")
print(f"[INFO] ratings shape: {ratings_df.shape}, movies shape: {movies_df.shape}")

# ──────────────────────────────────────────────
# Chọn 5 user đã đánh giá tất cả 7 bộ phim
# ──────────────────────────────────────────────
def pick_qualified_users(seed: int = 42) -> list[int]:
    sub = ratings_df[ratings_df["movieId"].isin(ALL_IDS)]
    counts = sub.groupby("userId")["movieId"].nunique()
    all7   = counts[counts == 7].index.tolist()

    if len(all7) >= 5:
        random.seed(seed)
        return sorted(random.sample(all7, 5))

    # Fallback: 5 anchor + ít nhất 1 test
    anchor_users = set(
        ratings_df[ratings_df["movieId"].isin(ANCHOR_IDS)]
        .groupby("userId")["movieId"].nunique()
        .pipe(lambda s: s[s == 5].index)
    )
    test_users = set(
        ratings_df[ratings_df["movieId"].isin(TEST_IDS)]
        .groupby("userId")["movieId"].nunique()
        .pipe(lambda s: s[s >= 1].index)
    )
    relaxed = sorted(anchor_users & test_users)
    random.seed(seed)
    return sorted(random.sample(relaxed, min(5, len(relaxed))))


SELECTED_USERS = pick_qualified_users()
print(f"[INFO] Nguoi dung duoc chon: {SELECTED_USERS}")

# ──────────────────────────────────────────────
# Load SVD cache (U, S, Vt) + xây dựng mapping
# ──────────────────────────────────────────────
print("[INFO] Dang load ma tran SVD tu cache...")

# Re-build ma trận rating giống SVDbasedRecommender.py
all_movies = ratings_df["movieId"].unique()
all_users  = ratings_df["userId"].unique()

movie_to_idx = {m: i for i, m in enumerate(all_movies)}
user_to_idx  = {u: i for i, u in enumerate(all_users)}

ratings_mat = np.zeros((len(all_movies), len(all_users)), dtype=np.float32)
for row in ratings_df.itertuples(index=False):
    ratings_mat[movie_to_idx[row.movieId], user_to_idx[row.userId]] = row.rating

# Mean centering (giống file gốc)
sums   = ratings_mat.sum(axis=1)
counts = (ratings_mat != 0).sum(axis=1)
movie_means = sums / (counts + 1e-9)

# Load pre-computed SVD (chỉ cần U và shape)
U  = np.loadtxt(CACHE_DIR / "U_demo.csv",  delimiter=",", dtype=np.float32)
S  = np.loadtxt(CACHE_DIR / "S_demo.csv",  delimiter=",", dtype=np.float32)
Vt = np.loadtxt(CACHE_DIR / "Vt_demo.csv", delimiter=",", dtype=np.float32)

if U.ndim == 1:  U  = U.reshape(-1, 1)
if S.ndim == 0:  S  = np.array([float(S)], dtype=np.float32)
if Vt.ndim == 1: Vt = Vt.reshape(1, -1)

# Dùng k chiều tiềm ẩn (giá trị nhỏ để ổn định)
K = min(20, U.shape[1])
movie_features = U[:, :K]   # shape (n_movies, K)

print(f"[INFO] SVD loaded - U:{U.shape}, S:{S.shape}, Vt:{Vt.shape} - k={K}")


# ──────────────────────────────────────────────
# Helper: dự đoán rating cho 1 user từ anchor
# ──────────────────────────────────────────────
def predict_for_user(user_anchor_ratings: dict[int, float]) -> dict[int, float]:
    """
    anchor_ratings: {movieId: rating}  (chỉ anchor movies)
    Trả về {movieId: predicted_rating} cho TEST_IDS
    """
    A_rows, b_vals = [], []
    for m_id, rating in user_anchor_ratings.items():
        if m_id not in movie_to_idx:
            continue
        idx_m = movie_to_idx[m_id]
        A_rows.append(movie_features[idx_m, :])
        b_vals.append(rating - float(movie_means[idx_m]))

    A_mat = np.array(A_rows, dtype=np.float64)
    b_vec = np.array(b_vals, dtype=np.float64)

    # Pseudoinverse để giải hệ overdetermined Ax = b
    A_pinv    = np.linalg.pinv(A_mat)
    user_vec  = A_pinv @ b_vec

    predictions = {}
    for m_id in TEST_IDS:
        if m_id not in movie_to_idx:
            continue
        idx_m = movie_to_idx[m_id]
        pred_centered = float(movie_features[idx_m, :].astype(np.float64) @ user_vec)
        pred_rating   = float(np.clip(pred_centered + movie_means[idx_m], 0.5, 5.0))
        predictions[m_id] = round(pred_rating, 2)

    return predictions


# ──────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────

@app.route("/api/users-initial", methods=["GET"])
def users_initial():
    """
    Trả về danh sách 5 users với ratings thực của họ:
    - anchor movies: có rating
    - test movies:   null
    """
    movie_meta = {m["movieId"]: m for m in MOVIE_CONFIG}
    result = []

    for uid in SELECTED_USERS:
        user_ratings = ratings_df[
            (ratings_df["userId"] == uid) &
            (ratings_df["movieId"].isin(ALL_IDS))
        ].set_index("movieId")["rating"].to_dict()

        movies_data = []
        for m in MOVIE_CONFIG:
            mid = m["movieId"]
            movies_data.append({
                "movieId":  mid,
                "name":     m["name"],
                "year":     m["year"],
                "role":     m["role"],
                "emoji":    m["emoji"],
                "rating":   float(user_ratings[mid]) if mid in user_ratings else None,
            })

        result.append({
            "userId":   uid,
            "label":    f"User {uid}",
            "movies":   movies_data,
        })

    return jsonify({
        "users":   result,
        "movies":  MOVIE_CONFIG,
        "anchors": ANCHOR_IDS,
        "tests":   TEST_IDS,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Dự đoán rating cho 2 test movies dựa trên anchor ratings.
    Trả về: predicted, actual, absolute_error cho từng user × test_movie.
    """
    movie_meta = {m["movieId"]: m for m in MOVIE_CONFIG}
    results = []

    for uid in SELECTED_USERS:
        # Lấy anchor ratings thực của user
        anchor_real = ratings_df[
            (ratings_df["userId"] == uid) &
            (ratings_df["movieId"].isin(ANCHOR_IDS))
        ].set_index("movieId")["rating"].to_dict()
        anchor_real = {int(k): float(v) for k, v in anchor_real.items()}

        # Lấy test ratings thực để so sánh
        test_real = ratings_df[
            (ratings_df["userId"] == uid) &
            (ratings_df["movieId"].isin(TEST_IDS))
        ].set_index("movieId")["rating"].to_dict()
        test_real = {int(k): float(v) for k, v in test_real.items()}

        # Dự đoán
        preds = predict_for_user(anchor_real)

        test_results = []
        for m_id in TEST_IDS:
            predicted = preds.get(m_id)
            actual    = test_real.get(m_id)
            error     = round(abs(predicted - actual), 2) if (predicted is not None and actual is not None) else None

            test_results.append({
                "movieId":       m_id,
                "name":          movie_meta[m_id]["name"],
                "emoji":         movie_meta[m_id]["emoji"],
                "predicted":     round(predicted, 2) if predicted is not None else None,
                "actual":        actual,
                "absoluteError": error,
            })

        results.append({
            "userId": uid,
            "label":  f"User {uid}",
            "anchor": [
                {"movieId": k, "rating": v, "name": movie_meta.get(k, {}).get("name", "")}
                for k, v in anchor_real.items()
            ],
            "testResults": test_results,
        })

    return jsonify({"results": results})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
