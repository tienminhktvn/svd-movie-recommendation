#!/usr/bin/env pythons

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from decomposition import custom_svd

# ===== Runtime options =====
QUICK_DEMO = False
QUICK_DEMO_MAX_USERS = 120
QUICK_DEMO_MAX_MOVIES = 300

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def test_custom_svd(input_matrix, tolerance=1e-5):
    """
    Validate pure-Python SVD implementation against np.linalg.svd.

    Args:
        input_matrix: 2D list or np.ndarray
        tolerance: threshold for pass/fail
    """
    a_np = np.array(input_matrix, dtype=float)
    if a_np.ndim != 2:
        raise ValueError("input_matrix must be 2D")

    u_custom, s_custom, vt_custom = custom_svd(a_np.tolist())
    u_custom = np.array(u_custom, dtype=float)
    s_custom = np.array(s_custom, dtype=float)
    vt_custom = np.array(vt_custom, dtype=float)

    u_np, s_np, vt_np = np.linalg.svd(a_np, full_matrices=False)

    a_custom_reconstructed = u_custom @ np.diag(s_custom) @ vt_custom
    a_np_reconstructed = u_np @ np.diag(s_np) @ vt_np

    singular_error = float(np.max(np.abs(s_custom - s_np)))
    recon_error_vs_input = float(np.max(np.abs(a_custom_reconstructed - a_np)))
    recon_error_vs_numpy = float(
        np.max(np.abs(a_custom_reconstructed - a_np_reconstructed))
    )

    passed = (
        singular_error < tolerance
        and recon_error_vs_input < tolerance
        and recon_error_vs_numpy < tolerance
    )

    print("\n[TEST CUSTOM SVD]")
    print(f"  - Max singular value error: {singular_error:.8f}")
    print(f"  - Max reconstruction error vs input: {recon_error_vs_input:.8f}")
    print(f"  - Max reconstruction error vs numpy: {recon_error_vs_numpy:.8f}")
    print(f"  - Result: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "singular_error": singular_error,
        "recon_error_vs_input": recon_error_vs_input,
        "recon_error_vs_numpy": recon_error_vs_numpy,
    }


def matrix_cache_key(matrix):
    """Build a stable cache key from matrix shape + raw bytes hash."""
    arr = np.ascontiguousarray(matrix.astype(np.float32))
    digest = hashlib.md5(arr.tobytes()).hexdigest()[:16]
    return f"{arr.shape[0]}x{arr.shape[1]}_{digest}"


def cache_paths(cache_key):
    return {
        "u": CACHE_DIR / f"U_{cache_key}.csv",
        "s": CACHE_DIR / f"S_{cache_key}.csv",
        "vt": CACHE_DIR / f"Vt_{cache_key}.csv",
    }


def load_svd_from_cache(paths):
    if not (paths["u"].exists() and paths["s"].exists() and paths["vt"].exists()):
        return None

    print("Tìm thấy cache SVD, đang load từ CSV...")
    u = np.loadtxt(paths["u"], delimiter=",", dtype=np.float32)
    s = np.loadtxt(paths["s"], delimiter=",", dtype=np.float32)
    vt = np.loadtxt(paths["vt"], delimiter=",", dtype=np.float32)

    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if np.ndim(s) == 0:
        s = np.array([s], dtype=np.float32)
    if vt.ndim == 1:
        vt = vt.reshape(1, -1)

    return u, s, vt


def save_svd_to_cache(u, s, vt, paths):
    print("Đang lưu cache SVD ra CSV...")
    np.savetxt(paths["u"], u, delimiter=",")
    np.savetxt(paths["s"], s, delimiter=",")
    np.savetxt(paths["vt"], vt, delimiter=",")


# Điểm số 5 phim dùng để "bắt mạch" sở thích (MovieID, Rating)
ANCHOR_MOVIES = [
    (1, 4.5),  # Toy Story
    (480, 4.2),  # Jurassic Park
    (72998, 4.5),  # Avatar (2009)
    (5349, 3.5),  # Spider-Man (2002)
    (79091, 3.8),  # Despicable Me (2010)
    (59784, 4.5),  # Kung Fu Panda (2008)
    (
        4896,
        3.0,
    ),  # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    # (1721, 2.0),  # Titanic (1997)
    (2081, 3.0),  # Little Mermaid, The (1989)
    (72921, 3.0),  # Snow White (1916)
    (588, 4.0),  # Aladdin (1992)
    (6377, 3.9),  # Finding Nemo (2003)
    (5618, 4.1),  # Spirited Away (Sen to Chihiro no kamikakushi) (2001)
    (114713, 1.5),  # Annabelle (2014)
    (50872, 4.0),  # Ratatouille (2007)
    (63992, 3.0),  # Twilight (2008)
    # (6539, 5.0),  # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    (595, 3.0),  # Beauty and the Beast (1991)
    (27611, 4.0),  # Howl's Moving Castle (Hauru no ugoku shiro) (2004)
]

# 5 phim dùng để kiểm chứng dự đoán (MovieID)
TEST_MOVIES = [59784, 1721]

print("Đang tải dữ liệu...")
data_ratings = pd.read_csv("data/ratings.csv")
data_movies = pd.read_csv("data/movies.csv")

if QUICK_DEMO:
    print(
        f"Quick demo mode ON: giữ tối đa {QUICK_DEMO_MAX_USERS} users, {QUICK_DEMO_MAX_MOVIES} movies"
    )
    top_users = data_ratings["userId"].value_counts().head(QUICK_DEMO_MAX_USERS).index
    ratings_by_users = data_ratings[data_ratings["userId"].isin(top_users)]
    top_movies = (
        ratings_by_users["movieId"].value_counts().head(QUICK_DEMO_MAX_MOVIES).index
    )
    data_ratings = ratings_by_users[ratings_by_users["movieId"].isin(top_movies)].copy()
    print(f"Kích thước ratings sau giảm mẫu: {data_ratings.shape}")

movies = data_ratings["movieId"].unique()
users = data_ratings["userId"].unique()

# Dictionary map ID thật của phim/user sang Index của ma trận (0, 1, 2...)
movie_to_idx = {res: idx for idx, res in enumerate(movies)}
idx_to_movie = {idx: res for idx, res in enumerate(movies)}
user_to_idx = {res: idx for idx, res in enumerate(users)}

# Khởi tạo ma trận Rating (Float32 để tính toán số thập phân)
ratings_mat = np.zeros(shape=(len(movies), len(users)), dtype=np.float32)

for row in data_ratings.itertuples():
    ratings_mat[movie_to_idx[row.movieId], user_to_idx[row.userId]] = row.rating

print("Đang xử lý Mean Centering...")
# Tính điểm trung bình của từng bộ phim (chỉ tính các ô có rating > 0)
sums = ratings_mat.sum(axis=1)
counts = (ratings_mat != 0).sum(axis=1)
movie_means = sums / (counts + 1e-9)

# Chuẩn hóa ma trận bằng cách trừ đi điểm trung bình (Numpy Broadcasting)
normalised_mat = ratings_mat - movie_means.reshape(-1, 1)
# Đưa các ô chưa xem (rating = 0 ban đầu) về lại 0
normalised_mat[ratings_mat == 0] = 0

print("Đang phân rã SVD...")
paths = cache_paths("demo")
cached = load_svd_from_cache(paths)

if cached is None:
    print("Không có cache phù hợp, chạy custom_svd...")
    # Áp dụng thuật toán SVD: A = U * S * V^T
    U, S, Vt = custom_svd(normalised_mat)
    U = np.array(U, dtype=np.float32)
    S = np.array(S, dtype=np.float32)
    Vt = np.array(Vt, dtype=np.float32)
    save_svd_to_cache(U, S, Vt, paths)
else:
    U, S, Vt = cached

# Trích xuất ma trận Đặc trưng phim (Latent Features) với k=50 chiều
k = min(len(ANCHOR_MOVIES) - 3, U.shape[1])
movie_features = U[:, :k]

# ==========================================
# 2. XÁC ĐỊNH VECTOR SỞ THÍCH CỦA NGƯỜI DÙNG
# ==========================================
print(f"\n[BƯỚC 1] Tìm Vector Sở Thích từ {len(ANCHOR_MOVIES)} phim đầu vào:")
A_matrix = []
b_vector = []

for m_id, rating in ANCHOR_MOVIES:
    if m_id not in movie_to_idx:
        print(f"  - Bỏ qua MovieID {m_id}: không nằm trong tập demo hiện tại")
        continue

    title_rows = data_movies[data_movies.movieId == m_id]
    title = title_rows.title.values[0] if not title_rows.empty else f"MovieID {m_id}"
    print(f"  - {rating} sao : {title}")

    idx_m = movie_to_idx[m_id]
    # Lấy 50 đặc trưng của phim này làm phương trình
    A_matrix.append(movie_features[idx_m, :])
    # Chuẩn hóa điểm số đầu vào để khớp với không gian ma trận
    b_vector.append(rating - movie_means[idx_m])

A_matrix = np.array(A_matrix)
b_vector = np.array(b_vector)

if len(A_matrix) == 0:
    raise RuntimeError("Không có anchor movie hợp lệ để suy ra sở thích.")

# Dùng Ma trận giả nghịch đảo để giải hệ phương trình: A * x = b
# Đầu ra là 'user_vector' chứa 50 giá trị đại diện cho sở thích
A_pinv = np.linalg.pinv(A_matrix)
user_vector = np.dot(A_pinv, b_vector)


# ==========================================
# 3. DỰ ĐOÁN ĐIỂM SỐ
# ==========================================
print(f"\n[BƯỚC 2] Dự đoán điểm cho {len(TEST_MOVIES)} phim kiểm chứng:")
for m_id in TEST_MOVIES:
    if m_id not in movie_to_idx:
        print(f"  -> Bỏ qua MovieID {m_id}: không nằm trong tập demo hiện tại")
        continue

    title_rows = data_movies[data_movies.movieId == m_id]
    title = title_rows.title.values[0] if not title_rows.empty else f"MovieID {m_id}"
    idx_m = movie_to_idx[m_id]

    # Tích vô hướng (Dot Product) giữa Đặc trưng phim và Vector sở thích
    pred_centered = np.dot(movie_features[idx_m, :], user_vector)

    # Cộng lại điểm trung bình của phim đó (De-normalize)
    # Hàm clip đảm bảo điểm số dự đoán không vượt quá giới hạn 1-5
    pred_rating = np.clip(pred_centered + movie_means[idx_m], 1.0, 5.0)

    print(f"  -> {pred_rating:.1f} sao : {title}")
