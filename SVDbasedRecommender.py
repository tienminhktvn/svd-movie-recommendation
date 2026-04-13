#!/usr/bin/env pythons

import numpy as np
import pandas as pd

from decomposition import custom_svd


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


# Điểm số 5 phim dùng để "bắt mạch" sở thích (MovieID, Rating)
ANCHOR_MOVIES = [
    (1, 5.0),  # Toy Story
    (2571, 4.0),  # The Matrix
    (318, 5.0),  # Shawshank Redemption
    (480, 5.0),  # Jurassic Park
    (593, 5.0),  # Silence of the Lambs
]

# 5 phim dùng để kiểm chứng dự đoán (MovieID)
TEST_MOVIES = [
    260,  # Star Wars: Episode IV
    589,  # Terminator 2
    356,  # Forrest Gump
    2858,  # American Beauty
    1196,  # Star Wars: Episode V
]

print("Đang tải dữ liệu...")
data_ratings = pd.read_csv("data/ratings.csv")
data_movies = pd.read_csv("data/movies.csv")

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
# Áp dụng thuật toán SVD: A = U * S * V^T
U, S, V = np.linalg.svd(normalised_mat, full_matrices=False)

# Demo validate thuật toán custom trên ma trận nhỏ (để chạy nhanh)
_ = test_custom_svd([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]])

# Trích xuất ma trận Đặc trưng phim (Latent Features) với k=50 chiều
k = 50
movie_features = U[:, :k]

# ==========================================
# 2. XÁC ĐỊNH VECTOR SỞ THÍCH CỦA NGƯỜI DÙNG
# ==========================================
print(f"\n[BƯỚC 1] Tìm Vector Sở Thích từ {len(ANCHOR_MOVIES)} phim đầu vào:")
A_matrix = []
b_vector = []

for m_id, rating in ANCHOR_MOVIES:
    title = data_movies[data_movies.movieId == m_id].title.values[0]
    print(f"  - {rating} sao : {title}")

    idx_m = movie_to_idx[m_id]
    # Lấy 50 đặc trưng của phim này làm phương trình
    A_matrix.append(movie_features[idx_m, :])
    # Chuẩn hóa điểm số đầu vào để khớp với không gian ma trận
    b_vector.append(rating - movie_means[idx_m])

A_matrix = np.array(A_matrix)
b_vector = np.array(b_vector)

# Dùng Ma trận giả nghịch đảo để giải hệ phương trình: A * x = b
# Đầu ra là 'user_vector' chứa 50 giá trị đại diện cho sở thích
A_pinv = np.linalg.pinv(A_matrix)
user_vector = np.dot(A_pinv, b_vector)


# ==========================================
# 3. DỰ ĐOÁN ĐIỂM SỐ
# ==========================================
print(f"\n[BƯỚC 2] Dự đoán điểm cho {len(TEST_MOVIES)} phim kiểm chứng:")
for m_id in TEST_MOVIES:
    title = data_movies[data_movies.movieId == m_id].title.values[0]
    idx_m = movie_to_idx[m_id]

    # Tích vô hướng (Dot Product) giữa Đặc trưng phim và Vector sở thích
    pred_centered = np.dot(movie_features[idx_m, :], user_vector)

    # Cộng lại điểm trung bình của phim đó (De-normalize)
    # Hàm clip đảm bảo điểm số dự đoán không vượt quá giới hạn 1-5
    pred_rating = np.clip(pred_centered + movie_means[idx_m], 1.0, 5.0)

    print(f"  -> {pred_rating:.1f} sao : {title}")
