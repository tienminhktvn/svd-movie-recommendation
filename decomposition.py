import math

from diagonalization import jacobi_eigen_decomposition


def transpose(a):
    if not a:
        return []
    return [list(col) for col in zip(*a)]


def matmul(a, b):
    if not a or not b:
        return []
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    if cols_a != rows_b:
        raise ValueError("Incompatible matrix shapes for multiplication")

    out = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols_b):
                out[i][j] += aik * b[k][j]
    return out


def matvec(a, x):
    rows = len(a)
    cols = len(a[0]) if a else 0
    if cols != len(x):
        raise ValueError("Incompatible shapes for matrix-vector product")
    out = [0.0 for _ in range(rows)]
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += a[i][j] * x[j]
        out[i] = s
    return out


def vector_norm(x):
    return math.sqrt(sum(v * v for v in x))


def get_column(a, col):
    return [a[i][col] for i in range(len(a))]


def sort_eigenpairs_desc(eigenvalues, eigenvectors):
    # Eigenvectors are stored as columns in eigenvectors.
    n = len(eigenvalues)
    pairs = []
    for i in range(n):
        pairs.append((eigenvalues[i], get_column(eigenvectors, i)))
    pairs.sort(key=lambda item: item[0], reverse=True)
    sorted_vals = [p[0] for p in pairs]

    # Convert back to matrix with vectors as columns.
    if n == 0:
        return sorted_vals, []
    m = len(pairs[0][1])
    sorted_vecs = [[0.0 for _ in range(n)] for _ in range(m)]
    for j in range(n):
        col = pairs[j][1]
        for i in range(m):
            sorted_vecs[i][j] = col[i]
    return sorted_vals, sorted_vecs


def custom_svd(a, tolerance=1e-10):
    """
    Pure Python SVD via eigen decomposition.

    Returns reduced SVD similar to np.linalg.svd(full_matrices=False):
      U: (m x r), S: (r,), Vt: (r x n)
    where r = min(m, n).
    """
    if not a:
        return [], [], []

    m = len(a)
    n = len(a[0])
    for row in a:
        if len(row) != n:
            raise ValueError("Input matrix has inconsistent row sizes")

    r = min(m, n)

    if m >= n:
        at = transpose(a)
        ata = matmul(at, a)  # n x n

        eigenvalues, v = jacobi_eigen_decomposition(ata, tolerance=tolerance)
        eigenvalues, v = sort_eigenpairs_desc(eigenvalues, v)

        s = [math.sqrt(ev) if ev > 0.0 else 0.0 for ev in eigenvalues]

        # Reduced shapes.
        s_r = s[:r]
        v_r = [[v[i][j] for j in range(r)] for i in range(n)]  # n x r
        vt_r = transpose(v_r)  # r x n

        u = [[0.0 for _ in range(r)] for _ in range(m)]
        for j in range(r):
            sigma = s_r[j]
            vj = get_column(v_r, j)
            av = matvec(a, vj)
            if sigma > tolerance:
                inv_sigma = 1.0 / sigma
                for i in range(m):
                    u[i][j] = av[i] * inv_sigma
            else:
                norm_av = vector_norm(av)
                if norm_av > tolerance:
                    inv_norm = 1.0 / norm_av
                    for i in range(m):
                        u[i][j] = av[i] * inv_norm

        return u, s_r, vt_r

    # m < n: compute U from A A^T, then V from A^T U / sigma.
    aat = matmul(a, transpose(a))  # m x m

    eigenvalues, u = jacobi_eigen_decomposition(aat, tolerance=tolerance)
    eigenvalues, u = sort_eigenpairs_desc(eigenvalues, u)

    s = [math.sqrt(ev) if ev > 0.0 else 0.0 for ev in eigenvalues]

    s_r = s[:r]
    u_r = [[u[i][j] for j in range(r)] for i in range(m)]  # m x r

    v_r = [[0.0 for _ in range(r)] for _ in range(n)]
    at = transpose(a)
    for j in range(r):
        sigma = s_r[j]
        uj = get_column(u_r, j)
        atu = matvec(at, uj)
        if sigma > tolerance:
            inv_sigma = 1.0 / sigma
            for i in range(n):
                v_r[i][j] = atu[i] * inv_sigma
        else:
            norm_atu = vector_norm(atu)
            if norm_atu > tolerance:
                inv_norm = 1.0 / norm_atu
                for i in range(n):
                    v_r[i][j] = atu[i] * inv_norm

    vt_r = transpose(v_r)
    return u_r, s_r, vt_r
