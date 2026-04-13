import math


def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def copy_matrix(a):
    return [row[:] for row in a]


def jacobi_eigen_decomposition(a, tolerance=1e-10, max_iterations=10000):
    """
    Jacobi eigen decomposition for real symmetric matrix.
    Returns:
      eigenvalues: list[float]
      eigenvectors: matrix with eigenvectors as columns
    """
    n = len(a)
    if n == 0:
        return [], []

    for row in a:
        if len(row) != n:
            raise ValueError("Matrix must be square")

    # Validate symmetry quickly.
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > 1e-8:
                raise ValueError("Matrix must be symmetric")

    d = copy_matrix(a)
    v = identity_matrix(n)

    for _ in range(max_iterations):
        # Find largest off-diagonal element.
        p, q = 0, 1 if n > 1 else 0
        max_off_diag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(d[i][j])
                if value > max_off_diag:
                    max_off_diag = value
                    p, q = i, j

        if max_off_diag < tolerance:
            break

        app = d[p][p]
        aqq = d[q][q]
        apq = d[p][q]

        if abs(apq) < tolerance:
            continue

        tau = (aqq - app) / (2.0 * apq)
        t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
        if tau < 0:
            t = -t
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        # Rotate rows/columns p and q.
        for k in range(n):
            if k != p and k != q:
                dkp = d[k][p]
                dkq = d[k][q]
                d[k][p] = c * dkp - s * dkq
                d[p][k] = d[k][p]
                d[k][q] = s * dkp + c * dkq
                d[q][k] = d[k][q]

        dpp = c * c * app - 2.0 * s * c * apq + s * s * aqq
        dqq = s * s * app + 2.0 * s * c * apq + c * c * aqq
        d[p][p] = dpp
        d[q][q] = dqq
        d[p][q] = 0.0
        d[q][p] = 0.0

        # Update eigenvector matrix.
        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    eigenvalues = [d[i][i] for i in range(n)]
    return eigenvalues, v
