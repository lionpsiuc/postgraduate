import numpy as np


def arnoldi(A, u, m, tol=1e-8):
    n = len(u)
    Q = np.zeros((n, m + 1), dtype=complex)
    H = np.zeros((m + 1, m), dtype=complex)
    Q[:, 0] = u / np.linalg.norm(u)

    for j in range(m):
        Q[:, j + 1] = A @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = np.vdot(Q[:, i], Q[:, j + 1])
            Q[:, j + 1] = Q[:, j + 1] - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(Q[:, j + 1])
        if abs(H[j + 1, j]) < tol:
            return Q[:, : j + 1], H[: j + 1, : j + 1]
        Q[:, j + 1] = Q[:, j + 1] / H[j + 1, j]
    return Q, H


A = np.array(
    [
        [3, 8, 7, 3, 3, 7, 2, 3, 4, 8],
        [5, 4, 1, 6, 9, 8, 3, 7, 1, 9],
        [3, 6, 9, 4, 8, 6, 5, 6, 6, 6],
        [5, 3, 4, 7, 4, 9, 2, 3, 5, 1],
        [4, 4, 2, 1, 7, 4, 2, 2, 4, 5],
        [4, 2, 8, 6, 6, 5, 2, 1, 1, 2],
        [2, 8, 9, 5, 2, 9, 4, 7, 3, 3],
        [9, 3, 2, 2, 7, 3, 4, 8, 7, 7],
        [9, 1, 9, 3, 3, 1, 2, 7, 7, 1],
        [9, 3, 2, 2, 6, 4, 4, 7, 3, 5],
    ]
)

u = np.array(
    [
        0.757516242460009,
        2.734057963614329,
        -0.555605907443403,
        1.144284746786790,
        0.645280108318073,
        -0.085488474462339,
        -0.623679022063185,
        -0.465240896342741,
        2.382909057772335,
        -0.120465395885881,
    ]
)

Q, H = arnoldi(A, u, 9)

orthonormal_check = np.abs(Q.conjugate().T @ Q - np.eye(Q.shape[1]))
print("Max deviation from orthonormality:", np.max(orthonormal_check))

arnoldi_relation = np.linalg.norm(A @ Q[:, :-1] - Q @ H)
print("Arnoldi relation error:", arnoldi_relation)
