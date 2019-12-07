import numpy as np

M = np.loadtxt('temple_matches.txt')

# print(M)
A = []

def compute_F_raw(M):
    A = []
    for x1, y1, x2, y2 in M:
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    u, s, vh = np.linalg.svd(A)
    return np.reshape(vh[8], (3, 3))

F = compute_F_raw(M)
print(F)