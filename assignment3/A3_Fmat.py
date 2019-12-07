import numpy as np
from compute_avg_reproj_error import *

'''
1-1-c : 이미지 크기 절반 빼기, /100
1-1-d : 나의 fundamental matrix 구하기. 란삭을 하던 뭘 하던 마음대로. 에러 최소화.
1-1-e : 출력
1-1-f : 3쌍 다 돌려보기
1-1-g : 팁.

1-2-a : rgb로 epipole line 다 그리기.아무 키 누르면 다른 샘플 보여주고 q누르면 꺼지게. 그림은 opencv.

2-1. 비슷한 이미지 찾기. 이미지 검색.
250개 다른 이미지 4개씩 총 1000장.
4개중 몇 개를 찾는지가 점수.
Sift features는 줌. 1 이미지당 128byte feature n개. binary파일.
D는 최대 1024.
경로는 상대경로
레포트
랜덤 사용시 랜덤시드

Sift/
Image/
eval_list.exe
compute_desc.exe
eval.exe

…

numpy만 사용.
Backword model 기본적으로 할 듯
TF-IDF : 덜 등장하는 것에 weight 줌.
Dictionary Learning: 
Learn Visual Words using clustering 
Encode: 
build Bags‐of‐Words (BOW) vectors for each image 
여기까지 2번과제 과정.
'''

M = np.loadtxt('temple_matches.txt')

# print(M)
A = []

def compute_F_raw(M):
    A = []
    for x1, y1, x2, y2 in M:
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    u, s, vh = np.linalg.svd(A)
    return np.reshape(vh[8], (3, 3))

def compute_F_norm(M):
    _M = M.copy()
    half_img = 250
    A = []
    for i in range(len(_M)):
        for j in range(4):
            _M[i][j] -= half_img
    for i in range(len(_M)):
        for j in range(4):
            _M[i][j] /= half_img

    for x1, y1, x2, y2 in _M:
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    u, s, vh = np.linalg.svd(A)
    F = np.reshape(vh[8], (3, 3))
    # F = np.dot(np.dot(np.linalg.inv(Td), F), Ts)

    # normalize 행렬
    sub_M_s = [[1, 0, -half_img],
               [0, 1, -half_img],
               [0, 0, 1]]
    scal_M_s = [[1/half_img, 0, 0],
                [0, 1/half_img, 0],
                [0, 0, 1]]

    T = np.dot(np.array(scal_M_s), np.array(sub_M_s))
    F = np.dot(np.dot(np.transpose(T),F),T)

    return F

def compute_F_mine(M):
    _M = M.copy()
    A = []
    # img 가로, 세로 1000으로 가정.(이후 M에서 최소, 최대값으로 바꾸기)
    # 1. M에서 img 가로,세로만큼 빼고
    # 2. 가로,세로만큼 나누기.
    s_x_max, s_y_max, d_x_max, d_y_max = np.max(_M,axis=0)

    for i in range(len(_M)):
        _M[i][0] -= s_x_max/2
        _M[i][1] -= s_y_max/2
        _M[i][2] -= d_x_max/2
        _M[i][3] -= d_y_max/2
    for i in range(len(M)):
        _M[i][0] /= s_x_max / 2
        _M[i][1] /= s_y_max / 2
        _M[i][2] /= d_x_max / 2
        _M[i][3] /= d_y_max / 2
    # M[0:len(M)][0] -= s_x_max/2
    # M[0:len(M)][1] -= s_y_max/2
    # M[0:len(M)][2] -= d_x_max/2
    # M[0:len(M)][3] -= d_y_max/2

    # normalize 행렬
    sub_M_s = [[1, 0, -s_x_max / 2],
               [0, 1, -s_y_max / 2],
               [0, 0, 1]]
    scal_M_s = [[2 / s_x_max, 0, 0],
                [0, 2 / s_y_max, 0],
                [0, 0, 1]]
    sub_M_d = [[1, 0, -d_x_max / 2],
               [0, 1, -d_y_max / 2],
               [0, 0, 1]]
    scal_M_d = [[2 / d_x_max, 0, 0],
                [0, 2 / d_y_max, 0],
                [0, 0, 1]]

    Ts = np.dot(np.array(scal_M_s), np.array(sub_M_s))
    Td = np.dot(np.array(scal_M_d), np.array(sub_M_d))

    for x1, y1, x2, y2 in _M:
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
    u, s, vh = np.linalg.svd(A)
    F = np.reshape(vh[8], (3, 3))
    # F = np.dot(np.dot(np.linalg.inv(Td), F), Ts)

    return F
F = compute_F_raw(M)
print(compute_avg_reproj_error(M,F))
# print(F)

F = compute_F_norm(M)
print(compute_avg_reproj_error(M,F))
# print(F)

F = compute_F_mine(M)
print(compute_avg_reproj_error(M,F))
# print(F)