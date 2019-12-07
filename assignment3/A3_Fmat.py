import numpy as np
import cv2
from compute_avg_reproj_error import *

'''
할 것 체크
@@@@그림 3개에 대해서 모두 실행하게끔 바꾸기.
epipolar line 맞는지 다시 확인하기. line이 점을 안지나는 경우 발생함.
compute_F_mine()함수 작성하기. 방법 체크.
'''

'''
1-1-c : 이미지 크기 절반 빼기, /100
1-1-d : 나의 fundamental matrix 구하기. 란삭을 하던 뭘 하던 마음대로. 에러 최소화.
1-1-e : 출력
1-1-f : 3쌍 다 돌려보기
1-1-g : 팁.

1-2-a : rgb로 epipole line 다 그리기.아무 키 누르면 다른 샘플 보여주고 q누르면 꺼지게. 그림은 opencv.
'''

file_name = 'library'
ext = 'jpg'

M = np.loadtxt('{}_matches.txt'.format(file_name))

# print(M)
global h1,w1,h2,w2


def compute_F_raw(M):
    A = []
    cnt = 0
    for x1, y1, x2, y2 in M:
        cnt += 1
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        # 8 point 8개까지만 해야하는지?
        # if cnt>=8:
        #     break
    u, s, vh = np.linalg.svd(A)
    return np.reshape(vh[8], (3, 3))

# ppt 8chapter 72p 참고.
def compute_F_norm(M):
    global h1,w1
    _M = M.copy()
    half_img = max([h1,w1])/2 # 그림 shape받아 절반 size 저장.
    A = []
    for i in range(len(_M)):
        for j in range(4):
            _M[i][j] -= half_img
    for i in range(len(_M)):
        for j in range(4):
            _M[i][j] /= half_img
    cnt = 0
    for x1, y1, x2, y2 in _M:
        cnt += 1
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        # if cnt>=8:
        #     break
    u, s, vh = np.linalg.svd(A)
    F = np.reshape(vh[8], (3, 3))

    # normalize 행렬 T 구하기.
    sub_M_s = [[1, 0, -half_img],
               [0, 1, -half_img],
               [0, 0, 1]]
    scal_M_s = [[1/half_img, 0, 0],
                [0, 1/half_img, 0],
                [0, 0, 1]]

    T = np.dot(np.array(scal_M_s), np.array(sub_M_s))
    # denormalize.
    F = np.dot(np.dot(np.transpose(T),F),T)

    return F

def compute_F_mine(M):
    # normalize를 다른 방식으로 해보자.
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

img = cv2.imread('{}1.{}'.format(file_name,ext), cv2.IMREAD_COLOR)
# temple1.astype(float)
h1, w1, _ = np.shape(img)

F = compute_F_raw(M)
print(compute_avg_reproj_error(M,F))
# print(F)

F = compute_F_norm(M)
print(compute_avg_reproj_error(M,F))
# print(F)

# F = compute_F_mine(M)
# print(compute_avg_reproj_error(M,F))
# print(F)

# 1-2.
'''
for 3회
    (왼쪽 한 점 - 오른쪽 한 점)쌍 구하기
    왼쪽 한 점에 매칭되는 오른쪽의 epipolar line을 구해 표시.
    오른쪽 한 점에 매칭되는 왼쪽의 epipolar line을 구해 표시.'
'''

colors = [(255,0,0),(0,255,0),(0,0,255)]

while True:
    point3 = np.random.choice(len(M),3,replace=False)
    # print(point3)

    img1 = cv2.imread('{}1.{}'.format(file_name,ext), cv2.IMREAD_COLOR)
    img2 = cv2.imread('{}2.{}'.format(file_name,ext), cv2.IMREAD_COLOR)
    # temple1.astype(float)
    h1, w1, _ = np.shape(img1)
    h2, w2, _ = np.shape(img2)
    # print([h1,w1,h2,w2])

    for point, color in zip(point3,colors):
        # np.random.randint()
        sx = M[point][0]
        sy = M[point][1]
        dx = M[point][2]
        dy = M[point][3]

        n1_np = np.dot(F,np.transpose(np.array([sx,sy,1])))
        a = n1_np[0]
        b = n1_np[1]
        c = n1_np[2]

        n2_np = np.dot(np.array([dx,dy,1]),F)
        a_ = n2_np[0]
        b_ = n2_np[1]
        c_ = n2_np[2]

        # a*x+by+c=0 -> x=0때 좌표, x=w일 때 좌표 line
        # y = -(a/b)x-(c/b)
        cv2.circle(img1,(int(sx),int(sy)),5,color,2)
        cv2.circle(img2,(int(dx),int(dy)),5,color,2)
        cv2.line(img1,(0,-int(c/b)),(w1,int(-(a/b)*w1-(c/b))),color,2)
        cv2.line(img2,(0,-int(c_/b_)),(w2,int(-(a_/b_)*w2-(c_/b_))),color,2)

    img_bind = np.hstack((img1,img2))
    cv2.imshow("3point",img_bind)
    pressed_key = cv2.waitKey(0)
    if pressed_key == ord('q'):
        cv2.destroyAllWindows()
        break