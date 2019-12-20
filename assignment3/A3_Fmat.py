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

files = ['temple','library','house']
exts = ['png','jpg','jpg']
Fs = []

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
    half_img_h = h1/2 # 그림 shape받아 절반 size 저장.
    half_img_w = w1/2

    A = []
    for i in range(len(_M)):
        _M[i][0] -= half_img_w
        _M[i][1] -= half_img_h
        _M[i][2] -= half_img_w
        _M[i][3] -= half_img_h
    for i in range(len(_M)):
        _M[i][0] /= half_img_w
        _M[i][1] /= half_img_h
        _M[i][2] /= half_img_w
        _M[i][3] /= half_img_h
    cnt = 0
    for x1, y1, x2, y2 in _M:
        cnt += 1
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        # if cnt>=8:
        #     break
    u, s, vh = np.linalg.svd(A)
    F = np.reshape(vh[8], (3, 3))

    # normalize 행렬 T 구하기.
    sub_M_s = [[1, 0, -half_img_w],
               [0, 1, -half_img_h],
               [0, 0, 1]]
    scal_M_s = [[1/half_img_w, 0, 0],
                [0, 1/half_img_h, 0],
                [0, 0, 1]]

    T = np.dot(np.array(scal_M_s), np.array(sub_M_s))
    # denormalize.
    F = np.dot(np.dot(np.transpose(T),F),T)

    return F

def compute_F_mine(M):
    global h1, w1
    _M = M.copy()
    half_img_h = h1 / 2  # 그림 shape받아 절반 size 저장.
    half_img_w = w1 / 2

    A = []
    for i in range(len(_M)):
        _M[i][0] -= half_img_w
        _M[i][1] -= half_img_h
        _M[i][2] -= half_img_w
        _M[i][3] -= half_img_h
    for i in range(len(_M)):
        _M[i][0] /= half_img_w
        _M[i][1] /= half_img_h
        _M[i][2] /= half_img_w
        _M[i][3] /= half_img_h

    min = 100  # ransan용
    np.random.seed(23) # 23
    for k in range(200):
        random_i = []
        # 서로다른 랜덤수 추출
        while len(random_i) <= 7:
            t = np.random.randint(np.shape(_M)[0])
            if t not in random_i:
                random_i.append(t)
        for ele in random_i:
            x1, y1, x2, y2 = _M[ele]
        # for x1, y1, x2, y2 in _M[0:50]:
            A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        u, s, vh = np.linalg.svd(A)
        F = np.reshape(vh[8], (3, 3))

        # normalize 행렬 T 구하기.
        sub_M_s = [[1, 0, -half_img_w],
                   [0, 1, -half_img_h],
                   [0, 0, 1]]
        scal_M_s = [[1 / half_img_w, 0, 0],
                    [0, 1 / half_img_h, 0],
                    [0, 0, 1]]

        T = np.dot(np.array(scal_M_s), np.array(sub_M_s))
        # denormalize.
        F = np.dot(np.dot(np.transpose(T), F), T)

        score = compute_avg_reproj_error(M, F)
        if min > score:
            min = score
            Fsave = F.copy()

    return Fsave

for i in range(3):
    file_name = files[i]
    ext = exts[i]

    M = np.loadtxt('{}_matches.txt'.format(file_name))

    img = cv2.imread('{}1.{}'.format(file_name,ext), cv2.IMREAD_COLOR)
    # temple1.astype(float)
    h1, w1, _ = np.shape(img)

    F = compute_F_raw(M)
    print("Raw = {}".format(compute_avg_reproj_error(M,F)))
    # print(F)

    F = compute_F_norm(M)
    print("Norm = {}".format(compute_avg_reproj_error(M,F)))
    # print(F)

    F = compute_F_mine(M)
    score = compute_avg_reproj_error(M, F)
    print("mine = {}".format(score))
    Fs.append(F)

    # F = cv2.findFundamentalMat(M[:, 0:2], M[:, 2:4])[0]  # 내장함수 테스트

# 1-2.
'''
for 3회
    (왼쪽 한 점 - 오른쪽 한 점)쌍 구하기
    왼쪽 한 점에 매칭되는 오른쪽의 epipolar line을 구해 표시.
    오른쪽 한 점에 매칭되는 왼쪽의 epipolar line을 구해 표시.'
'''

colors = [(255,0,0),(0,255,0),(0,0,255)]

for i in range(3):
    file_name = files[i]
    ext = exts[i]
    F = Fs[i]

    M = np.loadtxt('{}_matches.txt'.format(file_name))

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

            n2_np = np.dot(np.transpose(F),np.transpose(np.array([dx,dy,1])))
            a_ = n2_np[0]
            b_ = n2_np[1]
            c_ = n2_np[2]

            # a*x+by+c=0 -> x=0때 좌표, x=w일 때 좌표 line
            # y = -(a/b)x-(c/b)
            cv2.circle(img1,(int(sx),int(sy)),5,color,2)
            cv2.circle(img2,(int(dx),int(dy)),5,color,2)
            cv2.line(img2,(0,-int(c/b)),(w1,int(-(a/b)*w1-(c/b))),color,2)
            cv2.line(img1,(0,-int(c_/b_)),(w2,int(-(a_/b_)*w2-(c_/b_))),color,2)

        img_bind = np.hstack((img1,img2))
        cv2.imshow("3point",img_bind)
        pressed_key = cv2.waitKey(0)
        if pressed_key == ord('q'):
            cv2.destroyAllWindows()
            break