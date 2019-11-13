import cv2
import numpy as np
import time
import os
from math import *

M_IDENTITY = [[1,0,0],
              [0,1,0],
              [0,0,1]]

M_MOVE_MINUS_X = [[1,0,0],
                  [0,1,0],
                  [-5,0,1]]
M_MOVE_PLUS_X = [[1,0,0],
                 [0,1,0],
                 [5,0,1]]
M_MOVE_MINUS_Y = [[1,0,0],
                  [0,1,0],
                  [0,-5,1]]
M_MOVE_PLUS_Y = [[1,0,0],
                 [0,1,0],
                 [0,5,1]]

M_ROTATE_COUNTER_CLOCK = [[cos(5*pi/180), sin(5*pi/180), 0],
                          [-sin(5*pi/180), cos(5*pi/180), 0],
                          [0,0,1]]
M_ROTATE_CLOCK = [[cos(-5*pi/180), sin(-5*pi/180), 0],
                  [-sin(-5*pi/180), cos(-5*pi/180), 0],
                  [0,0,1]]

M_FLIP_X_AXIS = [[1,0,0],
                 [0,-1,0],
                 [0,0,1]]
M_FLIP_Y_AXIS = [[-1,0,0],
                 [0,1,0],
                 [0,0,1]]

M_SHRINK_X = [[0.95,0,0],
              [0,1,0],
              [0,0,1]]
M_ENLARGE_X = [[1.05, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]
M_SHRINK_Y = [[1, 0, 0],
              [0, 0.95, 0],
              [0, 0, 1]]
M_ENLARGE_Y = [[1, 0, 0],
               [0, 1.05, 0],
               [0, 0, 1]]

result_xy = []

# 이미지파일 열기.
def open_img(input_img_file_name):
    gray = cv2.imread(input_img_file_name, cv2.IMREAD_GRAYSCALE)
    gray = gray / 255  # 0~1 scaling
    gray = gray.astype('float32')
    return gray

def ij(x,y):
    return 400 - y, 400 + x
def xy(i,j):
    return j-400, 400-i

def paint(plane,img_at):
    for ele in img_at:
        i,j = ij(int(ele[0]), int(ele[1]))
        plane[i][j] = 0 #ele[2] # 검정으로 칠할지 색정보 넣을지?
    return plane

def get_transformed_image(img,M):
    global result_xy
    plane = np.ones((801,801),dtype=np.float32)

    h,w = img.shape
    img_xy = []
    if M == M_IDENTITY:
        # img 원점에 놓았을 때 좌표들 img_xy 리스트에 추가해주기.
        # (?) for문 안쓰고 속도 빠르게 가능한지?
        for i in range(0,h):
            for j in range(0,w):
                if img[i][j] != 1:
                    img_xy.append([j-int(w/2),int(h/2)-i,1])
        img_xy = np.array(img_xy)
        result_xy = np.copy(img_xy)

    # print("h:{},w:{}".format(h,w))
    # print(img_xy)
    # print("img_xy shape : {} \n M shape : {}".format(np.shape(img_xy),np.shape(M)))

    img_xy = result_xy
    # M tranformation 하기
    # nx3 * 3x3 = nx3 matrix.
    result_xy = np.dot(img_xy,M)
    result_plane = paint(plane,result_xy)

    return result_plane

# 2-2
def compute_homography(srcP, destP):
    '''

    :param srcP: N x 2 src의 매칭되는 feature points 좌표들
    :param destP: N x 2 dest의 매칭되는 feature points 좌표들
    :return: 변환하는 3 x 3 size H 행렬
    '''
    N = 26# len(srcP)

    # print("srcP:{},{}".format(srcP[400][0],srcP[400][1]))

    # mean subtraction
    sum_s_x, sum_s_y, sum_d_x, sum_d_y = 0,0,0,0

    for i in range(N):
        sum_s_x += srcP[i][0]
        sum_s_y += srcP[i][1]
        sum_d_x += destP[i][0]
        sum_d_y += destP[i][1]

    mean_s_x, mean_s_y = sum_s_x / N, sum_s_y / N
    mean_d_x, mean_d_y = sum_d_x / N, sum_d_y / N

    # print("mean_s:{},{}".format(mean_s_x,mean_s_y))
    # print("mean_d:{},{}".format(mean_d_x, mean_d_y))

    for i in range(N):
        srcP[i][0] -= mean_s_x
        srcP[i][1] -= mean_s_y
        destP[i][0] -= mean_d_x
        destP[i][1] -= mean_d_y

    # scaling
    longest_s = 0
    longest_d = 0
    for i in range(N):
        tmp = srcP[i][0]**2 + srcP[i][1]**2
        if longest_s**2 < tmp:
            longest_s = sqrt(tmp)
        tmp = destP[i][0]**2 + destP[i][1]**2
        if longest_d**2 < tmp:
            longest_d = sqrt(tmp)
        # if i<10:
        #     print("srcP_d : {}".format(sqrt(srcP[i][0]**2 + srcP[i][1]**2)))
        #     print("destP_d : {}".format(sqrt(destP[i][0] ** 2 + destP[i][1] ** 2)))
    # longest_s = np.max(np.abs(srcP))
    # longest_d = np.max(np.abs(destP))

    srcP = srcP * sqrt(2) / longest_s
    destP = destP * sqrt(2) / longest_d

    # print("l_s:{}\nl_d:{}".format(longest_s,longest_d))

    mean_sub_M_s = [[1, 0, -mean_s_x],
                    [0, 1, -mean_s_y],
                    [0, 0, 1]]
    scal_M_s = [[sqrt(2)/longest_s, 0, 0],
                [0, sqrt(2)/longest_s, 0],
                [0, 0, 1]]
    mean_sub_M_d = [[1, 0, -mean_d_x],
                    [0, 1, -mean_d_y],
                    [0, 0, 1]]
    scal_M_d = [[sqrt(2)/longest_d, 0, 0],
                [0, sqrt(2)/longest_d, 0],
                [0, 0, 1]]

    Ts = np.dot(np.array(scal_M_s),np.array(mean_sub_M_s))
    Td = np.dot(np.array(scal_M_d),np.array(mean_sub_M_d))

    print("Ts:{}\nTd:{}".format(Ts,Td))

    # computing Hn
    # A = 2N x 9 matrix

    for i in range(N):
        xs,ys,xd,yd = srcP[i][0],srcP[i][1],destP[i][0],destP[i][1]
        Ai = [-xs,-ys,-1,0,0,0,xs*xd,ys*xd,xd,0,0,0,-xs,-ys,-1,xs*yd,ys*yd,yd]
        Ai = np.reshape(Ai,(2,9))
        if i==0:
            A = Ai
        else:
            A = np.vstack([A,Ai])

    print("A:{}\nN:{}\n".format(np.shape(A),N))

    u,s,vh = np.linalg.svd(A)

    print("s:{}".format(s))
    print("vh:{}".format(vh))
    print("s_argmin:{}".format(np.argmin(s)))
    h = vh[np.argmin(s)]
    print("h_bef:{}".format(h))
    h = h/h[8]
    print("h_aft:{}".format(h))
    Hn = np.reshape(h,(3,3))

    print("Hn:{}".format(Hn))

    # 2-2-e
    H = np.dot(np.dot(np.linalg.inv(Td),Hn),Ts)
    H = H / H[2][2]

    # print("Ts:{}\nTd:{}".format(np.shape(Ts),np.shape(Td)))
    print("u:{}\ns:{}\nvh:{}\n".format(np.shape(u),np.shape(s),np.shape(vh)))

    print("H:{}".format(H))

    return H