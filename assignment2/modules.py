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
    if not os.path.exists('./result'):
        os.makedirs('./result')
    gray = cv2.imread(input_img_file_name, cv2.IMREAD_GRAYSCALE)
    gray = gray / 255  # 0~1 scaling
    gray = gray.astype('float32')
    return gray

def ij(x,y):
    # 400,400 -> 0,0
    # return 400+y, 400-x
    return 400 - y, 400 + x
def xy(i,j):
    return j-400, 400-i

def xy_cor(x,y):
    return x+400,-y+400

def paint(plane,img_at):
    for ele in img_at:
        i,j = ij(int(ele[0]), int(ele[1]))
        plane[i][j] = 0 #ele[2] # 검정으로 칠할지 색정보 넣을지?
    return plane


#
# def xy(i,j):
#     return j,-i

# 스마일 좌표 [x,y]들 저장.

def get_transformed_image(img,M):
    global result_xy
    plane = np.ones((801,801),dtype=np.float32)

    h,w = img.shape
    img_xy = []
    if M == M_IDENTITY:
        print("clear")
        # img 원점에 놓았을 때 좌표들 img_xy 리스트에 추가해주기.
        # (?) for문 안쓰고 속도 빠르게 가능한지?
        for i in range(0,h):
            for j in range(0,w):
                if img[i][j] != 1:
                    img_xy.append([j-int(w/2),int(h/2)-i,1])
        img_xy = np.array(img_xy)
        result_xy = np.copy(img_xy)

    print("h:{},w:{}".format(h,w))
    print(img_xy)
    print("img_xy shape : {} \n M shape : {}".format(np.shape(img_xy),np.shape(M)))

    img_xy = result_xy
    # M tranformation 하기
    # nx3 * 3x3 = nx3 matrix.
    result_xy = np.dot(img_xy,M)
    print("result : {}".format(result_xy))
    #result_xy = result_xy.astype(int) #정보손실 있음
    result_plane = paint(plane,result_xy)
    print("result_plane type : {}".format(np.shape(result_plane)))

    return result_plane