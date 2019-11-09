import cv2
import numpy as np
import time
import os
from math import *

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
    return 400+y, 400-x
def xy(i,j):
    return 400-j, i-400

def xy_cor(x,y):
    return x+400,-y+400

def paint(plane,img_at):
    for ele in img_at:
        i,j = ij(ele[0], ele[1])
        plane[i][j] = 0 #ele[2] # 검정으로 칠할지 색정보 넣을지?
    return plane


#
# def xy(i,j):
#     return j,-i

# 스마일 좌표 [x,y]들 저장.

def get_transformed_image(img,M):
    plane = np.ones((801,801),dtype=np.float32)

    # cv2.arrowedLine(plane,ij(400,0),ij(-400,0),0)
    # cv2.arrowedLine(plane,ij(0,400),ij(0,-400),0)

    h,w = img.shape
    img_xy = []

    # img 원점에 놓았을 때 좌표들 img_xy 리스트에 추가해주기.
    for i in range(0,h):
        for j in range(0,w):
            if img[i][j] != 1:
                img_xy.append([int(w/2)-j,i-int(h/2),1])
    print("h:{},w:{}".format(h,w))
    print(img_xy)
    print("img_xy shape : {} \n M shape : {}".format(np.shape(img_xy),np.shape(M)))

    # M tranformation 하기
    # nx3 * 3x3 = nx3 matrix.
    result_xy = np.dot(img_xy,M)
    print("result : {}".format(result_xy))
    result_xy = result_xy.astype(int)
    result_plane = paint(plane,result_xy)
    print("result_plane type : {}".format(np.shape(result_plane)))

    # smile_at = []
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if img[i][j] != 0:
    #             smile_at.append([i, j])
    return result_plane