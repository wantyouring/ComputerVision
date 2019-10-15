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

# 1-1.gray 이미지 입력받아 (3*h,3*w) 크기의 패딩된 이미지 return.
def padding(gray,kernel_size):
    #start = time.time() # debug

    h, w = gray.shape  # h->i, w->j
    padding_img = np.zeros((h+2*kernel_size,w+2*kernel_size), dtype=np.float32)

    padding_img[0:kernel_size,0:kernel_size] = gray[0][0] # 좌상단 패딩
    padding_img[0:kernel_size,kernel_size:kernel_size+w] = gray[0,0:w].copy() # 중상단 패딩
    padding_img[0:kernel_size,kernel_size+w:kernel_size*2+w] = gray[0][w-1] # 우상단 패딩

    # 좌중간 패딩
    for i in range(kernel_size):
        padding_img[kernel_size:kernel_size+h,i] = gray[0:h,0].copy()
    # 우중간 패딩
    for i in range(kernel_size):
        padding_img[kernel_size:kernel_size+h,kernel_size+w+i] = gray[0:h,w-1].copy()

    padding_img[kernel_size:kernel_size+h,kernel_size:kernel_size+w] = gray[0:h, 0:w].copy() # 정중앙 이미지 그대로
    padding_img[kernel_size+h:kernel_size*2+h, 0:kernel_size] = gray[h-1][0] # 좌하단 패딩
    padding_img[kernel_size+h:kernel_size*2+h, kernel_size:kernel_size+w] = gray[h - 1, 0:w].copy()  # 중간하단 패딩
    padding_img[kernel_size+h:kernel_size*2+h, kernel_size+w:kernel_size*2+w] = gray[h - 1][w - 1]  # 우하단 패딩

    #cv2.imshow('padding', padding_img)
    #compute_time = time.time() - start  # debug
    #print('padding time : {}'.format(compute_time))
    return padding_img


# 1-1.1d cross correlation 함수로 img에 kernel적용한 결과 return.
def cross_correlation_1d(img, kernel):
    h, w = img.shape  # h->i, w->j
    result_img = np.zeros((h, w), dtype=np.float32)

    # kernel이 (1,n)인지 (n,1)인지 확인
    i, j = kernel.shape
    if i == 1:
        horizontal = True
        kernel_size = kernel.shape[1]
    else:
        horizontal = False
        kernel_size = kernel.shape[0]

    padding_img = padding(img, kernel_size)

    # (1,n)의 1d kernel인 경우
    if horizontal:
        for i in range(0, h):
            for j in range(0, w):
                cross_mat = kernel * padding_img[kernel_size + i:kernel_size + i + 1,
                                     kernel_size + j - (int)(kernel_size / 2):kernel_size + j + (int)(kernel_size / 2) + 1]
                result_img[i][j] = cross_mat.sum()

                '''
                sum = 0
                for k in range(0,kernel_size):

                    sum += kernel[0][kernel_size-k-1] * padding_img[h+i-(int)(kernel_size/2)][w+j-(int)(kernel_size/2)+k] # convolutional
                    # sum += kernel[0][k] * padding_img[h+i-(int)(kernel_size/2)][w+j-(int)(kernel_size/2)+k] # 그냥 correlation
                result_img[i][j] = sum
                '''
    # (n,1)의 1d kernel인 경우
    else:
        for i in range(0, h):
            for j in range(0, w):
                cross_mat = kernel * padding_img[kernel_size + i - (int)(kernel_size / 2):kernel_size + i + (int)(kernel_size / 2) + 1,
                                     kernel_size + j:kernel_size + j + 1]
                result_img[i][j] = cross_mat.sum()

                '''
                sum = 0
                for k in range(0,kernel_size):
                    sum += kernel[kernel_size-k-1][0] * padding_img[h+i-(int)(kernel_size/2)+k][w+j-(int)(kernel_size/2)] # convolutional
                    # sum += kernel[k][0] * padding_img[h+i-(int)(kernel_size/2)+k][w+j-(int)(kernel_size/2)] # 그냥 correlation
                result_img[i][j] = sum
                '''

    return result_img


# 1-1.2d cross correlation 함수로 img에 kernel적용한 결과 return.
def cross_correlation_2d(img, kernel):
    h, w = img.shape  # h->i, w->j
    result_img = np.zeros((h, w), dtype=np.float32)

    # kernel
    kernel_h, kernel_w = kernel.shape

    padding_img = padding(img,kernel_h)

    for i in range(0, h):
        for j in range(0, w):
            cross_mat = kernel * padding_img[kernel_h + i - (int)(kernel_h / 2):kernel_h + i + (int)(kernel_h / 2) + 1,
                                 kernel_h + j - (int)(kernel_w / 2):kernel_h + j + (int)(kernel_w / 2) + 1]
            result_img[i][j] = cross_mat.sum()

    '''

    # convolution 기존 코드---
    for i in range(0,h):
        for j in range(0,w):
            sum = 0
            for p in range(0,kernel_h):
                for q in range(0,kernel_w):
                    # sum += kernel[p][q] * padding_img[h+i-(int)(kernel_h/2)+p][w+j-(int)(kernel_w/2)+q] # 이건 그냥 correlation
                    sum += kernel[kernel_h - p - 1][kernel_w - q - 1] * padding_img[h + i - (int)(kernel_h / 2) + p][w + j - (int)(kernel_w / 2) + q] # cross-correlation. (convolution)
            result_img[i][j] = sum
    # ---
    '''
    return result_img


# 1-2.size, sigma 입력받아 1d 가우시안 filter kernel 출력.
def get_gaussian_filter_1d(size, sigma):
    gaussian_kernel_horizontal = np.zeros((1, size))
    gaussian_kernel_vertical = np.zeros((size, 1))

    # horizontal 1d gaussian kernel
    for i in range(0, size):
        _i = i - (int)(size / 2)
        gaussian_kernel_horizontal[0][i] = (1 / (sqrt(2 * pi) * sigma)) * exp((-_i ** 2) / (2 * sigma ** 2))

    # vertical 1d gaussian kernel
    for i in range(0, size):
        _i = i - (int)(size / 2)
        gaussian_kernel_vertical[i][0] = (1 / (sqrt(2 * pi) * sigma)) * exp((-_i ** 2) / (2 * sigma ** 2))

    # sum=1로 scaling
    sum = np.sum(gaussian_kernel_horizontal)  # hori,verti sum은 같음.
    gaussian_kernel_horizontal = gaussian_kernel_horizontal / sum
    gaussian_kernel_vertical = gaussian_kernel_vertical / sum

    # for i in range(0,size):
    #     gaussian_kernel_horizontal[0][i] = gaussian_kernel_horizontal[0][i]/sum
    # for i in range(0,size):
    #     gaussian_kernel_vertical[i][0] = gaussian_kernel_vertical[i][0]/sum

    return gaussian_kernel_horizontal, gaussian_kernel_vertical


# size, sigma 입력받아 2d 가우시안 filter kernel 출력.
def get_gaussian_filter_2d(size, sigma):
    gaussian_kernel = np.zeros((size, size))

    for i in range(0, size):
        for j in range(0, size):
            _i = i - (int)(size / 2)
            _j = j - (int)(size / 2)
            gaussian_kernel[i][j] = 1 / (2 * pi * sigma ** 2) * exp(-1 * (_i ** 2 + _j ** 2) / (2 * sigma ** 2))

    # sum=1로 scaling
    sum = np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel / sum

    return gaussian_kernel


# 9개의 다른 가우시안 filter 적용해 합친 이미지 결과 return 함수
def use_gaussian_filters(img):
    h, w = img.shape  # h->i, w->j
    imgs = np.zeros((3 * h, 3 * w), dtype=np.float32)

    kernel_sizes = [5, 11, 17]
    sigmas = [1, 6, 11]

    for kernel_size in range(0, 3):
        for sigma in range(0, 3):
            filter = get_gaussian_filter_2d(kernel_sizes[kernel_size], sigmas[sigma])
            filtered_img = cross_correlation_2d(img, filter)

            imgs[kernel_size * h:kernel_size * h + h, sigma * w:sigma * w + w] = filtered_img[0:h, 0:w].copy()

            # for i in range(0,h):
            #    for j in range(0,w):
            #        imgs[i+kernel_size * h][j+sigma * w] = filtered_img[i][j]

            cv2.putText(imgs, '{}x{} s={}'.format(kernel_sizes[kernel_size], kernel_sizes[kernel_size], sigmas[sigma]),
                        (10 + sigma * w, 10 + kernel_size * h), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                        (0, 0, 0))  # image caption. (x,y)로 입력받아 i,j 순서 주의.

    return imgs


# gaussian 2d kernel과 gaussian 1d kernel vertical, horizontal 적용 결과 비교 함수
def diff_gaussian_1d2d(img):
    # gaussian 2d
    kernel_2d = get_gaussian_filter_2d(5, 3)
    filtered_2d = cross_correlation_2d(img, kernel_2d)

    # gaussian 1d
    kernel_horizontal, kernel_vertical = get_gaussian_filter_1d(5, 3)
    filtered_1d_horizontal = cross_correlation_1d(img, kernel_horizontal)
    filtered_1d_complete = cross_correlation_1d(filtered_1d_horizontal, kernel_vertical)

    # diff with 2d and 1d
    diff_img = cv2.subtract(filtered_2d, filtered_1d_complete)
    diff_img = np.abs(diff_img)
    sum = np.sum(diff_img)

    return diff_img, sum


### Part 2. Edge Detection

# 2-1. input img에 gaussian filter 전처리. 결과 img return
def apply_gaussian(img):
    filter = get_gaussian_filter_2d(7, 1.5)
    return cross_correlation_2d(img, filter)


# 2-2. img gradient 계산하기.
def compute_image_gradient(img):
    h, w = img.shape
    direction = np.zeros((h, w), dtype=np.float32)
    magnitude = np.zeros((h, w), dtype=np.float32)

    # sobel filter x,y방향 적용하기 (conv버전)
    # sobel_x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # x direction sobel filter
    # sobel_y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # sobel not conv 버전
    sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # x direction sobel filter
    sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    df_dx = cross_correlation_2d(img, sobel_x_filter)
    df_dy = cross_correlation_2d(img, sobel_y_filter)

    for i in range(0, h):
        for j in range(0, w):
            magnitude[i][j] = sqrt(df_dy[i][j] ** 2 + df_dx[i][j] ** 2)
            direction[i][j] = atan2(df_dy[i][j], df_dx[i][j])

    return magnitude, direction


def non_maximum_suppression_dir(mag, dir):
    h, w = mag.shape
    dir = np.rad2deg(dir)
    dir = (dir+360) % 360
    suppresed_mag = mag.copy()

    for i in range(1,h - 1):
        for j in range(1, w - 1):
            val = dir[i][j]
            val2 = mag[i][j]

            if val < 22.5 or val > 337.5 or (val > 157.5 and val < 202.5):
                # 좌우
                if val2<=mag[i][j-1] or val2<=mag[i][j+1]:
                    suppresed_mag[i][j] = 0
            elif (val > 22.5 and val < 67.5) or (val > 202.5 and val < 247.5):
                # 우상
                # if val2 <= mag[i + 1][j - 1] or val2 <= mag[i - 1][j + 1]:
                #     suppresed_mag[i][j] = 0
                if val2 <= mag[i - 1][j-1] or val2 <= mag[i + 1][j+1]:
                    suppresed_mag[i][j] = 0
            elif (val > 67.5 and val < 112.5) or (val > 247.5 and val < 292.5):
                # 상하
                if val2 <= mag[i-1][j] or val2 <= mag[i+1][j]:
                    suppresed_mag[i][j] = 0
            elif (val > 112.5 and val < 157.5) or (val > 292.5 and val < 337.5):
                # 좌상
                # if val2 <= mag[i - 1][j-1] or val2 <= mag[i + 1][j+1]:
                #     suppresed_mag[i][j] = 0
                if val2 <= mag[i + 1][j - 1] or val2 <= mag[i - 1][j + 1]:
                    suppresed_mag[i][j] = 0

    return suppresed_mag

# 3번. haris corner detection하기.
def compute_corner_response(img):
    h, w = img.shape
    sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # x direction sobel filter
    sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 3-2. a)sobel filter 적용.
    df_dx = cross_correlation_2d(img, sobel_x_filter)
    df_dy = cross_correlation_2d(img, sobel_y_filter)

    a = df_dx * df_dx
    b = df_dx * df_dy
    c = df_dx * df_dy
    d = df_dy * df_dy

    window = np.ones((5,5))

    a = cross_correlation_2d(a,window)
    b = cross_correlation_2d(b,window)
    c = cross_correlation_2d(c, window)
    d = cross_correlation_2d(d, window)

    R = a*d-b*c - 0.04*(a+d)**2

    R[R<0] = 0

    scaling = R.max()
    R = R / scaling

    return R

def non_maximum_suppression_win(R,winSize):
    h,w = R.shape
    suppressed_R = np.zeros((h, w), dtype=np.float32)
    padded_suppressed_R = padding(R,winSize) # padding시 i,j 대신 h+i, w+j로 바꿔주기.

    for i in range(0,h):
        for j in range(0,w):
            # suppressed_R[i][j] = R[i][j] if R[i-(int)(winSize/2):i+(int)(winSize/2)+1,j-(int)(winSize/2):j+(int)(winSize/2)+1].max() > 0.1 else 0 # padding없을때
            local_max = np.max(padded_suppressed_R[winSize + i - (int)(winSize / 2):winSize + i + (int)(winSize / 2) + 1, winSize + j - (int)(winSize / 2):winSize + j + (int)(winSize / 2) + 1])
            suppressed_R[i][j] = R[i][j] if (local_max > 0.1 and local_max == R[i][j]) else 0 # a if test else b

    return suppressed_R


# a1_image_filtering 결과출력 함수
def a1_image_filtering_result(input_img_file_name,gray):
    ###출력코드###
    # 1-2.c
    filter_1d_hori, filter_1d_verti = get_gaussian_filter_1d(5, 1)
    print("gaussian_filter_1d_horizontal : \n{}".format(filter_1d_hori))
    print("gaussian_filter_1d_vertical : \n{}".format(filter_1d_verti))
    print("gaussian_filter_2d : \n{}".format(get_gaussian_filter_2d(5, 1)))
    # 1-2.d
    gaussian_imgs = use_gaussian_filters(gray)
    cv2.imwrite("./result/part_1_gaussian_filtered_{}".format(input_img_file_name), gaussian_imgs * 255)
    cv2.imshow("part_1_gaussian_filtered_{}".format(input_img_file_name), gaussian_imgs)
    # 1-2.e
    start = time.time()
    diff_img, sum = diff_gaussian_1d2d(gray)
    compute_time = time.time() - start
    cv2.imshow("diff_1d_2d", diff_img)
    print("gaussian 1d, 2d diff sum : {}".format(sum))
    print("computational times of 1D and 2D filterings : {}".format(compute_time))


# a1_edge_detection 결과출력 함수
def a1_edge_detection_result(input_img_file_name,gray):
    ###출력코드###
    # 2-1.
    gaus_applied_img = apply_gaussian(gray)
    # 2-2.d
    start = time.time()
    mag, dir = compute_image_gradient(gaus_applied_img)
    compute_time = time.time() - start
    print("computational times of compute_image_gradient : {}".format(compute_time))
    cv2.imwrite("./result/part_2_edge_raw_{}".format(input_img_file_name), mag * 255)
    cv2.imshow("part_2_edge_raw_{}".format(input_img_file_name), mag)

    # 2-3.
    start = time.time()
    suppresed_mag = non_maximum_suppression_dir(mag, dir)
    compute_time = time.time() - start
    print("computational times of non_maximum_suppression_dir : {}".format(compute_time))
    cv2.imwrite("./result/part_2_edge_sup_{}".format(input_img_file_name), suppresed_mag * 255)
    cv2.imshow("part_2_edge_sup_{}".format(input_img_file_name), suppresed_mag)

# a1_corner_detection 결과출력 함수
def a1_corner_detection_result(input_img_file_name, gray):
    h,w = gray.shape
    # 3-1.
    gaus_applied_img = apply_gaussian(gray)
    # 3-2.e
    start = time.time()
    R = compute_corner_response(gaus_applied_img)
    compute_time = time.time() - start
    print("computational times of compute_corner_response : {}".format(compute_time))
    cv2.imwrite("./result/part_3_corner_raw_{}".format(input_img_file_name), R * 255)
    cv2.imshow("part_3_corner_raw_{}".format(input_img_file_name), R)

    # 3-3.b
    rgb_img = cv2.cvtColor(np.uint8(gray * 255), cv2.COLOR_GRAY2RGB)  # rgb로 convert시 *255 scaling 필요.
    # rgb_img = gray
    for i in range(0, h):
        for j in range(0, w):
            if R[i][j] > 0.1:
                cv2.circle(rgb_img, (j, i), 1, (0, 255, 0), 1)

    cv2.imwrite("./result/part_3_corner_bin_{}".format(input_img_file_name), rgb_img)
    cv2.imshow("part_3_corner_bin_{}".format(input_img_file_name), rgb_img)

    # 3-3.d
    suppressed_R = non_maximum_suppression_win(R, 11)
    rgb_img = cv2.cvtColor(np.uint8(gray * 255), cv2.COLOR_GRAY2RGB)
    for i in range(0, h):
        for j in range(0, w):
            if suppressed_R[i][j] > 0.1:
                cv2.circle(rgb_img, (j, i), 3, (0, 255, 0), 1)

    cv2.imwrite("./result/part_3_corner_sup_{}".format(input_img_file_name), rgb_img)
    cv2.imshow("part_3_corner_sup_{}".format(input_img_file_name), rgb_img)