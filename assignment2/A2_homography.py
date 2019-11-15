
from modules import *

### solve 초안
# top 10 feature matching을 어떻게 할 것인가???. matcher 구현하기 문제.
# orb similarity는 hamming distance로 계산.
# 이미지의 key point와 descriptor.
# A의 i번째 descriptor - B의 모든 descriptor와 비교해 가장 일치하는(hamming distance 짧은) j번째 descriptor 선정.
# match 배열에 1. A descriptor들과 2. 짝지은 B descriptor 3. hamming distance 튜플로 저장하자.
# ㄴ 매칭한 hamming distance 저장하는건 제일 잘 매칭되는 top10 뽑기 위해서.
# top 10 match point 출력하기.

### 최종 solve
# A의 i번째 descriptor와 가장 일치하는 B의 descriptor 찾아 DMatch obj로 matches에 저장.

def main():
    gray_desk = cv2.imread('cv_desk.PNG', cv2.IMREAD_GRAYSCALE)
    gray_cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

    ############################# 2-1 #############################
    orb = cv2.ORB_create()
    # cover 그림
    kp2 = orb.detect(gray_cover, None)
    kp2, des2 = orb.compute(gray_cover, kp2)
    # print("kp : {} \n des : {} \n".format(np.shape(kp1),np.shape(des1)))

    # desk 그림
    kp1 = orb.detect(gray_desk, None)
    kp1, des1 = orb.compute(gray_desk, kp1)
    # print("kp : {}\n des : {} \n".format(kp2[0].pt, np.shape(des2)))

    matches = []

    for i in range(len(kp1)):
        min_ham_dis = 100000000
        for j in range(len(kp2)):
            ham_dis = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)

            if min_ham_dis > ham_dis:
                min_ham_dis = ham_dis
                min_index = j

        # save shortest hamming distance property
        match_obj = cv2.DMatch(_queryIdx=i, _trainIdx=min_index, _distance=min_ham_dis)
        matches.append(match_obj)

    matches = sorted(matches, key=lambda obj: obj.distance)  # distance 기준 정렬

    res = None
    res = cv2.drawMatches(gray_desk, kp1, gray_cover, kp2, matches[:10], res,
                          flags=2)  # flags=2 option single points 없애줌.
    cv2.namedWindow("matching")
    cv2.moveWindow("matching", 0, 0)
    cv2.imshow('matching', res)

    ############################# 2-2 #############################

    # cover 그림
    kp1 = orb.detect(gray_cover, None)
    kp1, des1 = orb.compute(gray_cover, kp1)
    # print("kp : {} \n des : {} \n".format(np.shape(kp1),np.shape(des1)))

    # desk 그림
    kp2 = orb.detect(gray_desk, None)
    kp2, des2 = orb.compute(gray_desk, kp2)
    # print("kp : {}\n des : {} \n".format(kp2[0].pt, np.shape(des2)))

    matches = []

    for i in range(len(kp1)):
        min_ham_dis = 100000000
        for j in range(len(kp2)):
            ham_dis = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)

            if min_ham_dis > ham_dis:
                min_ham_dis = ham_dis
                min_index = j

        # save shortest hamming distance property
        match_obj = cv2.DMatch(_queryIdx=i,_trainIdx=min_index,_distance=min_ham_dis)
        matches.append(match_obj)

    ############################ Bf matcher test
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    # matches = bf.match(des1, des2)  # kp1[i] -> kp2[matches[i]]
    ############################

    matches = sorted(matches,key=lambda obj:obj.distance) # distance 기준 정렬

    # srcP에 distance 낮은 순서로 정렬된 상태로 넣기
    srcP = []
    destP = []

    print("matches size : {}".format(len(matches)))

    for i in range(500):
        srcP.append([kp1[matches[i].queryIdx].pt[0],kp1[matches[i].queryIdx].pt[1]])
        destP.append([kp2[matches[i].trainIdx].pt[0],kp2[matches[i].trainIdx].pt[1]])
    srcP = np.array(srcP)  # shape of srcP : (N,2) (N = 500)
    destP = np.array(destP)

    for i in range(30):
        print("matches dis:{}".format(matches[i].distance))

    # print("srcP shape:{}".format(np.shape(srcP)))

    # print("matches shape : {}, matches : {}".format(np.shape(matches),matches))
    # print("match 1 element : distance{}\n trainidx:{}\n queryidx:{}\n imgidx:{}".
    #       format(matches[2].distance,matches[2].trainIdx,matches[2].queryIdx,matches[2].imgIdx))

    H = compute_homography(srcP,destP)
    h_desk,w_desk = np.shape(gray_desk)
    gray_desk_homo = np.copy(gray_desk)
    dst = cv2.warpPerspective(gray_cover,H,(w_desk,h_desk))
    cv2.imshow("homography only", dst)
    for i in range(h_desk):
        for j in range(w_desk):
            if dst[i][j] != 0:
                gray_desk_homo[i][j] = dst[i][j]

    ############################# 2-3, 2-4 #############################
    srcP = np.copy(srcP[20:]) # huristic
    destP = np.copy(destP[20:])
# [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24] 6.94303890995347
# [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 24] 7.706421086794899
# [0, 1, 2, 3, 5, 6, 7, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24] 7.3504919606982

    # max_dis: 7.510316821935321
    # max_in_cnt: 19
    # max_inliers: [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24]

    #ransac threshold 찾기 위한 코드
    # for th in range(1,40):
    #     H_ransac = compute_homography_ransac(srcP,destP,7.617144125) #
    #     gray_desk_ransac = np.copy(gray_desk)
    #     dst2 = cv2.warpPerspective(gray_cover, H_ransac, (w_desk, h_desk))
    #     for i in range(h_desk):
    #         for j in range(w_desk):
    #             if dst2[i][j] != 0:
    #                 gray_desk_ransac[i][j] = dst2[i][j]
    #     cv2.imwrite('{}.png'.format(th), gray_desk_ransac)

    # 잘나오는 inliers 조합
    # N = 70
    # max_in_cnt: 31
    # max_inliers: [17, 20, 21, 22, 25, 26, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 44, 46, 47, 50, 53, 54, 57, 58,
    #               59, 60, 61, 65, 66, 68]

    # N = 70
    # max_in_cnt: 37
    # max_inliers: [17, 20, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 44, 46, 47, 50,
    #               53, 54, 55, 57, 58, 59, 60, 61, 62, 65, 66, 68]

    #N = 70
    # max_in_cnt: 34
    # max_inliers: [17, 20, 21, 22, 25, 26, 27, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 44, 46, 47, 50, 53, 54,
    #               55, 57, 58, 59, 60, 61, 65, 66, 68]

    H_ransac = compute_homography_ransac(srcP, destP, 7.617144125)  # 26 7373 7378 7381 7389 7.617144125
    gray_desk_ransac = np.copy(gray_desk)
    dst2 = cv2.warpPerspective(gray_cover, H_ransac, (w_desk, h_desk))
    cv2.imshow("ransac only",dst2)
    for i in range(h_desk):
        for j in range(w_desk):
            if dst2[i][j] != 0:
                gray_desk_ransac[i][j] = dst2[i][j]


    # 해리포터 cover
    hp_cover = cv2.imread('hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    h,w = np.shape(gray_cover)
    hp_cover2 = cv2.resize(hp_cover,(w,h))
    hp_desk_ransac = np.copy(gray_desk)
    hp_cover_desk = cv2.warpPerspective(hp_cover2, H_ransac, (w_desk, h_desk))
    cv2.imshow("hp only",hp_cover_desk)
    for i in range(h_desk):
        for j in range(w_desk):
            if hp_cover_desk[i][j] != 0:
                hp_desk_ransac[i][j] = hp_cover_desk[i][j]


    ############################# 2-5 #############################
    img1 = cv2.imread('diamondhead-10.PNG', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('diamondhead-11.PNG', cv2.IMREAD_GRAYSCALE)
    # cover 그림
    kp1 = orb.detect(img2, None)
    kp1, des1 = orb.compute(img2, kp1)
    # print("kp : {} \n des : {} \n".format(np.shape(kp1),np.shape(des1)))

    # desk 그림
    kp2 = orb.detect(img1, None)
    kp2, des2 = orb.compute(img1, kp2)
    # print("kp : {}\n des : {} \n".format(kp2[0].pt, np.shape(des2)))

    matches = []

    for i in range(len(kp1)):
        min_ham_dis = 100000000
        for j in range(len(kp2)):
            ham_dis = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)

            if min_ham_dis > ham_dis:
                min_ham_dis = ham_dis
                min_index = j

        # save shortest hamming distance property
        match_obj = cv2.DMatch(_queryIdx=i, _trainIdx=min_index, _distance=min_ham_dis)
        matches.append(match_obj)

    matches = sorted(matches, key=lambda obj: obj.distance)  # distance 기준 정렬

    # srcP에 distance 낮은 순서로 정렬된 상태로 넣기
    srcP = []
    destP = []

    for i in range(500):
        srcP.append([kp1[matches[i].queryIdx].pt[0], kp1[matches[i].queryIdx].pt[1]])
        destP.append([kp2[matches[i].trainIdx].pt[0], kp2[matches[i].trainIdx].pt[1]])
    srcP = np.array(srcP)  # shape of srcP : (N,2) (N = 500)
    destP = np.array(destP)

    H = compute_homography_ransac(srcP, destP,25) # 33
    #H = compute_homography(srcP, destP) # homography test. 잘 나옴.
    h_desk, w_desk = np.shape(img1)
    # 이미지 블렌딩 없이 합치기
    dst = cv2.warpPerspective(img2, H, (w_desk+400, h_desk)) # 출력할 이미지
    dst[0:h_desk,0:w_desk] = np.copy(img1)
    cv2.imshow("no blending", dst)

    # blending1 : 왼쪽 이미지, blending2 : 오른쪽 이미지
    blending2 = cv2.warpPerspective(img2, H, (w_desk + 400, h_desk))
    blending1 = np.zeros((h_desk,w_desk + 400),dtype=float)
    blending1[0:h_desk,0:w_desk] = np.copy(img1)

    for i in range(0,h_desk):
        alpha = 1
        for j in range(1024-100,1024):
            alpha -= 0.01
            dst[i][j] = alpha * blending1[i][j] + (1-alpha) * blending2[i][j]

    ############################# print #############################
    print("end")
    cv2.namedWindow("hp")  # Create a named window
    cv2.moveWindow("hp", 100, 10)
    cv2.imshow("hp", hp_desk_ransac)
    cv2.namedWindow("homography")  # Create a named window
    cv2.moveWindow("homography", 150, 30)
    cv2.imshow("homography",gray_desk_homo)
    cv2.namedWindow("ransac")  # Create a named window
    cv2.moveWindow("ransac", 700, 30)
    cv2.imshow("ransac",gray_desk_ransac)
    cv2.namedWindow("stitch")  # Create a named window
    cv2.moveWindow("stitch", 60, 70)
    cv2.imshow("stitch", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()