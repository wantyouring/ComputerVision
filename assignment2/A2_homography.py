
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

    # 2-1

    orb = cv2.ORB_create()

    # desk 그림
    kp1 = orb.detect(gray_desk, None)
    kp1, des1 = orb.compute(gray_desk, kp1)
    # print("kp : {} \n des : {} \n".format(np.shape(kp1),np.shape(des1)))

    # cover 그림
    kp2 = orb.detect(gray_cover, None)
    kp2, des2 = orb.compute(gray_cover, kp2)
    print("kp : {}\n des : {} \n".format(kp2[0].pt, np.shape(des2)))

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

    # 정렬 전에 매칭되는 scrP, destP 구해놓기(2-2)
    srcP = []
    destP = []

    for i in range(len(matches)):
        kp1_x = kp1[i].pt[0]
        kp1_y = kp1[i].pt[1]
        kp2_x = kp2[matches[i].trainIdx].pt[0]
        kp2_y = kp2[matches[i].trainIdx].pt[1]

        srcP.append([kp1_x,kp1_y])
        destP.append([kp2_x,kp2_y])

    srcP = np.array(srcP) # shape of srcP : (N,2) (N = 500)
    destP = np.array(destP)

    # print("srcP:{}\n".format(np.shape(srcP)))

    matches = sorted(matches,key=lambda obj:obj.distance) # distance 기준 정렬

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.match(des1,des2) # kp1[i] -> kp2[matches[i]]

    # print("matches shape : {}, matches : {}".format(np.shape(matches),matches))
    # print("match 1 element : distance{}\n trainidx:{}\n queryidx:{}\n imgidx:{}".
    #       format(matches[2].distance,matches[2].trainIdx,matches[2].queryIdx,matches[2].imgIdx))

    res = None
    res = cv2.drawMatches(gray_desk,kp1,gray_cover,kp2,matches[:10],res,flags=2) # flags=2 option single points 없애줌.
    cv2.imshow('res',res)
    cv2.waitKey(0)

    # 2-2
    H = compute_homography(srcP,destP)



if __name__ == '__main__':
    main()