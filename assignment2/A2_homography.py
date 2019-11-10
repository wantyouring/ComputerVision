
from modules import *

# top 10 feature matching을 어떻게 할 것인가???. matcher 구현하기 문제.
# orb similarity는 hamming distance로 계산.
# 이미지의 key point와 descriptor.
# A의 i번째 descriptor - B의 모든 descriptor와 비교해 가장 일치하는(hamming distance 짧은) j번째 descriptor 선정.
# match 배열에 1. A descriptor들과 2. 짝지은 B descriptor 3. hamming distance 튜플로 저장하자.
# ㄴ 매칭한 hamming distance 저장하는건 제일 잘 매칭되는 top10 뽑기 위해서.
# top 10 match point 출력하기.

def main():
    img_desk = 'cv_desk.PNG'
    img_cover = 'cv_cover.jpg'

    gray_desk = cv2.imread('cv_desk.PNG', cv2.IMREAD_GRAYSCALE)
    gray_cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = None

    orb = cv2.ORB_create()
    kp1 = orb.detect(gray_desk, None)
    kp1, des1 = orb.compute(gray_desk, kp1)
    print("kp : {} \n des : {} \n".format(np.shape(kp1),np.shape(des1)))

    # img2 = cv2.drawKeypoints(gray_desk, kp, img2, (0, 0, 255))
    # cv2.imshow('orb', img2)
    # cv2.waitKey(0)



    kp2 = orb.detect(gray_cover, None)
    kp2, des2 = orb.compute(gray_cover, kp2)
    print("kp : {} \n des : {} \n".format(np.shape(kp2), np.shape(des2)))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1,des2)

    print("matches shape : {}, matches : {}".format(np.shape(matches),matches))
    print("match 1 element : {}".format(matches[0]))
    res = None
    res = cv2.drawMatches(gray_desk,kp1,gray_cover,kp2,matches[:30],res)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    # img2 = cv2.drawKeypoints(gray_cover,kp,img2,(0,0,255))
    # cv2.imshow('orb',img2)
    # cv2.waitKey(0)

    # hamming distance로 top10 match pairs 구하기.


if __name__ == '__main__':
    main()