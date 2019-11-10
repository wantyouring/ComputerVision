
from modules import *

def main():
    img_desk = 'cv_desk.PNG'
    img_cover = 'cv_cover.jpg'

    gray_desk = cv2.imread('cv_desk.PNG', cv2.IMREAD_GRAYSCALE)
    gray_cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp = orb.detect(gray_desk, None)
    kp, des = orb.compute(gray_desk, kp)
    print("kp : {} \n des : {} \n".format(kp,des))

    kp = orb.detect(gray_cover, None)
    kp, des = orb.compute(gray_cover, kp)
    print("kp : {} \n des : {} \n".format(kp, des))

    # hamming distance로 top10 match pairs 구하기.


if __name__ == '__main__':
    main()