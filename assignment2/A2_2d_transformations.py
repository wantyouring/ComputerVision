from modules import *

def main():
    print('-----A2_2d_transformations start-----')
    input_img_file_name = 'smile.PNG'
    gray = open_img(input_img_file_name)
    M = [[1,0,0],[0,1,0],[0,0,1]]
    plane = get_transformed_image(gray,M) # 초기상태 plane

    cv2.imshow('test', plane)
    cv2.waitKey(1)

    while True:
        pressed_key = input("put key : ")
        plane_draw = np.copy(plane) # 그림 계속 그리며 출력할 plane
        # x,y축 이동
        if pressed_key == 'a':
            M = np.dot(M_MOVE_MINUS_X,M)
            plane_draw = get_transformed_image(gray,M)
        elif pressed_key == 'd':
            M = np.dot(M_MOVE_PLUS_X, M)
            plane_draw = get_transformed_image(gray, M)
        elif pressed_key == 'w':
            M = np.dot(M_MOVE_PLUS_Y,M)
            plane_draw = get_transformed_image(gray,M)
        elif pressed_key == 's':
            M = np.dot(M_MOVE_MINUS_Y,M)
            plane_draw = get_transformed_image(gray,M)
        # 회전이동
        elif pressed_key == 'r':
            M = np.dot(M_ROTATE_COUNTER_CLOCK,M)
            plane_draw = get_transformed_image(gray,M)
        elif pressed_key == 'R':
            M = np.dot(M_ROTATE_CLOCK,M)
            plane_draw = get_transformed_image(gray,M)
        # axis flip
        elif pressed_key == 'f':
            M = np.dot(M_FLIP_Y_AXIS,M)
            plane_draw = get_transformed_image(gray,M)
        elif pressed_key == 'F':
            M = np.dot(M_FLIP_X_AXIS,M)
            plane_draw = get_transformed_image(gray, M)
        # x,y방향 확대, 축소
        elif pressed_key == 'x':
            M = np.dot(M_SHRINK_X,M)
            plane_draw = get_transformed_image(gray,M)
        elif pressed_key == 'X':
            M = np.dot(M_ENLARGE_X, M)
            plane_draw = get_transformed_image(gray, M)
        elif pressed_key == 'y':
            M = np.dot(M_SHRINK_Y, M)
            plane_draw = get_transformed_image(gray, M)
        elif pressed_key == 'Y':
            M = np.dot(M_ENLARGE_Y, M)
            plane_draw = get_transformed_image(gray, M)
        # 초기상태
        elif pressed_key == 'H':
            M = M_IDENTITY
            plane_draw = np.copy(plane)
        # 종료
        elif pressed_key == 'Q':
            cv2.destroyAllWindows()
            return

        cv2.arrowedLine(plane_draw, ij(400, 0), ij(-400, 0), 0) # 함수 자체에서 xy로 받아 순서 바꿔줬음
        cv2.arrowedLine(plane_draw, ij(0, 400), ij(0, -400), 0)

        # cv2.arrowedLine(plane_draw, xy_cor(-400, 0), xy_cor(400, 0), 0)
        # cv2.arrowedLine(plane_draw, xy_cor(0, -400), xy_cor(0, 400), 0)
        cv2.imshow('test',plane_draw)
        cv2.waitKey(1)

    print('-----A2_2d_transformations end-----')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
