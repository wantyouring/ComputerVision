from modules import *

def main():
    print('-----A2_2d_transformations start-----')
    input_img_file_name = 'smile.PNG'
    gray = open_img(input_img_file_name)
    M = M_IDENTITY
    init_plane = get_transformed_image(gray,M) # 초기상태 plane
    smile_plane = get_transformed_image(gray,M) # smile만 존재하는 plane.
    plane_draw = np.copy(init_plane) # 화살표 함께 출력하는 plane.
    cv2.arrowedLine(plane_draw, ij(400, 0), ij(-400, 0), 0)  # 함수 자체에서 xy로 받아 순서 바꿔줬음
    cv2.arrowedLine(plane_draw, ij(0, 400), ij(0, -400), 0)
    cv2.imshow('test', init_plane)
    cv2.waitKey(1)

    while True:
        pressed_key = input("put key : ")
        #plane_draw = np.copy(plane) # 그림 계속 그리며 출력할 plane

        if pressed_key == 'a':
            smile_plane = get_transformed_image(smile_plane,M_MOVE_MINUS_X)
        elif pressed_key == 'd':
            smile_plane = get_transformed_image(smile_plane, M_MOVE_PLUS_X)
        elif pressed_key == 'w':
            smile_plane = get_transformed_image(smile_plane, M_MOVE_PLUS_Y)
        elif pressed_key == 's':
            smile_plane = get_transformed_image(smile_plane, M_MOVE_MINUS_Y)
            # 회전이동
        elif pressed_key == 'r':
            smile_plane = get_transformed_image(smile_plane, M_ROTATE_COUNTER_CLOCK)
        elif pressed_key == 'R':
            smile_plane = get_transformed_image(smile_plane, M_ROTATE_CLOCK)
            # axis flip
        elif pressed_key == 'f':
            smile_plane = get_transformed_image(smile_plane, M_FLIP_Y_AXIS)
        elif pressed_key == 'F':
            smile_plane = get_transformed_image(smile_plane, M_FLIP_X_AXIS)
            # x,y방향 확대, 축소
        elif pressed_key == 'x':
            smile_plane = get_transformed_image(smile_plane, M_SHRINK_X)
        elif pressed_key == 'X':
            smile_plane = get_transformed_image(smile_plane, M_ENLARGE_X)
        elif pressed_key == 'y':
            smile_plane = get_transformed_image(smile_plane, M_SHRINK_Y)
        elif pressed_key == 'Y':
            smile_plane = get_transformed_image(smile_plane, M_ENLARGE_Y)
        elif pressed_key == 'R':
            smile_plane = get_transformed_image(smile_plane,M_ROTATE_COUNTER_CLOCK)
        elif pressed_key == 'H':
            smile_plane = get_transformed_image(init_plane,M_IDENTITY)

        # 종료
        elif pressed_key == 'Q':
            cv2.destroyAllWindows()
            return

        plane_draw = np.copy(smile_plane) # 화살표까지 그리는 plane
        cv2.arrowedLine(plane_draw, ij(400, 0), ij(-400, 0), 0) # 함수 자체에서 xy로 받아 순서 바꿔줬음
        cv2.arrowedLine(plane_draw, ij(0, 400), ij(0, -400), 0)

        cv2.imshow('test',plane_draw)
        cv2.waitKey(1)

    print('-----A2_2d_transformations end-----')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
