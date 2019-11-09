from modules import *

def main():
    print('-----A2_2d_transformations start-----')
    input_img_file_name = 'smile.PNG'
    gray = open_img(input_img_file_name)
    M = [[1,0,0],[0,1,0],[0,0,1]]
    plane = get_transformed_image(gray,M) # 초기상태 plane
    cv2.imshow('test', plane)
    cv2.waitKey()

    while True:
        pressed_key = input("put key : ")
        plane_draw = np.copy(plane) # 그림 계속 그리며 출력할 plane
        if pressed_key == 'a':
            #plane_draw = get_transformed_image(gray, M)
            1
        elif pressed_key == 'x':
            M = np.dot([[0.95,0,0],[0,0.95,0],[0,0,1]],M)
            plane_draw = get_transformed_image(gray,M)
        cv2.arrowedLine(plane_draw, xy_cor(-400, 0), xy_cor(400, 0), 0)
        cv2.arrowedLine(plane_draw, xy_cor(0, -400), xy_cor(0, 400), 0)
        cv2.imshow('test',plane_draw)
        cv2.waitKey()

    print('-----A2_2d_transformations end-----')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
