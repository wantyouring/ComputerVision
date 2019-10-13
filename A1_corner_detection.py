from modules import *

def main():
    print('-----a1_corner_detection shapes.PNG start-----')
    input_img_file_name = 'shapes.PNG'
    gray = open_img(input_img_file_name)
    a1_corner_detection_result(input_img_file_name, gray)  # a1_edge_detection 결과 출력함수
    print('-----a1_corner_detection shapes.PNG end-----')

    print('-----a1_corner_detection lenna.PNG start-----')
    input_img_file_name = 'lenna.PNG'
    gray = open_img(input_img_file_name)
    a1_corner_detection_result(input_img_file_name, gray)  # a1_edge_detection 결과 출력함수
    print('-----a1_corner_detection lenna.PNG end-----')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
