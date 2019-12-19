import numpy as np

'''
n * 128차원 SIFT features.

numpy만 사용.
Backword model 기본적으로 할 듯
TF-IDF : 덜 등장하는 것에 weight 줌.
Dictionary Learning: 
Learn Visual Words using clustering 
Encode: 
build Bags‐of‐Words (BOW) vectors for each image 
여기까지 2번과제 과정.

2-1. 비슷한 이미지 찾기. 이미지 검색.
250개 다른 이미지 4개씩 총 1000장.
4개중 몇 개를 찾는지가 점수.
Sift features는 줌. 1 이미지당 128byte feature n개. binary파일.
D는 최대 1024.
경로는 상대경로
레포트
랜덤 사용시 랜덤시드

Sift/
Image/
eval_list.exe
compute_desc.exe
eval.exe
'''

'''
sift features 
'''

sifts = []

# sift들 불러오기.
for i in range(100000,101000):
    with open("./sift/sift{}".format(i), "rb") as f:
        sift = []
        bytes = f.read()
        sift = np.frombuffer(np.array(bytes),dtype=np.uint8)
        np_sift = np.asarray(sift)
        np_sift = np.reshape(np_sift, (int(len(sift) / 128), 128))
        sifts.append(np_sift)

print(np.shape(sifts))

# sift 거리 비교하기.

# feature matching을 해서 일정 threshold이하로 매칭되는게 많은 이미지 판별.
print(sifts[0])
print(sifts[1])

# print(np.array(sifts[1]) - np.array(sifts[0]))
# print(sifts[1] - sifts[0])

N = np.array([1000])
D = np.array(512)
d = np.zeros((1000,512),dtype=np.float32)

with open("./A3_2014312993.des", "wb") as f:
    f.write(N.tobytes())
    f.write(D.tobytes())
    f.write(d.tobytes())