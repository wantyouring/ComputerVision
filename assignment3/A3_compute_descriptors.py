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

sift = []

with open("./sift/sift100000", "rb") as f:
    byte = f.read(1)
    cnt = 0
    while byte:
        # print(int.from_bytes(byte,"big"))
        sift.append(int.from_bytes(byte, "big"))
        byte = f.read(1)
        cnt += 1
print(cnt)

np_sift = np.asarray(sift)
np_sift = np.reshape(np_sift,(int(len(sift)/128),128))
print(np.shape(np_sift))