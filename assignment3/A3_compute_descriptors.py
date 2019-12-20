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
N = 1000
D = 1000

sifts = [] # (1000,x,128)
features = np.zeros((760512,128),dtype=np.float32)
clusters = np.zeros((D,),dtype=int)
_feature_i = 0

N_np = np.array([N],dtype=np.int32)
D_np = np.array([D],dtype=np.int32)
d = np.zeros((N,D),dtype=np.float32)

print(N_np)
print(D_np)

# sift들 불러오기, kmeans clustering용 붙여서 저장.
for i in range(100000,101000):
    with open("./sift/sift{}".format(i), "rb") as f:
        sift = []
        bytes = f.read()
        sift = np.frombuffer(np.array(bytes),dtype=np.uint8)
        np_sift = np.asarray(sift)
        np_sift = np.reshape(np_sift, (int(len(sift) / 128), 128))
        sifts.append(np_sift)
        features[_feature_i:_feature_i + int(len(sift) / 128),0:128] = np_sift.copy()
        _feature_i += int(len(sift) / 128)

print(np.shape(sifts))

centers = np.zeros((D,128),dtype=np.float32) # 중심 좌표들 저장. (D,128)

# 1줄 평균
for i in range(D):
    centers[i,:] = np.sum(sifts[i][:,:]/128,axis=0).copy() # 2.67

# # 1줄 + 4개 평균
# for i in range(D):
#     if i%4==0:
#         tmp = np.zeros((128))
#         for j in range(4):
#             tmp += np.sum(sifts[i+j][0,:]/512,axis=0) # 수정 필요!
#         for j in range(4):
#             if i+j>=D:
#                 break
#             centers[i+j,:] = tmp.copy()

# # 4개 평균
# for i in range(D):
#     if i%4==0:
#         tmp = np.zeros((128))
#         for j in range(4):
#             tmp += sifts[i+j][0,:]/4
#
#         for j in range(4):
#             if i+j>=D:
#                 break
#             centers[i+j,:] = tmp.copy()

# 중심 임의로 D개.(각 이미지들 처음 feature로 하기)
# centers[:,:] = features[0:D,:].copy()

clusters = [] # (1000,x)

# k means 학습
Learn = 20
for i in range(Learn): # 학습 횟수
    _d = np.zeros((N, D), dtype=np.float32)
    for j in range(N):
        cluster = np.zeros(len(sifts[j]),dtype=int)
        print(j)
        for k in range(len(sifts[j])):
            dis_center = sifts[j][k,:] - centers[:,:] # 1줄 빼기 center들.
            dis = np.linalg.norm(dis_center,axis=1)
            mins = np.argmin(dis) # D까지 중 최소 index
            cluster[k] = mins
            if i==Learn-1: # 마지막에 d계산.
                d[j][mins] += 1
            _d[j][mins] += 1
        clusters.append(cluster)

    # descriptor 파일 만들기.
    with open("./A3_2014312993_{}.des".format(i), "wb") as f:
        f.write(N_np.tobytes())
        f.write(D_np.tobytes())
        f.write(_d.tobytes())

    # 중심 한 클러스터 중심값으로 갱신.
    clusters_cnt = np.zeros((D))
    clusters_avg = np.zeros((D, 128))

    print("len(clusters[j]):{}".format(len(clusters[j])))

    for j in range(N):
        for k in range(len(clusters[j])):
            clusters_avg[clusters[j][k]] += sifts[j][k]
            clusters_cnt[clusters[j][k]] += 1
    centers = (clusters_avg / np.reshape((clusters_cnt+np.finfo(float).eps),(-1,1))).copy()

print(d)
print(np.sum(d))

# descriptor 파일 만들기.
with open("./A3_2014312993.des", "wb") as f:
    f.write(N_np.tobytes())
    f.write(D_np.tobytes())
    f.write(d.tobytes())


