from sklearn.model_selection import train_test_split
from pipeline.runner import *
import seaborn as sns
import numpy as np
import random
import scipy


def split_train_test(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=10)
    return train_data, test_data

def make_train_matrix(train_data, test_data):
    user_item_mtx = train_data.pivot_table(values="rating", index="userId", columns="movieId")
    # empty entry는 평균 평점으로 fill
    user_item_mtx = user_item_mtx.apply(lambda x: x.fillna(x.mean()), axis=1)
    return user_item_mtx

def calculate_SVD(matrix, k=50):
    # numpy SVD
    u, sig, i = np.linalg.svd(matrix)
    # 유저 matrix 중 k개 latent factor만 사용
    u_hat = u[:, :k]
    # 아이템 matrix 중 k개 latent factor만 사용
    i_hat = i[:k, :]
    # latent factor 대각 행렬
    # sig : singular matrix로 가장 큰 값부터 내림차순
    sig_hat = sig[:k] * np.identity(k, np.float)
    user_factors = u_hat
    item_factors = np.matmul(sig_hat, i_hat)
    return user_factors, item_factors



@runner
def main(data, breakpoint = False, verbose=False, tag="SVD_movielens", final_result=True):
    return {
        Step(1): (split_train_test, data),
        Step(2): (make_train_matrix, Step(1)),
        Step(3): (calculate_SVD, Step(2)),
    }