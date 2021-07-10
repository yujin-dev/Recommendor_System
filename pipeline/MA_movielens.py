import numpy as np
from abc import abstractmethod
from surprise import KNNBasic, KNNWithMeans, SVD
from surprise.model_selection.validation import cross_validate
from pipeline.runner import *

def make_array(data):
    user_item_mtx = data.pivot_table(values="rating", index="userId", columns="movieId").fillna(0)
    return user_item_mtx.to_numpy()

class MF:
    def __init__(self, R, K, learning_rate, regularization, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_rmse(self):
        pass

    @abstractmethod
    def predict_ratings(self):
        pass

    @abstractmethod
    def get_R_hat(self):
        pass


class SGD(MF):

    def train(self):

        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        self.bias_u = np.zeros(self.num_users)
        self.bias_i = np.zeros(self.num_items)
        self.bias = np.mean(self.R[np.where(self.R != 0)])
        self.train_data = [
            (u, i, self.R[u, i])
            for u in range(self.num_users)
            for i in range(self.num_items)
            if self.R[u, i] > 0]

        training_result=[]
        for epoch in range(self.iterations):
            np.random.shuffle(self.train_data)
            self.update_sgd()
            rmse = self.get_rmse()
            training_result.append([epoch+1, rmse])
        self.result = pd.DataFrame(training_result, columns=["epoch", "rmse"])


    def update_sgd(self):

        for u, i, true_r in self.train_data:
            prediction = self.predict_ratings(u, i)
            err = (true_r - prediction)
            # 파라미터 업데이트
            self.bias_u[u] += self.learning_rate * (err - self.regularization * self.bias_u[u])
            self.bias_i[i] += self.learning_rate * (err - self.regularization * self.bias_i[i])
            self.P[u, :] += self.learning_rate * (err * self.Q[i, :] - self.regularization * self.P[u, :])
            self.Q[i, :] += self.learning_rate * (err * self.P[u, :] - self.regularization * self.Q[i, :])

    def get_rmse(self):
        users, items = self.R.nonzero()
        predicted_r = self.get_R_hat()
        error = []
        for x, y in zip(users, items):
            error.append(pow(self.R[x, y] - predicted_r[x, y], 2))
        rmse = np.sqrt(np.asarray(error).mean())
        return rmse

    def predict_ratings(self, user, item):
        prediction = self.bias + self.bias_u[user] + self.bias_i[item] + self.P[user, :].dot(self.Q[item, :].T)
        return prediction

    def get_R_hat(self):
        return self.bias + self.bias_u[:, np.newaxis] + self.bias_i[np.newaxis, :] + self.P.dot(self.Q.T)

class ALS(MF):

    def train(self):
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        training_result=[]
        for epoch in range(self.iterations):
            for u, Ru in enumerate(self.R):
                self.P[u] = self.user_latent(u)
            for i, Ri in enumerate(self.R.T):
                self.Q[i] = self.item_latent(i)

            rmse = self.get_rmse()
            training_result.append((epoch, rmse))

    def user_latent(self, user):
        return np.linalg.solve(np.dot(self.P.T, self.P) + self.regularization * np.eye(self.K),
                               np.dot(self.P.T, self.R[user].T)).T

    def item_latent(self, item):
        return np.linalg.solve(np.dot(self.Q.T, self.Q) + self.regularization * np.eye(self.K),
                               np.dot(self.Q.T, self.R[:, item]))

    def get_rmse(self):
        xi, yi = self.R.nonzero()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self.R[x, y] - self.predict_ratings(x, y), 2)
        return np.sqrt(cost/len(xi))

    def predict_ratings(self, item, user):
        return self.P[item, :].dot(self.Q[user, :].T)

    def get_R_hat(self):
        return self.P.dot(self.Q.T)


def main(data):
    ratings_mtx = make_array(data)
    sgd = SGD(ratings_mtx)
    als = ALS(ratings_mtx)
    sgd.train()
    als.train()
    return sgd, als
