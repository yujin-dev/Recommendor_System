import matplotlib.pyplot as plt

from pipeline.MA_movielens import *
from data.setup_data import Data
from scripts.utils import *

ratings_df = pd.read_csv(Data.ratings, encoding='utf-8')
sgdmodel, alsmodel = main(ratings_df)

header("SGD 학습")
x = sgdmodel.result.epoch.values + 1
y = sgdmodel.result.rmse.values
plt.figure(figsize=((8,4)))
plt.plot(x, y)
plt.xticks(x, y)
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.grid(axis="y")


header()