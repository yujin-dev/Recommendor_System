from pipeline.SVD_movielens import *
from model import get_rmse
from data.setup_data import Data
from scripts.utils import *
from tqdm import tqdm

ratings_df = pd.read_csv(Data.ratings, encoding='utf-8')

results = main(ratings_df, final_result=False)

train_data, test_data = results["Step1"]
user_item_matrix = results["Step2"]
user_factors, item_factors = results["Step3"]

header("latent factor = 50에 대한 user factors, item_factors shape")
print(user_factors.shape, item_factors.shape)

header("user factors, item_factors로 평점 예측치 산출")
prediction_result = pd.DataFrame(np.matmul(user_factors, item_factors),
                                 columns=user_item_matrix.columns.values, index=user_item_matrix.index.values)
print(prediction_result)

header("test 데이터에 대한 예측")
# test 데이터에서 새로 등장하는 유저, 아이템에 대입하기 위한 global rating
global_rating = train_data['rating'].mean()
result=[]
for _, row in tqdm(test_data.iterrows()):
    user_id, movie_id, = row['userId'], row['movieId']
    true_rating = row['rating']
    if user_id in prediction_result.index.values and movie_id in prediction_result.columns.values:
        pred_rating = prediction_result.loc[user_id][movie_id]
    else:
        pred_rating = global_rating
    result.append([user_id, movie_id, true_rating, pred_rating])
pred_result = pd.DataFrame(result, columns=['user_id', 'movie_id', 'true_rating', 'pred_rating'])
print(pred_result)

header("RMSE")
print(get_rmse(pred_result['true_rating'].values, pred_result['pred_rating'].values))


