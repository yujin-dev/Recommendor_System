from pipeline.LDA_onlie_ratail import *
from data.setup_data import Data
from scripts.utils import *

parse_date = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M")
retail_df = pd.read_csv(Data.online_retail, encoding="utf-8", parse_dates=["InvoiceDate"],
                        date_parser=parse_date)

train_data, test_data = main(retail_df, breakpoint="Step2")
train_groupby = main(retail_df, breakpoint="Step3")
id2word, corpus = main(retail_df, breakpoint="Step4")
ldamodel = main(retail_df)

header("topic별 top 10 아이템 분포")
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)

header("20개 토픽의 Top 5 아이템 분포")
stock_to_description = {row["StockCode"]: row["Description"] for _, row in retail_df.iterrows()}
for i in range(20):
    recommend = ldamodel.show_topic(topicid=i, topn=5)
    print(i, [stock_to_description[item] for item, score in recommend])
    
header("전체 유저의 토픽 분포")
user_topic_dist = {}
for user_id, user_df in train_groupby:
    document = user_df['StockCode'].values.tolist()
    user_topic_dist[user_id] = ldamodel.get_document_topics(id2word.doc2bow(document), minimum_probability=0)
print("12682번 user 토픽 분포\n", user_topic_dist[12682])


header("12682번 유저 토픽 분포에서 가장 확률이 높은 토픽 선택 → Top 20 아이템 추천")
user_topics = user_topic_dist[12682]
user_topics = sorted(user_topics, key=lambda x: (x[1]), reverse=True)
user_topic = user_topics[0][0]
print('가장 확률이 높은 토픽 : ', user_topic)
recommend = ldamodel.show_topic(topicid=user_topic, topn=20)
print('Top 20 아이템 추천 : ', [item for item, _ in recommend])
relevant = test_data[test_data["CustomerID"]==12682]['StockCode'].unique()
print("실제 12682번 유저가 데이터에서 선호한 아이템 : ", relevant)

header("유저별 LDAmodel 추천(Top 20) 성능")
train_user_ids = train_data['CustomerID'].unique()
test_user_ids = test_data['CustomerID'].unique()

topn = 20
default_recommend = list(train_data.groupby('StockCode')['Quantity'].count().sort_values(ascending=False)[:topn].index)
precisions, recalls = [], []
for user_id in test_user_ids:
    if user_id in train_user_ids:
        user_topics = user_topic_dist[user_id]
        user_topics = sorted(user_topics, key = lambda x: (x[1]), reverse=True)
        user_topic = user_topics[0][0]
        recommend = [item for item, _ in ldamodel.show_topic(topicid=user_topic, topn=topn)]
    else:
        recommend =default_recommend

    relevant = test_data[test_data['CustomerID'] == user_id]['StockCode'].unique()
    precisions.append(get_precision(relevant, recommend))
    recalls.append(get_recall(relevant, recommend))
print("Precision@K : ", np.mean(precisions))
print("Recall@K : ", np.mean(recalls))