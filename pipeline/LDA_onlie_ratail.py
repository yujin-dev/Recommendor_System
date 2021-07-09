from model import LDAmodel, get_precision, get_recall
from pipeline.runner import *
import gensim.corpora as corpora

def preprocess(data):
    # 사용자 아이디가 없는 데이터 제외
    # quantity 음수인 데이터 제외
    df = data[~(data['CustomerID'].isnull())&(data['Quantity']>0)]
    df['CustomerID'] = df.CustomerID.astype(int)
    df['StockCode'] = df['StockCode'].astype(str)
    df = df[['InvoiceNo', 'StockCode', 'Quantity', 'CustomerID', 'InvoiceDate']]
    df['ym'] = df['InvoiceDate'].apply(lambda x: str(x)[:7])
    return df

def split_train_test(data):
    train_data = data[(data["ym"]>='2011-09')&(data["ym"]<='2011-11')]
    test_data = data[(data['ym']=='2011-12')]
    return train_data, test_data

def train_groupby(train_data, test_data):
    train_groupby = train_data.groupby(['CustomerID'])
    return train_groupby


def make_train_data(data):
    doc_list = []
    for user_id, user_df in data:
        stockcodes = user_df['StockCode'].values.tolist()
        doc_list.append(stockcodes)
    id2word = corpora.Dictionary(doc_list)
    corpus = [id2word.doc2bow(doc) for doc in doc_list]
    return id2word, corpus


@runner
def main(data, breakpoint=False, verbose=False, tag="LDA_online_ratail"):
    return {
        Step(1): (preprocess, ),
        Step(2): (split_train_test, Step(1)),
        Step(3): (train_groupby, Step(2)),
        Step(4): (make_train_data, Step(3)),
        Step(5): (partial(LDAmodel, per_word_topics=False), Step(4)),
    }

