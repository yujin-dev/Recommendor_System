from nltk.corpus import stopwords
import gensim.corpora as corpora
from pipeline.runner import *
from model import LDAmodel
import gensim
import nltk


def preprocess(data):
    # 텍스트 데이터 프리프로세싱
    news_df = pd.DataFrame({'document':data})
    # 문서 내 특수문자 제거
    news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
    # 문서 내 길이가 3 이하인 단어 제거
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    # 전체 단어에 대한 소문자 변환
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
    return news_df

def get_stopwords():
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    return stop_words

def remove_stopwords(data, stopwords):
    tokenized_doc = data['clean_doc'].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stopwords]) # list
    return tokenized_doc

def make_train_data(tokenized_doc):
    """
        각 단어 → (word_id, word_frequency)로 변환됨.
        word_id : 단어가 정수로 인코딩된 값
        word_frequency : 해당 문서에서의 빈도수
    """
    id2word = corpora.Dictionary(tokenized_doc)
    corpus = [id2word.doc2bow(text) for text in tokenized_doc]
    return id2word, corpus


@runner
def main(data, breakpoint=False, verbose=False, tag="LDA_fetch20groups"):
    return {
        Step(1): (preprocess, ),
        Step(2): (get_stopwords, ),
        Step(3): (remove_stopwords, [Step(1), Step(2)]),
        Step(4): (make_train_data, Step(3)),
        Step(5): (LDAmodel, Step(4))
    }
