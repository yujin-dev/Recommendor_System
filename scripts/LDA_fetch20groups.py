from sklearn.datasets import fetch_20newsgroups
from pipeline.LDA_fetch20groups import *
from scripts.utils import *

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

id2word, corpus = main(documents, breakpoint="Step4")
header("변환된 [문서0] (word_id, word, frequency) 확인")
print([(_id, id2word[_id], freq) for _id, freq in corpus[0]])

ldamodel = main(documents)
header("topic 0의 top 5 단어 및 확률")
for _id, prob in ldamodel.get_topic_terms(topicid=0, topn=5):
    print(ldamodel.id2word[_id], prob)

header("topic별 top 10 단어 분포")
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)

header("문서0의 토픽별 분포")
print(ldamodel.get_document_topics(corpus[0]))

header("문서0의 토픽별 분포(threshold = 0)")
print(ldamodel.get_document_topics(corpus[0], minimum_probability=0))

header("모든 문서의 토픽 분포")
result = []
for i, topic_list in enumerate(ldamodel[corpus]):
    topics = topic_list[0]
    topics = sorted(topics, key=lambda x: (x[1]), reverse=True) # 확률 기준으로 내림차순
    if len(topics) > 0:
        topic_num, prop_topic = topics[0][0], topics[0][1]
        result.append([i, int(topic_num), round(prop_topic, 4)])
topic_table = pd.DataFrame(result, columns=['doc_id', 'top_topic', 'prob'])
print(topic_table)

header('Perplexity(낮을수록 좋은 모델)')
print(ldamodel.log_perplexity(corpus))

## Ipython
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(ldamodel, corpus, id2word)
# print(vis)

