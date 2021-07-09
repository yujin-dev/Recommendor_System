import gensim
from model.performance import *

def LDAmodel(id2word, corpus, num_topics=20, passes=10, per_word_topics=True):
    LDAmodel= gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=passes,
                                              per_word_topics=per_word_topics)
    return LDAmodel