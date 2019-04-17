#!/usr/bin/env python
# coding: utf-8

# ## 词向量

# ## BOW

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# In[1]:


corpus = [
    'the sky is blue',
    'sky is blue and sky is beautiful',
    'the beautiful is so blue',
    'i love blue cheese'
]

doc = ['loving this blue sky today']


# In[47]:


def bow_extract(corpus, ngrams=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngrams)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# In[48]:


def display_features(fea_names, features):
    df = pd.DataFrame(data=features, columns=fea_names)
    print(df)


# In[49]:


bow_vectorizer, bow_features = bow_extract(corpus)

fea_names = bow_vectorizer.get_feature_names()
bow_features = bow_features.todense()

print(fea_names)
print(bow_features)

display_features(fea_names, bow_features)


# In[51]:


# transform new doc to vector of bow model
doc_features = bow_vectorizer.transform(doc).todense()

display_features(fea_names, doc_features)


# ## TFIDF
# ### 基于词袋模型的词进行计算

# In[29]:


from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


# In[39]:


def tfidf_extract(bow_features):
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    tfidf_features = transformer.fit_transform(bow_features)
    return transformer, tfidf_features


# In[52]:


tfidf_transformer, tfidf_features = tfidf_extract(bow_features)

print(tfidf_features)

tfidf_features = np.round(tfidf_features.todense(), 2)
print(tfidf_features)

display_features(fea_names, tfidf_features)


# In[53]:


# transform new doc to vector of bow model
doc_tfidf_features = tfidf_transformer.transform(doc_features)
doc_tfidf_features = np.round(doc_tfidf_features.todense(), 2)

display_features(fea_names, doc_tfidf_features)


# ## TFIDF - 2
# ### 直接计算

# In[50]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[51]:


def tfidf_extract2(corpus, ngrams=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=ngrams, norm='l2', smooth_idf=True, use_idf=True)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# In[55]:


tfidf_vectorizer, tfidf_features = tfidf_extract2(corpus)

display_features(fea_names, np.round(tfidf_features.todense(), 2))


# In[54]:


# transform new doc to vector of bow model
doc_tfidf_features = tfidf_vectorizer.transform(doc)

display_features(fea_names, np.round(doc_tfidf_features.todense(), 2))


# ## Word2Vec

# In[2]:


from gensim.models import Word2Vec
import nltk


# In[3]:


sentences = [nltk.word_tokenize(sen) for sen in corpus]
docs = [nltk.word_tokenize(sen) for sen in doc]

model = Word2Vec(sentences, size=10, window=10, min_count=2)


# In[5]:


print(model.wv['sky'])
print(model.wv.most_similar('blue'))


# In[4]:


# stored in a KeyedVectors instance
word_vec = model.wv
del model


# In[5]:


print(word_vec['sky'])


# In[ ]:


# save and load model
model.save(filename)
model = Word2Vec.load(filename)


# ## 句子向量
# ### 平均词向量表示

# In[6]:


import numpy as np


# In[32]:


def average_words_vector(sen, model, vocabulary, ndims):
    feature = np.zeros((ndims,), dtype="float64")
    word_count = 0
    
    for word in sen.split():
        if word in vocabulary:
            feature = np.add(feature, model.wv[word])
            word_count = word_count + 1
            
    feature = np.divide(feature, word_count)
    
    return feature


# In[34]:


def get_sentence_feature(corpus, model, ndims=10):
    vocabulary = set(model.wv.index2word)
    features = [average_words_vector(sen, model, vocabulary, ndims) for sen in corpus]
    return np.array(features)


# In[38]:


sen_feature = get_sentence_feature(corpus, model)
print(np.round(sen_feature, 3))


# ### TFIDF加权平均词向量

# In[ ]:




