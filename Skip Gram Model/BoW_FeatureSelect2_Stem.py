# .py version of the notebook curated from notebook prototypes to run and save results
# Bag of Words - this analysis is similar to the other BoW w/ the main change being feature selection
# The first part of feature selection is different and is based on paper by German authors


################## Speciying analysis configuration
config_json = {
    "Article_Set": "stem_articles", #"all_articles_para_cut.pickle",
    "Vocab": "FeatureSelectLemm_2000",
    "Start_date": "2011-01-01",
    "End_date": "2015-12-31",
    'Returns': "fwd_xmkt_1_10",
    "Output_Dir": "FSL_3900_LM_xmkt10"
    # Scores: "Tfidf"
}

################### Loading Packages
# print "--- Loading Python Packages"

import json
import sys, os, shutil
import numpy as np
from scipy.stats import norm, spearmanr, ttest_ind
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.switch_backend('agg')
from statsmodels.regression.linear_model import OLS
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.naive_bayes import MultinomialNB
import statsmodels.api as sm
import logging
import MySQLdb
import glob
import pymssql
import multiprocessing as mp
import cPickle as pickle
import collections
import pytz
import gc
import re
from scipy.stats.mstats import winsorize, zscore
from scipy import stats
import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

sys.path.append(os.path.expanduser('/research/home/rakesh/research/scripts/_modules/python'))
# /research/home/ryan/Dev/research/scripts/_modules/python - this is where the pybt and jv packages from Ryan are located
# checkout the repo with these files - and reference that

import pybt.data_access as da
import pybt.data_access as da
import pybt.fitting as fitting
import pybt.argument_checks as ac
import pybt.transforms as transforms
from pybt.config_helper import ConfigHelper
from pybt import trading_mappings
from pybt import modeling

import xml.etree.ElementTree as ET


from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize   # maybe other tokenization approaches
from nltk.corpus import wordnet


print "Running with maggi"

out_dir = '/research/home/rakesh/results/BoW/FeatureSelectLemm/' + config_json['Output_Dir']

################## Loading Datasets
print "---- Loading Article Mapping and Returns Data"
tstart_all = time.time()

# Loading universe of stocks
tstart = time.time()
udao = da.UniverseDataAccess('/research/amer_data/amer_v1/universe/')

temp = []
for refdate in udao.refdates:
    temp.append(udao.universe4refdate(refdate))
univ_df = pd.concat(temp)
univ_df = univ_df.drop_duplicates('gvkey', keep = 'last')

# # This loads the mapping between articles and events (which is gvkey + date)

mapping_file = '/sim_shared/DJ_News/amer-v1-filtered/article_mappings/post2_article_mapping.csv'
mapping = pd.read_csv(mapping_file, index_col = 0, parse_dates = ['display_date_EST', 'event_date'])

start_date = '2011-01-01'
end_date = '2015-12-31'

mapping['date'] = pd.to_datetime(mapping['date'])
mask = (mapping['date'] > start_date) & (mapping['date'] <= end_date)
mapping_select = mapping.loc[mask]

# event_date is what has to be matched with the returns where the event date is under the 'date' column - RG
mapping_select = mapping_select.drop(['date'], axis=1)
mapping_select['date'] = mapping_select['event_date']

print "No. of articles loaded between %s and %s are %d" %(start_date, end_date, len(mapping_select))


# Load returns - of each event, returns not related to each article directly

tstart = time.time()
temp = '/research/home/ryan/Dev/research-analysis/rgreen/amer-indicator-sets/packages/amer_v1/earnings/post-earnings/post2_earnings.csv'
ret_file = temp   #os.path.join(NETAPP_DIR, temp)
tdao = da.TrainingDataAccess(ret_file)

start_dt_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
end_dt_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

inds, rets = tdao.training_data_range(start_dt_date, end_dt_date, include_date=True)

# this in effect merges the returns and indices
rets['gvkey'] = inds['gvkey']
rets['date'] = pd.to_datetime(rets['date'])

print "Returns for %d events loaded in %0.2f seconds" % (len(rets), (time.time()-tstart))



#### Loading article data to extract features corresponding to positive and negative returns
print "---- Loading Article Data Set"

# defining format of the pickled data
ArticleMapping = collections.namedtuple('ArticleMapping', ['article_id', 'display_date_EST',
                                                           'level', 'symbology', 'symbol'])
ArticleText = collections.namedtuple('ArticleText', ['headline', 'text'])

# Loading articles
tstart = time.time()
# with open('/sim_shared/DJ_News/amer-v1-filtered/article_mappings/all_articles.pkl', 'rb') as ofp:
#     articles = pickle.load(ofp)

# with open('/research/home/rakesh/notebooks/all_articles_para_cut.pickle', 'rb') as ofp:
#     articles = pickle.load(ofp)

with open('/research/home/rakesh/Article_Pickles/all_articles_lemmed.pkl', 'rb') as ofp:
    articles = pickle.load(ofp)

print "All Articles Loaded in %0.1f sec" % round(time.time()-tstart)


################## Loading Article Processing Helper functions
# print "--- Loading Helper Methods"
import unicodedata
import string

def cleanup(articleText):
    articleText = articleText[:articleText.find("(END)")]
    articleText = articleText[:articleText.find("(MORE TO FOLLOW)")]
    articleText = articleText.replace('\n', '')
    return articleText

def tokenize(caption):
    caption = cleanup(caption)
    if type(caption) != unicode:
        caption = unicode(caption, 'utf-8')
    norm_caption = str(unicodedata.normalize(
                            'NFKD', caption).encode('ascii','ignore'))
    dehyphenated_caption = norm_caption.replace('-', ' ')
    tokenized = dehyphenated_caption.lower().translate(
                    None, string.punctuation).strip()
    return tokenized

def articles2corpus(article_ids):
    to_return = []
    ids = []
    for aid in article_ids:
        t = articles.get(aid)
        if t is not None:
            caption = tokenize(t.text)
            to_return.append(caption)
            ids.append(aid)
    return to_return, ids

def intersection(lst1, lst2):

    # Use of hybrid method - to get the intersection of two lists
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


ps = PorterStemmer()

def stem_list(list_of_words):

    # this takes a list of words instead of the whole text
    # most well suited to stem vocabs or other lists

    inp_tokens = list_of_words
    N = len(inp_tokens)

    list_stemmed = []

    for i in range(N):
        list_stemmed.append(ps.stem(inp_tokens[i]))

    return list_stemmed

## Methods to combine articles for an event

def one_big_article(ids):

    all_articles = ''.join(articles2corpus(ids)[0])

    return all_articles

def worker1(key):

    event_article = {}

    gvkey = key['gvkey']
    date = key['event_date']
    event_article_ids = mapping_select[(mapping_select['event_date'] == date) & (mapping_select['gvkey'] == gvkey)]
    event_article = one_big_article(event_article_ids['article_id'].values)

    return event_article


# # Getting the keys to each event - to get a list of article ids
#
# temp = mapping_select[mapping_select['date'] < end_date]
# temp = temp.drop_duplicates(subset = ['gvkey', 'event_date'])
# keys = temp.to_dict(orient='record')
#
# # collecting all articles together for an event
print "Generating event corpus"
tstart = time.time()
# event_corpus = {}
# i = 0
#
# for key in keys:
#
#     # if (np.mod(i, 5000) == 0):
#     #     print i
#
#     gvkey = key['gvkey']
#     date = str(key['event_date'])
#     event_corpus[gvkey + ' ' + date] = worker1(key)
#
#     i += 1

# output = open('lemmed_events_corpus.pkl', 'wb')
# pickle.dump(event_corpus, output)
# output.close()
# del event_corpus

pkl_file = open('lemmed_event_corpus.pkl', 'rb')
event_corpus = pickle.load(pkl_file)
pkl_file.close()

print "Successfully saved and loaded event corpus"

event_corpus_list = []
for ring in event_corpus.keys():

    event_corpus_list.append(event_corpus[ring])

event_keys = event_corpus.keys()

print "Event corpus generated in %0.1f secs for %d events" %((time.time() - tstart), len(event_keys))

print "---- Completed loading articles"


######### Feature selection section
print "---- Begin Feature Selection Process"

# use CountVectorizer to get the list of all words
tstart = time.time()

vec = CountVectorizer(stop_words='english')
vec.fit(event_corpus_list)

N = len(vec.vocabulary_)

print "CountVectorizer created in %0.2f seconds w/ a vocabulary of %d words" % ((time.time() - tstart), N)


############## WORD FILTERING
### Filtering words against LM vocabs

with open('/sim_shared/Vocabulary/financeVocab-wtoi.json', 'rb') as json_data:
    vocab = json.load(json_data)
vocab['asdf'] = 0 # it needs a zero value

Wordlist = vocab.keys()
wordlist = [element.lower() for element in Wordlist]
del Wordlist

print len(wordlist), "Finance Vocab Full"

# create a limited vocab based on the intersection
# vec_vocab_lower = [element.lower() for element in vec.vocabulary_]
limited_vocab = intersection(vec.vocabulary_, wordlist)
N = len(limited_vocab)
del wordlist
del vec


# ### use this section for other dictionaries
#
# # select only those words which are in a broad dictionary - to remove weblinks, numbers etc. Some proper nouns not removed
# allwords_dict_fname = "/research/home/rakesh/notebooks/400K_Stanford_Words.txt"
# # allwords_dict_fname = '/usr/share/dict/words'
#
# Wordlist = [line.strip() for line in open(allwords_dict_fname)]
# wordlist = [element.lower() for element in Wordlist]
# del Wordlist
#
# # wordlist_stemmed = stem_list(wordlist)
# # wordlist_set_stemmed = set(wordlist_stemmed)
# # wordlist_unique_stemmed = list(wordlist_set_stemmed)
# # print "Stemmed Large Vocabulary brings size down from %d to %d" %(len(wordlist_stemmed), len(wordlist_unique_stemmed))
# # del wordlist_stemmed, wordlist_set_stemmed, wordlist
#
# # lemmatize case - no need to stem/lemm wordlist, can lemm and check the impact
# wordlist_unique_lemmed = wordlist
# del wordlist
#
# # # create a limited vocab based on the intersection
# # limited_vocab = intersection(vec.vocabulary_, wordlist_unique_stemmed)
# # N = len(limited_vocab)
# # del wordlist_unique_stemmed
# # del vec
#
# # create a limited vocab based on the intersection
# limited_vocab = intersection(vec.vocabulary_, wordlist_unique_lemmed)
# N = len(limited_vocab)
# del wordlist_unique_lemmed
# del vec

# print "No. of words filtered w/ vec.vocab_ and %s is %d" % (allwords_dict_fname, N)


# transforming all articles in the corpus w/ the new vocab
# fit and transform to get the dataframe - no need to re-run for transform
tstart = time.time()

limited_vectorizer = CountVectorizer(vocabulary = limited_vocab)
limited_vectorizer_counts = limited_vectorizer.fit_transform(event_corpus_list)

cv_transform_df = pd.DataFrame(index = event_keys, columns = limited_vocab, data = limited_vectorizer_counts.toarray())
# print cv_transform_df.head(n=3)

print "CountVectorizer and transformer created in %0.2f seconds w/ a vocabulary of %d words" % ((time.time() - tstart), N)

print "Separating positive and negative events"
tstart = time.time()
# Create a single unique identifier for an event - later

# rets['date'] = pd.to_datetime(rets['date'])
rets['event_date'] = rets['date']  # just making a copy
rets['date'] = rets['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
rets['gvkey + event_date'] = rets['gvkey'] + ' ' + rets['date']
unique_event_ids = rets['gvkey + event_date'].unique()

# Separating out the positive and negative events and the feature counts for these events)

min_return = 0.0 # min + and - returns to select pos and neg articles

returns_select = config_json["Returns"]

pos_events = rets.loc[rets[returns_select] > min_return]
neg_events = rets.loc[rets[returns_select] <= -min_return]

pos_event_ids = pos_events['gvkey + event_date'].tolist()
neg_event_ids = neg_events['gvkey + event_date'].tolist()

# the limited_vectorizer_counts has all feature counts - so we separate it out to positive and negative
# compare with passing_ids

pos_event_ids2 = intersection(pos_event_ids, event_keys)
neg_event_ids2 = intersection(neg_event_ids, event_keys)

# positive and negative data frame
features_pos_df = cv_transform_df.loc[pos_event_ids2]
features_neg_df = cv_transform_df.loc[neg_event_ids2]

print "Separated positive and negative articles in %0.2f seconds" %(time.time() - tstart)

# Once the positive and negative dfs with the feature counts for all positive and negative articles are available
# sum and divide by no. of articles

vec_all_pos = features_pos_df.sum(axis=0)
vec_all_neg = features_neg_df.sum(axis=0)

vec_pos_zscore = stats.zscore(vec_all_pos)
vec_neg_zscore = stats.zscore(vec_all_neg)

BNS = vec_pos_zscore - vec_neg_zscore
BNS_pval = stats.norm.cdf(BNS)

BNS_df = pd.DataFrame(np.array([BNS, BNS_pval]).T, index=vec_all_pos.index, columns=['BNS', 'Pval'])
BNS_sorted_df = BNS_df.sort_values(by='BNS', ascending=True)

### To plot BNS_array
# BNS_array = np.array(BNS_df['BNS'])
# plt.hist(BNS_array, bins=60)  # arguments are passed to np.histogram
# plt.title("BNS Scores")
# plt.ylabel("No. of features")
# figname = os.path.join(out_dir, 'BNS_hist.png')
# plt.savefig(figname)
# del BNS_array

# Selection - can be based on threshold of BNS value, or prob or no. of features from top and bottom

# # Selection based on prob cdf spread of underlying BNS
# BNS_select_features = BNS_df[(BNS_df['Pval'] <= 0.495) | (BNS_df['Pval'] >= 0.515)]
#
# # Selection based on BNS
# BNS_select_features = BNS_df[(BNS_df['BNS'] <= -0.01) | (BNS_df['BNS'] >= 0.01)]

# Selection based on BNS
nfeatures = 3900
BNS_select_features = pd.concat((BNS_sorted_df.ix[0:nfeatures/2], BNS_sorted_df.ix[-nfeatures/2:]), axis=0)

features_select = BNS_select_features.index.tolist()

print "Calculated BNS scores and selected %d features out of %d in the limited vocab" % (len(features_select), len(BNS_df))


# Creating the final countvectorizers and tfidfvectorizers
tstart = time.time()

twice_limited_vectorizer = CountVectorizer(vocabulary = features_select)
twice_limited_vectorizer_counts = twice_limited_vectorizer.fit_transform(event_corpus_list)

transformer = TfidfTransformer(smooth_idf=True)
tfidf_corpus = transformer.fit_transform(twice_limited_vectorizer_counts)

tfidf_df = pd.DataFrame(index = event_keys, columns = features_select, data = tfidf_corpus.toarray())
# print tfidf_df.head()
# # tfidf_df - is the dataframe of word tfidf density (from vocab) for each article


features_select_df = pd.DataFrame(features_select)

# save vocab
fname = os.path.join(out_dir, 'vocab_used.csv')
features_select_df.to_csv(fname)

print "Final Tfidf features fit in %0.2f secs" %(time.time() - tstart)


############## Creating BoW tfidf scores for all articles
print "---- Creating BoW df in order to feed the NB"

data_end_date = config_json["End_date"]

temp = mapping_select[mapping_select['date'] < data_end_date]
temp = temp.drop_duplicates(subset = ['gvkey', 'event_date'])
keys = temp.to_dict(orient='record')    # this is the df converted to a list! but why?

feature_names = ['feature_' + x for x in features_select]
# each word in the vocab is a feature
print "Total no. of features (vocab words) considered are", len(feature_names)

## functions to enable scoring of events

# input event id to combine all articles for the event and calculate a BoW score
def bow_4_ids(ids):

    one_big_article = ''.join(articles2corpus(ids)[0])

    if len(one_big_article) == 0:
        bow = pd.Series(index = feature_names, data = np.nan)
        bow.at['word_count'] = 0
        bow.at['num_articles'] = 0
    else:
        event_corpus = twice_limited_vectorizer.transform([one_big_article])
#         print event_corpus
        event_tfidf = transformer.transform(event_corpus)                        # calling the tfidf transformer that was fit with limited corpus
        bow = pd.Series(index = feature_names, data = event_tfidf.toarray()[0])  # tfidf values for that event based on the limited vocab
        bow.at['word_count'] = event_corpus.toarray().sum()                      # word count is how many of the vocabs words are counted in here
        bow.at['num_articles'] = len(ids)
    return bow


def worker(key):

    # Each worker is generating the BoW df which is the tfidf score for the features in the vocab
    # this is being run for all articles in the event / articles selected

    gvkey = key['gvkey']
    date = key['event_date']
    event_articles = mapping_select[(mapping_select['event_date'] == date) & (mapping_select['gvkey'] == gvkey)]
    bow = bow_4_ids(event_articles['article_id'].values)
    bow.at['gvkey'] = gvkey
    bow.at['date'] = date
    return bow

# calculating tfidf scores for each event

tstart = time.time()

pool = mp.Pool(processes = 2)
results = pool.map(worker, keys)   #keys are all articles - why pass the whole list of keys, just gvkey and date
# here the worker is being called with the keys selected above for a certain gvkey/date - but this can be done over all events / articles
bow_df = pd.DataFrame(results)
bow_df['date'] = pd.to_datetime(bow_df['date'].values).date

print "BoW df for all articles in %0.1f secs" %(time.time() - tstart)


################ Loading returns file
print "--- Loading Returns File"
tstart = time.time()

tf = '/home/ryan/Dev/research-analysis/rgreen/amer-indicator-sets/packages/amer_v1/earnings/post-earnings/post2_earnings.csv'
ret_data = pd.read_csv(tf, index_col = 0, parse_dates = ['date'])
ret_data['date'] = pd.to_datetime(ret_data['date'].values).date

# Merge the BoW df with the returns using gvkey and date
bow_df_merged = bow_df.merge(ret_data, on = ['gvkey', 'date'], how = 'left')

print "Loaded and merged returns file in %0.1f secs" % (time.time() - tstart)

# ## saving the combined returns and bow df - if needed
outname = os.path.join(out_dir, 'bow_tfidf_returns.csv')
bow_df_merged.to_csv(outname)

## loading the saved bow and return df
tf = os.path.join(out_dir, 'bow_tfidf_returns.csv')
dao = da.TrainingDataAccess(tf)

# There is definitely some overlap here since returns has already been loaded

train_start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
train_end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

all_inds, all_rets  = dao.training_data_range(start_date = train_start_date, #dt.date(2011,1,1),
                                              end_date= train_end_date, #dt.date(2015,12,31),
                                      include_date = True, drop_na='none')
all_rets['gvkey'] = all_inds['gvkey']    # there is only a 1 year difference in the full training data range for returns?
                                         # though there might be training against new models

returns_select = 'fwd_xmkt_1_10'
all_rets['binary'] = (all_rets[returns_select] > 0).astype(int)

features = []
for c in bow_df_merged.columns:
    if 'feature_' in c:
        features.append(c)


################## Fitting tfidf feature scores to returns
print "--- Loading functions for Results Analysis"


def enrich_univ(df):
    exp_list = []
    dates = sorted(df['date'].unique())

    for date in dates:
        univ = udao.last_two_universes(date)
        temp = df[df['date'] == date].copy()
        temp_merged = temp.merge(univ[['gvkey', 'gics_sector', 'gics_industry_group']], on='gvkey')
        exp_list.append(temp_merged)

    return pd.concat(exp_list)

def enrich_pctile(df):
    f = lambda x : 2*x.rank(pct=True)-1
    df['predicted_pctile'] = df.groupby('cv-start-date')['predicted'].apply(f)
    df['predicted_pctile_sec'] = df.groupby(['cv-start-date', 'gics_sector'])['predicted'].transform(f)
    df['predicted_pctile_indgrp'] = df.groupby(['cv-start-date', 'gics_industry_group'])['predicted'].transform(f)
    return df

def stats4period(df, col, quin=True):
    stats = pd.Series(index=['pearson_IC', 'rank_IC', 'rank_IC_pval',
                             'num_long', 'num_short',
                             'mean_long', 'mean_short',
                             'spread', 'spread_pval', 'count'])


    try:
        df['predicted_quintile'] = pd.qcut(df[col], 5, labels=range(1,6))
        df['predicted_decile'] = pd.qcut(df[col], 10, labels=range(1,11))
    except:
        print "Could not cut for date %s"  % cvdate
        return None

    if quin:
        longs = df[df['predicted_quintile'] == 5]
        shorts = df[df['predicted_quintile'] == 1]
    else:
        longs = df[df['predicted_decile'] == 10]
        shorts = df[df['predicted_decile'] == 1]

    stats.loc['pearson_IC'] = np.corrcoef(df['actual'], df[col])[0][1]
    stats.loc['rank_IC'] = spearmanr(df['actual'], df[col])[0]
    stats.loc['rank_IC_pval'] = spearmanr(df['actual'], df[col])[1]
    stats.loc['num_long'] = len(longs)
    stats.loc['num_short'] = len(shorts)
    stats.loc['mean_long'] = longs['actual'].mean()
    stats.loc['mean_short'] = shorts['actual'].mean()
    stats.loc['spread_pval'] = ttest_ind(longs['actual'],
                                         shorts['actual'],
                                         equal_var = False)[1]
    stats.loc['spread'] = stats.loc['mean_long'] - stats.loc['mean_short']
    stats.loc['count'] = df.shape[0]

    return stats

def result_stats(r, col='predicted', quin=True):
    cvdates = r['cv-start-date'].unique()
    stat_tracker = {}

    for cvdate in cvdates:

        temp = r[r['cv-start-date'] == cvdate].copy()
        s = stats4period(temp, col, quin)
        s.name = cvdate
        stat_tracker[cvdate] = s

    quin_stats = pd.DataFrame(data = stat_tracker).T
    quin_stats = quin_stats.sort_index()
    full_sample = stats4period(r, col, quin)

    return quin_stats, full_sample

class NaiveBayesModel(modeling.ForecastModel):

    @staticmethod
    def from_config(config):
        return NaiveBayesModel()

    @staticmethod
    def model_dates(model_dir):
        pass

    def __init__(self):
        self.model = MultinomialNB()

    def train(self, indicators, returns, validation_indicators=None, validation_returns=None, weights=None):
        self.model.fit(indicators, returns)
        self.trained = True

    def predict(self, indicators):
        yhat = self.model.predict_proba(indicators)
        return yhat[:,1]

    def persist(self, out_dir, datestr):
        pass

    def load(self, model_dir, datestr):
        pass


######### Loading the config file and specifying the var values
# The model fitting framework is based of configs so I need to define one
# and tell it what data to fit to and over what time range
# and what cross validation scheme to use.
print "--- Loading the Config file for analysis"

t = '/home/ryan/Dev/research-analysis/rgreen/dai_models/earnings/configs/post2-sentiment'

# This is a base config I have used for a number of approaches in this problem.
cfile = os.path.join(t, 'amer-post2-lincv-embeddings-2QScv-N10-v2.ini')
config = ConfigHelper.from_file(cfile)

# We only look at events with greater than 4 articles (arbitrarily chosen).

config.set_val('run_params', 'filters', 'num_articles;>;4')

# This defines our feature set

for feature in features:
    config.set_val('indicators', feature, 'Z')


# This tells the framework to change the problem to a classification problem where
# one class is if the return after the event is positive and one class is if the
# return after the event is negative.

config.set_val('run_params', 'objective', 'binary')
config.set_val('run_params', 'binary_threshold', '0.0')
config.set_val('run_params', 'binary_threshold_type', 'abs')
config.config_dict['run_params'].pop('train_ret_group_transform')

# This tells the model framework what model class to use.  I have a bunch of predefined ones
# coded in but can also tell it to look at models defined elsewhere (such as in this notebook).

config.set_val('model', 'external_module', '__main__')
config.set_val('model', 'class_name', 'NaiveBayesModel')


### Fitting the Model and Analyzing Results
print "--- Predicting the classification based on tfidf using cross validation"
tstart = time.time()

odir = '/research/home/rakesh/results/temp'
results_temp = fitting.run_cv(dao, config, odir, 1)   # what does this fitting do? I was thinking the NB was used to fit?
results_temp = enrich_univ(results_temp)
results_temp = enrich_pctile(results_temp)

results = results_temp.merge(all_rets, on = ['gvkey', 'date'], how = 'left')

print "Generated predictions in %0.1f secs" %(time.time() - tstart)

# Selecting the returns to use as outputs
t = results.drop('actual', axis=1)
t['actual'] = t[returns_select]
cv_stats, fs_stats = result_stats(t, 'predicted_pctile_sec')

############# Saving and displaying the results
# save json file indicating config params
fname = os.path.join(out_dir, 'config_results.txt')
with open(fname, 'w') as outfile:
    json.dump(config_json, outfile)

# save predicted results for all gvkey and date combination
fname = os.path.join(out_dir, 'results_fit_predicted.csv')
results.to_csv(fname)

# BoW results are already saved above

# save cv_stats and fs_stats
fname = os.path.join(out_dir, 'cross_val_stats.csv')
cv_stats.to_csv(fname)

fname = os.path.join(out_dir, 'full_sample_stats.csv')
fs_stats.to_csv(fname)



# saving plots
ndec = 10
results_plot = results
results_plot['predicted_decile'] = pd.qcut(results_plot['predicted_pctile_sec'], ndec, labels = range(1,ndec+1))
decile_returns = results_plot.groupby('predicted_decile')[returns_select].mean()   # save decile_returns
results_plot.groupby('predicted_decile')[returns_select].mean().plot(kind='bar')
figname = os.path.join(out_dir, 'deciles.png')
plt.savefig(figname)

ndec = 5
results_plot = results
results_plot['predicted_decile'] = pd.qcut(results_plot['predicted_pctile_sec'], ndec, labels = range(1,ndec+1))
quintile_returns = results_plot.groupby('predicted_decile')[returns_select].mean()   # save decile_returns
results_plot.groupby('predicted_decile')[returns_select].mean().plot(kind='bar')
figname = os.path.join(out_dir, 'quintiles.png')
plt.savefig(figname)

fname = os.path.join(out_dir, 'decile_returns.csv')
decile_returns.to_csv(fname)

fname = os.path.join(out_dir, 'quintile_returns.csv')
quintile_returns.to_csv(fname)


print "\n Completed BoW analysis in %0.1f sec" %(time.time() - tstart_all)
