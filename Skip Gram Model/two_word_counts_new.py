# this is an example setup to parallelize two word feature selection

import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import collections
import time
import cPickle as pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk import tokenize
import copy
import multiprocessing as mp
import string
import re
from collections import defaultdict

sys.path.append(os.path.expanduser('/research/home/rakesh/research/scripts/_modules/python'))

import pybt.data_access as da
import json

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize   # maybe other tokenization approaches
from funcy import project

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

#########
# Load stop words to be used in countvec

from sklearn.feature_extraction import stop_words

def load_stop_words():

    y = list(stop_words.ENGLISH_STOP_WORDS)
    df_temp = pd.read_csv("/research/home/rakesh/MyCode/not_stop_words.csv", header=None)
    not_stop_words = df_temp[0].values.tolist()

    for word in not_stop_words:

        if (word in y):
            y.remove(word)

    df_temp = pd.read_csv("/research/home/rakesh/MyCode/months_years.csv", header=None)
    months_years = df_temp[0].values.tolist()

    for word in months_years:

        y.append(word.lower())


    df_temp = pd.read_csv("/research/home/rakesh/MyCode/add_djns_stopwords.csv", header=None)
    add_djns_stopwords = df_temp[0].values.tolist()

    for word in add_djns_stopwords:

        y.append(word.lower())

    return y

stop_words_list = load_stop_words()
stop_words_list = stem_list(stop_words_list)
stop_words_list.append('dowjones')

# Helper functions for counting

def analyzer(sentence):

    tokens = [e.lower() for e in map(string.strip, re.split("(\W+)", sentence)) if len(e) > 1 and not re.match("[\W]",e)]

    return tokens

def dict_merge(d1, d2):

    dict_merged = lambda a,b: a.update(b) or a
    return dict_merged(d1, d2)

def intersection(lst1, lst2):

    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


##### Defining worker Functions

def pos_count_worker(select_corpus_list_vocab):
    '''
    Builds the count for the event corpus and vocab list (which is different for positive and negative)
    '''

    select_corpus_list, keys, vocab_list = select_corpus_list_vocab

    two_word_cts_dict = {}
    two_word_all = []
    m = 5
    ii = 0

    for article_select in select_corpus_list:

        key = keys[ii]
        two_word_cts = []
        art_sentences = tokenize.sent_tokenize(article_select)

        for s in art_sentences:

            s_split = analyzer(s)    # split sentence
            s_len = len(s_split)     # count num words

            if (s_len <= 75) and (s_len > 1):

                # find the elements of the sentence that are in the vocab list
                words_in_vocab = intersection(s_split, vocab_list)
                words_traversed = []

                for word1 in words_in_vocab:

                    if word1 not in words_traversed:
                        indices = [i for i, x in enumerate(s_split) if x == word1]

                        for j in indices:

                            words_use = intersection(s_split[j+1:j+min(m, s_len-j)], words_in_vocab)         # only count words in the vocab

                            if (words_use):

                                for word2 in words_use:

                                    if (word1 == word2):
                                        continue

                                    if (word1 < word2):
                                        two_word = word1 + ' ' + word2
                                    else:
                                        two_word = word2 + ' ' + word1

                                    if two_word not in two_word_cts:
                                        two_word_cts.append(two_word)

                    words_traversed.append(word1)

        two_word_cts_dict[key] = set(two_word_cts)
        two_word_all += two_word_cts_dict[key]

        ii += 1

    two_word_all = list(set(two_word_all))

    return two_word_cts_dict, two_word_all


def neg_count_worker(select_corpus_list_vocab):
    '''
    Builds the count for the event corpus and vocab list (which is different for positive and negative)
    '''

    select_corpus_list, keys, vocab_list = select_corpus_list_vocab

    two_word_cts_dict = {}
    two_word_all = []
    m = 5
    ii = 0

    for article_select in select_corpus_list:

        key = keys[ii]
        two_word_cts = []
        art_sentences = tokenize.sent_tokenize(article_select)

        for s in art_sentences:

            s_split = analyzer(s)    # split sentence
            s_len = len(s_split)     # count num words

            if (s_len <= 75) and (s_len > 1):

                # find the elements of the sentence that are in the vocab list
                words_in_vocab = intersection(s_split, vocab_list)
                words_traversed = []

                for word1 in words_in_vocab:

                    if word1 not in words_traversed:
                        indices = [i for i, x in enumerate(s_split) if x == word1]

                        for j in indices:

                            words_use = intersection(s_split[j+1:j+min(m, s_len-j)], words_in_vocab)         # only count words in the vocab

                            if (words_use):

                                for word2 in words_use:

                                    if (word1 == word2):
                                        continue

                                    if (word1 < word2):
                                        two_word = word1 + ' ' + word2
                                    else:
                                        two_word = word2 + ' ' + word1

                                    if two_word not in two_word_cts:
                                        two_word_cts.append(two_word)

                    words_traversed.append(word1)

        two_word_cts_dict[key] = set(two_word_cts)
        two_word_all += two_word_cts_dict[key]

        ii += 1

    two_word_all = list(set(two_word_all))

    return two_word_cts_dict, two_word_all


###### Calling the worker functions

def pos_caller(article_list1, keys, vocab_list, results_dir, cv_period):

    print len(article_list1), len(keys)
    #### breaking into chunks of 100 articles
    ncomb = 1000
    nevents = len(article_list1)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    corpus_100 = [(article_list1[q*ncomb:(q+1)*ncomb], keys[q*ncomb:(q+1)*ncomb], vocab_list) for q in range(nsplit)]

    print len(corpus_100), "pos"

    ### getting the counts for the entire article list
    tstart = time.time()
    pool = mp.Pool(processes = 20)
    results = pool.map(pos_count_worker, corpus_100)   # 6000 articles, 60 results
    print len(results)

    print "%0.2f pos" %(time.time()-tstart)


    ##### combining the pooled results
    tstart = time.time()
    nres = len(results)
    two_word_dict_full = copy.deepcopy(results[0][0])
    two_word_all = results[0][1]
    for k in range(1, nres):

        two_word_dict_full = dict_merge(two_word_dict_full,results[k][0])
        two_word_all += results[k][1]

    two_word_all = list(set(two_word_all))

    # save the two word count as a dict

    print "%0.1f pos - saving" %(time.time()-tstart)

    fname = os.path.join(results_dir, 'poscounts_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(two_word_dict_full, output)
    output.close()

    fname = os.path.join(results_dir, 'poscounts_list_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(two_word_all, output)
    output.close()

    print "%0.1f pos" %(time.time()-tstart)



def neg_caller(article_list1, keys, vocab_list, results_dir, cv_period):

    print len(article_list1), len(keys)
    #### breaking into chunks of 100 articles
    ncomb = 1000
    nevents = len(article_list1)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    corpus_100 = [(article_list1[q*ncomb:(q+1)*ncomb], keys[q*ncomb:(q+1)*ncomb], vocab_list) for q in range(nsplit)]

    print len(corpus_100), "neg"

    ### getting the counts for the entire article list
    tstart = time.time()
    pool = mp.Pool(processes = 20)
    results = pool.map(neg_count_worker, corpus_100)   # 6000 articles, 60 results
    print len(results)

    print "%0.2f neg" %(time.time()-tstart)


    ##### combining the pooled results
    tstart = time.time()
    nres = len(results)
    two_word_dict_full = copy.deepcopy(results[0][0])
    two_word_all = results[0][1]
    for k in range(1, nres):

        two_word_dict_full = dict_merge(two_word_dict_full,results[k][0])
        two_word_all += results[k][1]

    two_word_all = list(set(two_word_all))

    # save the two word count as a dict

    print "%0.1f neg - saving" %(time.time()-tstart)

    fname = os.path.join(results_dir, 'negcounts_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(two_word_dict_full, output)
    output.close()

    fname = os.path.join(results_dir, 'negcounts_list_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(two_word_all, output)
    output.close()

    print "%0.1f neg" %(time.time()-tstart)


# Managing the pos and neg counts as parallel function calls

def main_caller(article_list_pos, article_list_neg, article_list_event, pos_keys, neg_keys,  results_dir, cv_period):

    # create complete article list with these articles and save the pos and neg counts

    tstart_all = time.time()

    vec = CountVectorizer(stop_words=stop_words_list, max_df = 0.8, min_df = 0.01)  #stop_words='english'
    vec.fit(article_list_event)

    vocab_list = vec.vocabulary_.keys()
    Nv = len(vocab_list)
    print Nv, ": no. of vocab words"

    # digit filtering
    vocab_list1 = [item for item in vocab_list if not item.isdigit()]
    vocab_list = vocab_list1
    Nv = len(vocab_list)

    print Nv, ": no. of vocab words"
    del vec, vocab_list1

    print "Initiating Parallel Func. Calls with digit filter"

    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    p = mp.Process(target=pos_caller, args=(article_list_pos, pos_keys, vocab_list, results_dir, cv_period))
    jobs.append(p)
    p.start()

    p = mp.Process(target=neg_caller, args=(article_list_neg, neg_keys, vocab_list, results_dir, cv_period))
    jobs.append(p)
    p.start()

    for proc in jobs:
        proc.join()

    print "In %0.1f secs completed two word count for CV fold %d" %((time.time() - tstart_all), cv_period)

    # save this vocab list for consistency across versions
    fname = os.path.join(results_dir, 'oneW_features_' + str(cv_period) + '.csv')
    vocab_df = pd.DataFrame(vocab_list)
    vocab_df.to_csv(fname)

    return vocab_list
