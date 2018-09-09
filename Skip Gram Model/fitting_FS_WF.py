import sys, os, shutil
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing as mp
import itertools
import argparse
import cPickle as pickle
import logging
import glob
from scipy.stats import spearmanr
from shutil import copyfile
from scipy.stats import norm
import json
import gc

from pybt import modeling
from pybt import trading_mappings
from pybt import data_access as da
from pybt.config_helper import ConfigHelper
from pybt import utils
from pybt import transforms

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import stop_words
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from scipy import stats, sparse
from datetime import datetime
import fitting_ret as fitting

sys.path.append(os.path.expanduser('/home/ryan/anaconda2/lib/python2.7/site-packages/'))
from funcy import project

from nltk import tokenize
import copy
import string
from collections import defaultdict
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import two_word_counts_new as TW_counts
from two_word_counts_new import analyzer


##### General helper functions
def intersection(lst1, lst2):

    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def add_dicts(tests):
    ret = defaultdict(int)
    for d in tests:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


#### Split positive and event negative article lists

def posneg_articles_split(event_corpus_select, returns_df, min_return_thresh):

    print "Separating positive and negative events"
    tstart = time.time()

    min_return = min_return_thresh  #0.0 # min + and - returns to select pos and neg articles

    pos_events = returns_df.loc[returns_df['returns_select'] > min_return]
    neg_events = returns_df.loc[returns_df['returns_select'] <= -min_return]

    pos_event_ids = pos_events['gvkey + event_date'].tolist()
    neg_event_ids = neg_events['gvkey + event_date'].tolist()

    # the limited_vectorizer_counts has all feature counts - so we separate it out to positive and negative
    # compare with passing_ids

    event_keys_select = event_corpus_select.keys()

    pos_event_ids2 = intersection(pos_event_ids, event_keys_select)
    neg_event_ids2 = intersection(neg_event_ids, event_keys_select)

    pos_corpus_list = []
    neg_corpus_list = []
    pos_counts, neg_counts = 0, 0
    pos_keys = []
    neg_keys = []

    for key in event_keys_select:

        if key in pos_event_ids2:
            pos_corpus_list.append(event_corpus_select[key])
            pos_keys.append(key)
            pos_counts += 1

        if key in neg_event_ids2:
            neg_corpus_list.append(event_corpus_select[key])
            neg_keys.append(key)
            neg_counts += 1

    print pos_counts, neg_counts, len(pos_event_ids2), len(neg_event_ids2), len(event_keys_select)
    print len(pos_keys), len(neg_keys)


    return pos_corpus_list, neg_corpus_list, pos_keys, neg_keys


##### Feature Selection functions

def pos_2count_worker(inps):

    featurelist, pos_two_word_cts = inps
    cts = []
    vals = pos_two_word_cts.values()

    for feature in featurelist:
        ct = 0
        for value in vals:
            if feature in value:
                ct += 1
        cts.append(ct)

    return cts

def neg_2count_worker(inps):

    featurelist, neg_two_word_cts = inps
    cts = []
    vals = neg_two_word_cts.values()

    for feature in featurelist:
        ct = 0
        for value in vals:
            if feature in value:
                ct += 1

        cts.append(ct)

    return cts


def pos_counter(pos_two_word_list, pos_two_word_cts, results_dir, cv_period):

    # counting all possible two word combinations
    ncomb = 100000
    nevents = len(pos_two_word_list)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    corpus_100 = [(pos_two_word_list[q*ncomb:(q+1)*ncomb], pos_two_word_cts) for q in range(nsplit)]

    print len(corpus_100), "pos"

    tstart = time.time()
    pool = mp.Pool(processes = 20)
    results = pool.map(pos_2count_worker, corpus_100)   # 6000 articles, 60 results

    print "Finished pos counts for feature selection in", time.time()-tstart

    # combine results together
    nres = len(results)
    tw_counts = []
    for j in range(nres):
        tw_counts += results[j]

    Npos = len(pos_two_word_cts.keys())
    tw_counts2 = [x*1.0/Npos for x in tw_counts]

    dict_res = {}
    dict_res['features'] = pos_two_word_list
    dict_res['counts'] = tw_counts
    dict_res['inv_cdf'] = norm.ppf(tw_counts2)

    # save pos dict
    fname = os.path.join(results_dir, 'pos_2freq_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(dict_res, output)
    output.close()
    # return_dict[0] = copy.deepcopy(dict_res)


def neg_counter(neg_two_word_list, neg_two_word_cts, results_dir, cv_period):

    # counting all possible two word combinations
    ncomb = 100000
    nevents = len(neg_two_word_list)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    corpus_100 = [(neg_two_word_list[q*ncomb:(q+1)*ncomb], neg_two_word_cts) for q in range(nsplit)]

    print len(corpus_100), "neg"

    tstart = time.time()
    pool = mp.Pool(processes = 20)
    results = pool.map(neg_2count_worker, corpus_100)   # 6000 articles, 60 results
    print "Finished neg counts for feature selection in", time.time()-tstart

    # combine results together
    nres = len(results)
    tw_counts = []
    for j in range(nres):
        tw_counts += results[j]

    Nneg = len(neg_two_word_cts.keys())
    tw_counts2 = [x*1.0/Nneg for x in tw_counts]

    dict_res = {}
    dict_res['features'] = neg_two_word_list
    dict_res['counts'] = tw_counts
    dict_res['inv_cdf'] = norm.ppf(tw_counts2)

    # save pos dict
    fname = os.path.join(results_dir, 'neg_2freq_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(dict_res, output)
    output.close()
    # return_dict[1] = copy.deepcopy(dict_res)

acceptable_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
def tag_separator(article):

    inp_tokens = word_tokenize(article)

    tokens_pos = pos_tag(inp_tokens)
    N = len(inp_tokens)
    tks = []

    for i in range(0, N):

        tks.append(tokens_pos[i][1])

    return tks

def pos_filter_worker(pos_2word_list):
    pos_tw_adj_cts = []
    for word_pair in pos_2word_list:

        tags = tag_separator(word_pair)
        tags_use = False

        for x in tags:
            if x in acceptable_tags:
                tags_use = True
                break

        if tags_use:
            pos_tw_adj_cts.append(word_pair)

    return pos_tw_adj_cts

def pos_filter(pos_2word_list, results_dir, cv_period):

    ncomb = 100000
    nevents = len(pos_2word_list)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    corpus_100 = [pos_2word_list[q*ncomb:(q+1)*ncomb] for q in range(nsplit)]

    tstart = time.time()
    pool = mp.Pool(processes = 20)
    results = pool.map(pos_filter_worker, corpus_100)   # 6000 articles, 60 results
    print "Finished Pos filtering in", time.time()-tstart

    nres = len(results)
    tw_filter = []
    for j in range(nres):
        tw_filter += results[j]

    fname = os.path.join(results_dir, 'posfilter_list_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(tw_filter, output)
    output.close()

def neg_filter_worker(neg_2word_list):
    neg_tw_adj_cts = []
    for word_pair in neg_2word_list:

        tags = tag_separator(word_pair)
        tags_use = False

        for x in tags:
            if x in acceptable_tags:
                tags_use = True
                break

        if tags_use:
            neg_tw_adj_cts.append(word_pair)

    return neg_tw_adj_cts

def neg_filter(neg_2word_list, results_dir, cv_period):

    ncomb = 100000
    nevents = len(neg_2word_list)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    corpus_100 = [neg_2word_list[q*ncomb:(q+1)*ncomb] for q in range(nsplit)]

    tstart = time.time()
    pool = mp.Pool(processes = 20)
    results = pool.map(neg_filter_worker, corpus_100)   # 6000 articles, 60 results
    print "Finished neg filtering in", time.time()-tstart

    nres = len(results)
    tw_filter = []
    for j in range(nres):
        tw_filter += results[j]

    fname = os.path.join(results_dir, 'negfilter_list_' + str(cv_period) + '.pkl')
    output = open(fname, 'wb')
    pickle.dump(tw_filter, output)
    output.close()

def feature_select_ranking(results_dir, i, Npos, Nneg, pos_two_word_res, neg_two_word_res, nfeatures):

    # convert dict to df
    df_pos = pd.DataFrame.from_dict(pos_two_word_res)
    df_neg = pd.DataFrame.from_dict(pos_two_word_res)

    # merge both df
    df_merged = pd.merge(df_pos, df_neg, how='outer', on=['features'],
                       sort=False,                                         # left_index=True,
                       suffixes=('_pos', '_neg'), copy=True, indicator=False)

    # ranking on KLdiv
    KLdiv_select = df_merged
    KLdiv_select = KLdiv_select.drop(['inv_cdf_pos', 'inv_cdf_neg'], axis=1)
    fill_values = {'counts_pos': 1.0, 'counts_neg': 1.0}
    KLdiv_select = KLdiv_select.fillna(value = fill_values)

    # filtering out less than 1%
    KLdiv_select['counts_total'] = KLdiv_select['counts_pos'] + KLdiv_select['counts_neg']
    min_counts_pos = int(round(0.01*Npos))
    min_counts_neg = int(round(0.01*Nneg))
    min_counts_total = min_counts_pos + min_counts_neg
    KLdiv_select = KLdiv_select[KLdiv_select['counts_total'] > min_counts_total]

    # calculating the score
    KLdiv_select['A'] =  1.0*KLdiv_select['counts_pos']/Npos
    KLdiv_select['B'] =  1.0*KLdiv_select['counts_neg']/Nneg

    KLdiv_select['KLdiv1'] = KLdiv_select['A']*np.log(1.0*KLdiv_select['A']/KLdiv_select['B'])*1e5
    KLdiv_select['KLdiv0'] = KLdiv_select['B']*np.log(1.0*KLdiv_select['B']/KLdiv_select['A'])*1e5
    KLdiv_select['KLdiv'] = KLdiv_select['KLdiv1'] + KLdiv_select['KLdiv0']

    KLdiv_select = KLdiv_select.sort_values(by=['KLdiv'], ascending=False)
    KLdiv_select['rank'] = np.arange(1,len(KLdiv_select)+1)

    # second ranking chi2
    chi_select = df_merged

    chi_select = chi_select.drop(['inv_cdf_pos', 'inv_cdf_neg'], axis=1)
    fill_values = {'counts_pos': 0, 'inv_cdf_pos': 0 , 'counts_neg': 0, 'inv_cdf_neg': 0}
    chi_select = chi_select.fillna(value = fill_values)

    # filtering out less than 1%
    chi_select['counts_total'] = chi_select['counts_pos'] + chi_select['counts_neg']
    min_counts_pos = int(round(0.01*Npos))
    min_counts_neg = int(round(0.01*Nneg))
    min_counts_total = min_counts_pos + min_counts_neg
    chi_select = chi_select[chi_select['counts_total'] > min_counts_total]

    # calculating individual terms for chi squared
    chi_select['counts_notin_pos'] = Npos - chi_select['counts_pos']
    chi_select['counts_notin_neg'] = Nneg - chi_select['counts_neg']
    chi_select['counts_notin_total'] = Npos + Nneg - chi_select['counts_total']

    t1 = np.square(1.0*(chi_select['counts_pos'] - chi_select['counts_total'])/chi_select['counts_total'])
    t2 = np.square(1.0*(chi_select['counts_neg'] - chi_select['counts_total'])/chi_select['counts_total'])
    t3 = np.square(1.0*(chi_select['counts_notin_pos'] - chi_select['counts_notin_total'])/chi_select['counts_notin_total'])
    t4 = np.square(1.0*(chi_select['counts_notin_neg'] - chi_select['counts_notin_total'])/chi_select['counts_notin_total'])
    chi_select['chi_squared'] = t1 + t2 + t3 + t4

    chi_select = chi_select.sort_values(by=['chi_squared'], ascending=False)
    chi_select['rank'] = np.arange(1,len(chi_select)+1)

    df_ranks = pd.merge(KLdiv_select, chi_select, how='inner', on=['features'],
                   sort=False,                                         # left_index=True,
                   suffixes=('_kldiv', '_chi'), copy=True, indicator=False)

    df_ranks['avg_rank'] = 0.5*(df_ranks['rank_kldiv'] + df_ranks['rank_chi'])
    df_ranks = df_ranks.sort_values(by = 'avg_rank', ascending=True)

    print len(df_ranks), ": No of features remaining"

    if (len(df_ranks) < nfeatures):
        BNS_select = df_ranks
        BNS_select_features = BNS_select['features']
    else:
        BNS_select = df_ranks.iloc[0:nfeatures]
        BNS_select_features = BNS_select['features']
    print len(BNS_select_features), ": No of features selected"

    return BNS_select, BNS_select_features


def BNS_feature_select(results_dir, i, Npos, Nneg):

    print Npos, ": No. of pos events"
    print Nneg, ": No. of neg events"
    nfeatures = 4000

    tstart = time.time()
    fname = os.path.join(results_dir, 'poscounts_' + str(i) + '.pkl')
    with open(fname, 'rb') as ofp:
        pos_two_word_cts = pickle.load(ofp)

    fname = os.path.join(results_dir, 'poscounts_list_' + str(i) + '.pkl')
    with open(fname, 'rb') as ofp:
        pos_two_word_list = pickle.load(ofp)

    print "Pos Pickle Loaded in", time.time()-tstart
    print "Pos 2-Word list len", len(pos_two_word_list)

    tstart = time.time()
    fname = os.path.join(results_dir, 'negcounts_' + str(i) + '.pkl')
    with open(fname, 'rb') as ofp:
        neg_two_word_cts = pickle.load(ofp)

    fname = os.path.join(results_dir, 'negcounts_list_' + str(i) + '.pkl')
    with open(fname, 'rb') as ofp:
        neg_two_word_list = pickle.load(ofp)

    print "Neg Pickle Loaded in ", time.time()-tstart
    print "Neg 2-Word list len", len(neg_two_word_list)

    # #### add the adjective filtering here - so that only those are counted
    # manager = mp.Manager()
    # jobs = []
    #
    # p = mp.Process(target=pos_filter, args=(pos_two_word_list, results_dir, i))
    # jobs.append(p)
    # p.start()
    #
    # p = mp.Process(target=neg_filter, args=(neg_two_word_list, results_dir, i))
    # jobs.append(p)
    # p.start()
    #
    # for proc in jobs:
    #     proc.join()
    #
    # fname = os.path.join(results_dir, 'posfilter_list_' + str(i) + '.pkl')
    # with open(fname, 'rb') as ofp:
    #     pos_two_word_list = pickle.load(ofp)
    #
    # fname = os.path.join(results_dir, 'negfilter_list_' + str(i) + '.pkl')
    # with open(fname, 'rb') as ofp:
    #     neg_two_word_list = pickle.load(ofp)
    #
    # print "Pos 2-Word list len w/ adj filter", len(pos_two_word_list)
    # print "Neg 2-Word list len w/ adj filter", len(neg_two_word_list)

    # gather all the counts in a df to calculate BNS score
    manager = mp.Manager()
    jobs = []

    p = mp.Process(target=pos_counter, args=(pos_two_word_list, pos_two_word_cts, results_dir, i))
    jobs.append(p)
    p.start()

    p = mp.Process(target=neg_counter, args=(neg_two_word_list, neg_two_word_cts, results_dir, i))
    jobs.append(p)
    p.start()

    for proc in jobs:
        proc.join()

    fname = os.path.join(results_dir, 'pos_2freq_' + str(i) + '.pkl')
    with open(fname, 'rb') as ofp:
        pos_two_word_res = pickle.load(ofp)

    fname = os.path.join(results_dir, 'neg_2freq_' + str(i) + '.pkl')
    with open(fname, 'rb') as ofp:
        neg_two_word_res = pickle.load(ofp)

    # results_dir2 = '/research/home/rakesh/results/WalkForward/new_WF/TW_FS_chi'
    # fname = os.path.join(results_dir2, 'pos_2freq_' + str(i) + '.pkl')
    # with open(fname, 'rb') as ofp:
    #     pos_two_word_res = pickle.load(ofp)
    #
    # print "Loaded", fname
    #
    # fname = os.path.join(results_dir2, 'neg_2freq_' + str(i) + '.pkl')
    # with open(fname, 'rb') as ofp:
    #     neg_two_word_res = pickle.load(ofp)
    #
    # print "Loaded", fname

    # convert dict to df
    df_pos = pd.DataFrame.from_dict(pos_two_word_res)
    df_neg = pd.DataFrame.from_dict(neg_two_word_res)

    # merge both df
    df_merged = pd.merge(df_pos, df_neg, how='outer', on=['features'],
                       sort=False,                                         # left_index=True,
                       suffixes=('_pos', '_neg'), copy=True, indicator=False)

    # ###### Top N words - no feature selection
    # # this is not the same as top N counts though! words that are most frequent
    # df_merged = df_merged.drop(['inv_cdf_pos', 'inv_cdf_neg'], axis=1)
    # fill_values = {'counts_pos': 0.0, 'counts_neg': 0.0}
    # df_merged = df_merged.fillna(value = fill_values)
    #
    # df_merged['counts_total'] = df_merged['counts_pos'] + df_merged['counts_neg']
    # # no need for 1% filtering since only the most likely are selected
    #
    # df_merged = df_merged.sort_values(by=['counts_total'], ascending=False)
    # print len(df_merged), ": No of features remaining"
    #
    # if (len(df_merged) < nfeatures):
    #     BNS_select = df_merged
    #     BNS_select_features = BNS_select['features']
    # else:
    #     BNS_select = df_merged.iloc[0:nfeatures]
    #     BNS_select_features = BNS_select['features']
    # print len(BNS_select_features), ": No of features selected"

    # ####### KLdiv approach
    # df_merged = df_merged.drop(['inv_cdf_pos', 'inv_cdf_neg'], axis=1)
    # fill_values = {'counts_pos': 1.0, 'counts_neg': 1.0}
    # df_merged = df_merged.fillna(value = fill_values)
    #
    # # filtering out less than 1%
    # df_merged['counts_total'] = df_merged['counts_pos'] + df_merged['counts_neg']
    # min_counts_pos = int(round(0.01*Npos))
    # min_counts_neg = int(round(0.01*Nneg))
    # min_counts_total = min_counts_pos + min_counts_neg
    # df_merged = df_merged[df_merged['counts_total'] > min_counts_total]
    #
    # # calculating the score
    # df_merged['A'] =  df_merged['counts_pos']/Npos
    # df_merged['B'] =  df_merged['counts_neg']/Nneg
    #
    # df_merged['KLdiv1'] = df_merged['A']*np.log(df_merged['A']/df_merged['B'])*1e5
    # df_merged['KLdiv0'] = df_merged['B']*np.log(df_merged['B']/df_merged['A'])*1e5
    # df_merged['KLdiv'] = df_merged['KLdiv1'] + df_merged['KLdiv0']
    #
    # df_merged = df_merged.sort_values(by=['KLdiv'], ascending=False)
    # print len(df_merged), ": No of features remaining"
    #
    # if (len(df_merged) < nfeatures):
    #     BNS_select = df_merged
    #     BNS_select_features = BNS_select['features']
    # else:
    #     BNS_select = df_merged.iloc[0:nfeatures]
    #     BNS_select_features = BNS_select['features']
    # print len(BNS_select_features), ": No of features selected"


    ######## chi squared approach - with top end filter
    df_merged = df_merged.drop(['inv_cdf_pos', 'inv_cdf_neg'], axis=1)
    fill_values = {'counts_pos': 0, 'inv_cdf_pos': 0 , 'counts_neg': 0, 'inv_cdf_neg': 0}
    df_merged = df_merged.fillna(value = fill_values)

    # filtering out less than 1%
    df_merged['counts_total'] = df_merged['counts_pos'] + df_merged['counts_neg']
    min_counts_pos = int(round(0.01*Npos))
    min_counts_neg = int(round(0.01*Nneg))
    min_counts_total = min_counts_pos + min_counts_neg
    df_merged = df_merged[df_merged['counts_total'] > min_counts_total]

    # # filtering out greater than 2.25%
    # max_counts_total = int(round(0.0225*(Npos+Nneg)))
    # df_merged = df_merged[df_merged['counts_total'] <= max_counts_total]
    # above this count remove features - as they create a lot of noise
    # another way to do this is after 4K features are selected

    # calculating individual terms for chi squared
    df_merged['counts_notin_pos'] = Npos - df_merged['counts_pos']
    df_merged['counts_notin_neg'] = Nneg - df_merged['counts_neg']
    df_merged['counts_notin_total'] = Npos + Nneg - df_merged['counts_total']

    t1 = np.square((df_merged['counts_pos'] - df_merged['counts_total'])/df_merged['counts_total'])
    t2 = np.square((df_merged['counts_neg'] - df_merged['counts_total'])/df_merged['counts_total'])
    t3 = np.square((df_merged['counts_notin_pos'] - df_merged['counts_notin_total'])/df_merged['counts_notin_total'])
    t4 = np.square((df_merged['counts_notin_neg'] - df_merged['counts_notin_total'])/df_merged['counts_notin_total'])
    df_merged['chi_squared'] = t1 + t2 + t3 + t4

    df_merged = df_merged.sort_values(by=['chi_squared'], ascending=False)
    print len(df_merged), ": No of features remaining"

    nfeatures = 4000
    if (len(df_merged) < nfeatures):
        BNS_select = df_merged
        # filtering out greater than 2.25%
        # max_counts_total = int(round(0.0225*(Npos+Nneg)))
        # BNS_select = BNS_select[BNS_select['counts_total'] <= max_counts_total]
        BNS_select_features = BNS_select['features']
    else:
        BNS_select = df_merged.iloc[0:nfeatures]
        # max_counts_total = int(round(0.0225*(Npos+Nneg)))
        # BNS_select = BNS_select[BNS_select['counts_total'] <= max_counts_total]
        BNS_select_features = BNS_select['features']
    print len(BNS_select_features), ": No of features selected"


    # ###### BNS new approach
    # # fill nan values
    # # Npos = len(pos_two_word_cts.keys())    # this is the # of positive events
    # pos0_fill_ratio = 1.0/Npos
    # pos0_fill_invcdf = norm.ppf(pos0_fill_ratio)
    #
    # # Nneg = len(neg_two_word_cts.keys())
    # neg0_fill_ratio = 1.0/Nneg
    # neg0_fill_invcdf = norm.ppf(neg0_fill_ratio)
    #
    # fill_values = {'counts_pos': 0, 'inv_cdf_pos': pos0_fill_invcdf , 'counts_neg': 0, 'inv_cdf_neg': neg0_fill_invcdf}
    # df_merged = df_merged.fillna(value = fill_values)
    #
    # # remove features which are not common in pos and neg
    # min_counts_pos = int(round(0.01*Npos))
    # min_counts_neg = int(round(0.01*Nneg))
    # min_counts_total = min_counts_pos + min_counts_neg
    #
    # df_merged['counts_total'] = df_merged['counts_pos'] + df_merged['counts_neg']
    # df_merged = df_merged[df_merged['counts_total'] > min_counts_total]
    #
    # # calc BNS score
    # df_merged['BNS_scores'] = df_merged['inv_cdf_pos'] - df_merged['inv_cdf_neg']
    # df_merged = df_merged.sort_values(by=['BNS_scores'])
    # print len(df_merged), ": No of features remaining"
    #
    # # Selection based on BNS
    # nfeatures = 4000
    # nP = int(nfeatures*0.25)
    # nN = int(nfeatures*0.75)
    # if (len(df_merged) < nfeatures):
    #     BNS_select = df_merged
    #     BNS_select_features = BNS_select['features']
    # else:
    #     BNS_select = pd.concat((df_merged.iloc[0:nP], df_merged.iloc[-nN:]), axis=0)
    #     BNS_select_features = BNS_select['features']
    #
    # print len(BNS_select_features), ": No of features selected"

    # #### Average ranked features
    #
    # BNS_select, BNS_select_features = feature_select_ranking(results_dir, i, Npos, Nneg, pos_two_word_res, neg_two_word_res, nfeatures)

    ### save and return
    fname = os.path.join(results_dir, 'features_' + str(i) + '.csv')
    BNS_select.to_csv(fname)
    #
    # ### to load saved features
    # fname = os.path.join(results_dir, 'features_' + str(i) + '.csv')
    # BNS_select = pd.read_csv(fname)
    # BNS_select_features = BNS_select['features']

    return BNS_select_features.values.tolist()

# counting functions that feed the tfidf
def counts_worker(article, vocab_features):
    '''
    Builds the count for the event corpus and vocab list (which is different for positive and negative)
    '''

    # going through the articles and counting 2 word occurences

    counts = {}
    m = 5

    art_sentences = tokenize.sent_tokenize(article)

    for s in art_sentences:

        s_split = analyzer(s)    # split sentence
        s_len = len(s_split)     # count num words

        if (s_len <= 75) and (s_len > 1):

            # find the elements of the sentence that are in the vocab list
            words_in_vocab = intersection(s_split, vocab_features)
            words_traversed = []    # this is to not repeat words once they're accounted for

            for word in words_in_vocab:

                if word not in words_traversed:
                    indices = [i for i, x in enumerate(s_split) if x == word]

                    for j in indices:
                        words_use = intersection(s_split[j+1:j+min(m, s_len-j)], words_in_vocab)         # only count words in the vocab
                        if (words_use):                                  # create 4 word count dict
                            w_counts = dict.fromkeys(words_use, 1)
                            if word in counts.keys():
                                counts[word] = add_dicts([counts[word], w_counts])          # if the word already exists - add the dicts
                            else:
                                counts[word] = w_counts

                words_traversed.append(word)

    return counts


def two_word_counter(counts, two_word_features):

    two_word_cts = {}
    i = 0
    j = 0
    word1s = counts.keys()
    N1 = len(word1s)

    for word1 in word1s:

        word2s = counts[word1].keys()

        for word2 in word2s:

            if (word1 == word2):
                continue

            if (word1 < word2):
                two_word_key = word1 + ' ' + word2
            else:
                two_word_key = word2 + ' ' + word1

            if (two_word_key in two_word_features):

                if two_word_key in two_word_cts:
                    two_word_cts[two_word_key] += counts[word1][word2]
                else:
                    two_word_cts[two_word_key] = counts[word1][word2]

            else:
                continue


    return two_word_cts


def feature_counter(argins):

    article_list1, event_num, two_word_features, vocab_features, full_vocab_list = argins

    row = []
    col = []
    data = []
    fcounts = []
    aid = 0

    for a in article_list1:

        # call and get two_word_counts for an article - directly
        cts = counts_worker(a, vocab_features)
        two_word_cts = two_word_counter(cts, two_word_features)

        # go through the two_word_cts keys and check those that are in features only
        two_words = two_word_cts.keys()
        for xyz in two_words:
            col.append(two_word_features.index(xyz))
            row.append(event_num[aid])
            data.append(two_word_cts[xyz])

        # is it faster to do this as iterables??

        aid += 1

    return row, col, data


def tfidf_transform_gen(event_keys_use, event_corpus_list, features_select, full_vocab_list):

    tstart = time.time()

    # getting the unique list of words in the two word features
    vec = CountVectorizer()
    vec.fit(features_select)

    vocab_features = vec.vocabulary_.keys()
    Nf2 = len(vocab_features)
    print Nf2, ": no. of unique words in features"
    del vec

    print features_select[:5]

    feature_names = ['feature_' + x for x in features_select]
    print "Total no. of features (vocab words) considered are", len(feature_names)

    transformer_dict = {}
    tfidf_df2 = pd.DataFrame([])
    for ig in IG:

        keys_select = eventkeys_to_gicsIG[eventkeys_to_gicsIG['gics_industry_group'] == ig]['gvkey + event_date']

        event_ig_keys = intersection(keys_select.tolist(), event_keys_use)
        event_corpus_select_list = []
        for key in event_ig_keys:
            event_corpus_select_list.append(event_corpus[key])

        # vect_counts = twice_limited_vectorizer.transform(event_corpus_select_list)
        # Create a count vectorizer equivalent call
        # splitting the data into chunks
        nevents = len(event_corpus_select_list)
        event_num_list = range(nevents)
        ncomb = 40                        # what is the optimal split
        nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
        corpus_split = []

        inps = [(event_corpus_select_list[q*ncomb:(q+1)*ncomb], event_num_list[q*ncomb:(q+1)*ncomb], features_select, vocab_features, full_vocab_list) for q in range(nsplit)]
        print len(inps), ": chunks of events for parallel counting"

        # parallelizing the count calls
        pool = mp.Pool(processes = 30)   #, maxtasksperchild=1
        comb1 = pool.map(feature_counter, inps)
        pool.close()
        pool.join()
        pool.terminate()

        # going through the combined results to get sparse matrix representation
        r, c, d = [], [], []
        for q in range(nsplit):

            r += comb1[q][0]
            c += comb1[q][1]
            d += comb1[q][2]

        vectorizer_tw_counts = sparse.csr_matrix((d, (r, c)), shape=(len(event_num_list), len(features_select)))


        ### tfidf transformer
        transformer = TfidfTransformer(smooth_idf=True)
        tfidf_corpus = transformer.fit_transform(vectorizer_tw_counts)
        transformer_dict[ig] = transformer

        tfidf_df_select = pd.DataFrame(index = event_ig_keys, columns = feature_names, data = tfidf_corpus.toarray())
        tfidf_df2 = tfidf_df2.append(tfidf_df_select)

        del vectorizer_tw_counts, inps

    print len(tfidf_df_select), ": chunks of events for parallel counting"
    print "%0.2f count vec created" %(time.time()-tstart)

    vectorizer_tw_counts = []
    print len(tfidf_df2), ": all events used to build tfidf scores"

    print "Tfidf transformer built in %0.2f secs" %(time.time() - tstart)

    return vectorizer_tw_counts, transformer_dict, tfidf_df2


##### Functions to build BoW scores
def tfidf_transform_calc(argins):

    key, transformer_dict, event_art_num, two_word_features, full_vocab_list, feature_names = argins

    # getting the unique list of words in the two word features - this is quick so repeated
    vec = CountVectorizer()
    vec.fit(two_word_features)

    vocab_features = vec.vocabulary_.keys()
    Nf2 = len(vocab_features)
    del vec

    # Each worker is generating the BoW df which is the tfidf score for the features in the vocab
    # this is being run for all articles in the event / articles selected

    gvkey = key['gvkey']
    date = key['event_date']
    date_str = str(date)
    event_key = gvkey + ' ' + date_str
    # one_big_article = event_corpus[gvkey + ' ' + date_str]
    if event_key in event_keys:
        one_big_article = event_corpus[gvkey + ' ' + date_str]
    else:
        one_big_article = []

    if (len(one_big_article) == 0) or (event_key not in event_keys_intIG):
        bow = pd.Series(index = feature_names, data = np.nan)
        bow.at['word_count'] = 0
        bow.at['num_articles'] = 0
    else:
        if (event_key in tfidf_index_list):
            bow = pd.Series(index = tfidf_df.columns, data = tfidf_df.ix[event_key].values)
            bow.at['word_count'] = 0    # assuming word count is not used elsewhere
        else:
            r1, c1, d1 = feature_counter(([one_big_article], [0], two_word_features, vocab_features, full_vocab_list))   # event num shouldnt matter in this call...
            event_corpus2 = sparse.csr_matrix((d1, (r1, c1)), shape=(1, len(two_word_features)))

            # find the transformer associated w/ the event key
            ig = eventkeys_to_gicsIG[eventkeys_to_gicsIG['gvkey + event_date'] == event_key]['gics_industry_group'].values[0]
            transformer = transformer_dict[ig]

            event_tfidf2 = transformer.transform(event_corpus2)                        # calling the tfidf transformer that was fit with limited corpus
            bow = pd.Series(index = feature_names, data = event_tfidf2.toarray()[0])  # tfidf values for that event based on the limited vocab
            bow.at['word_count'] = event_corpus2.toarray().sum()

        bow.at['num_articles'] = event_art_num[gvkey + ' ' + date_str]


    bow.at['gvkey'] = gvkey
    bow.at['date'] = date

    return bow

##### Loading returns file and config objects
def load_returns_file(fname):

    ret_data = pd.read_csv(fname, index_col = 0, parse_dates = ['date'])
    ret_data['date'] = pd.to_datetime(ret_data['date'].values).date

    return ret_data

def config_set(features):

    config_dir = '/home/ryan/Dev/research-analysis/rgreen/dai_models/earnings/configs/post2-sentiment'

    # open this file and write the features each time
    cfile = os.path.join(config_dir, 'amer-post2-lincv-embeddings-2QScv-N10-v2.ini')
    config = ConfigHelper.from_file(cfile)

    # Set features
    for feature in features:
        config.set_val('indicators', feature, 'Z')

    # We only look at events with greater than 4 articles (arbitrarily chosen).
    config.set_val('run_params', 'filters', 'num_articles;>;0')

    # This tells the framework to change the problem to a classification problem

    config.set_val('run_params', 'objective', 'binary')
    config.set_val('run_params', 'binary_threshold', '0.5')
    config.set_val('run_params', 'binary_threshold_type', 'abs')

    # config.set_val('run_params', 'clip_rets', '3.5')   # clip norm returns in case

    config.config_dict['run_params'].pop('train_ret_group_transform')

    config.set_val('run_params', 'ycol_train', 'fwd_xmkt_projnorm_sec_0_1')   # fwd_xmkt_projnorm_sec_1_10  'fwd_xmkt_1_10' ret_demean_sec_1
    config.set_val('run_params', 'ycol_eval', 'fwd_xmkt_0_1')

    # config.set_val('run_params', 'ycol_train', 'fwd_xmkt_1_10')   #fwd_xmkt_projnorm_sec_1_10

    # This tells the model framework what model class to use.  I have a bunch of predefined ones
    # coded in but can also tell it to look at models defined elsewhere (such as in this notebook).

    config.set_val('model', 'external_module', '__main__')
    config.set_val('model', 'class_name', 'NaiveBayesModel')
    # config.set_val('model', 'class_name', 'SVCModel')

    return config


##### Main function

def run_cv_features(event_info, mapping_select, results_dir, cv, return_models=False):
    # the dao object is created internally

    global event_corpus
    event_corpus, event_dates, event_art_num = event_info

    assert(os.path.isdir(results_dir))

    # Loading the CV periods here
    fname = "/research/home/rakesh/MyCode/WF_periods" + ".csv"   #  str(cv) +
    cv_periods = pd.read_csv(fname, index_col=0, header=0, parse_dates=True)

    # print cv_periods

    # Loading the returns that will be used across all CV periods
    tf = '/home/ryan/Dev/research-analysis/rgreen/amer-indicator-sets/packages/amer_v1/earnings/post-earnings/post1_earnings.csv'
    ret_data = load_returns_file(tf)

    global event_keys
    event_keys = event_corpus.keys()   # indexed event keys referring to the entire event-article corpus

    model_results = []    # this is the combination of all CV models results and will be returned

    global tfidf_df
    global tfidf_index_list

    global eventkeys_to_gicsIG
    global IG
    global event_keys_intIG

    # create the event key to gics map here - make it global
    gvkey_to_gicsIG = ret_data[['gvkey', 'gics_industry_group']]
    gvkey_to_gicsIG = gvkey_to_gicsIG.drop_duplicates()
    gvkey_to_eventID = event_dates[['gvkey', 'gvkey + event_date']]
    gvkey_to_eventID = gvkey_to_eventID.drop_duplicates()

    eventkeys_to_gicsIG = pd.merge(gvkey_to_gicsIG, gvkey_to_eventID, on='gvkey', how='inner')
    eventkeys_to_gicsIG = eventkeys_to_gicsIG.drop_duplicates('gvkey + event_date')
    IG = eventkeys_to_gicsIG['gics_industry_group'].unique()   # list of industry group codes

    event_keys_intIG = intersection(event_keys, eventkeys_to_gicsIG['gvkey + event_date'].tolist())
    event_keys_intIG = set(event_keys_intIG)

    i = cv
    print cv_periods.at[i, 'train1-start']
    print cv_periods.at[i, 'train1-end']

    tloop = time.time()
    ##### Step 1: read event corpus dict and index according to date - to be able to split for cv purposes
    # selecting rows in event_dates ['dates'] column

    mask = (event_dates['date'] > cv_periods.at[i, 'train1-start']) & (event_dates['date'] < cv_periods.at[i, 'train1-end'])
    # mask = (event_dates['date'] > cv_periods.at[i, 'cv-start']) & (event_dates['date'] < cv_periods.at[i, 'cv-end'])
    # mask = (event_dates['date'] > cv_periods.at[i, 'train1-start']) & (event_dates['date'] < cv_periods.at[i, 'cv-end'])
    event_dates_select = event_dates.loc[mask]
    event_keys_select = event_dates_select['gvkey + event_date'].tolist()  # event keys for the desired time frame

    # determine the event keys subset by comparing to event_keys
    event_keys_use = intersection(event_keys_select, event_keys)         # event_keys_use = event_keys_use[:10]
    event_corpus_use = project(event_corpus, event_keys_use)

    print "Event keys selected"

    event_corpus_use_list = []
    for ring in event_keys_use:
        event_corpus_use_list.append(event_corpus_use[ring])
    # event_corpus_use_list = event_corpus_use_list[:10]   # use only info from 10 events - as an example

    ##### Step 2: call feature_select_cv(event corpus (not in this cv period) and associated returns) to obtain the features
    event_keys_use_df = pd.DataFrame(event_keys_use, columns = ['gvkey + event_date'])
    returns_cv_df = pd.merge(event_keys_use_df, event_dates, how='inner', on=['gvkey + event_date'],
                           sort=False,                                         # left_index=True,
                           suffixes=('_x', '_y'), copy=True, indicator=False)

    ##### Create Positive and Negative article lists
    returns_thresh = 0.0
    pos_corpus_use_list, neg_corpus_use_list, pos_keys, neg_keys = posneg_articles_split(event_corpus_use, returns_cv_df, returns_thresh)

    # # split the pos and negative event list and send it to two_word_count here to save the two word count for this particular CV period
    print "\nRunning Two Word Counter for CV fold", i
    full_vocab_list = TW_counts.main_caller(pos_corpus_use_list, neg_corpus_use_list, event_corpus_use_list, pos_keys, neg_keys, results_dir, i)

    # run the BNS feature selection function here - this could be in complete_cv_two itself
    Npos = len(pos_corpus_use_list)
    Nneg = len(neg_corpus_use_list)
    two_word_features = BNS_feature_select(results_dir, i, Npos, Nneg)

    feature_names = ['feature_' + x for x in two_word_features]
    print "Total no. of features (vocab words) considered are", len(feature_names)

    #### Step 3 build tfidf on event corpus (not in this cv period) w/ above features
    print "\nGenerating CV and Tfidf transformer"
    # count_vectorizer, tfidf_transformer, tfidf_df = tfidf_transform_gen(event_keys_use, event_corpus_use_list, two_word_features, full_vocab_list)
    count_vectorizer, tfidf_transformer_dict, tfidf_df = tfidf_transform_gen(event_keys_use, event_corpus_use_list, two_word_features, full_vocab_list)
    tfidf_index_list = tfidf_df.index.tolist()
    print len(tfidf_index_list), len(set(tfidf_index_list))
    del event_corpus_use, event_corpus_use_list

    ##### Step 4 score the (event corpus - complete set) and Step 5 is to generate the dao object - which can be fed below
    # Up to here only the events from the other CV folds are used - now w/ the above features the scores for all articles are calculated

    # mask = (mapping_select['date'] >= cv_periods.at[i, 'train1-start']) & (mapping_select['date'] <= cv_periods.at[i, 'train1-end'])
    mask = (mapping_select['date'] <= cv_periods.at[i, 'cv-end'])
    temp = mapping_select.loc[mask]
    # temp = mapping_select
    temp = temp.drop_duplicates(subset = ['gvkey', 'event_date'])
    keys = temp.to_dict(orient='record')    # this is the df converted to a list! but why?

    # calculating tfidf scores for each event
    tstart = time.time()
    print "\nGenerating BoW scores for NB input"
    pool = mp.Pool(processes = 20)   # if the whole loop gets parallelized - I might not want to sub parallelize...
    # inps = [(x, tfidf_transformer, event_art_num, two_word_features, full_vocab_list, feature_names) for x in keys]
    inps = [(x, tfidf_transformer_dict, event_art_num, two_word_features, full_vocab_list, feature_names) for x in keys]
    results = pool.map(tfidf_transform_calc, inps)
    bow_df = pd.DataFrame(results)
    bow_df['date'] = pd.to_datetime(bow_df['date'].values).date

    bow_df = bow_df.dropna(axis=0, how='any')

    print "Tfidf features fit in %0.1f secs" %(time.time() - tstart)

    ##### Step 5: Generating the dao object for this cv period - which will be used only once
    # so there will be a different dao object for each cv period - whereas there was one dao in the approach before

    # combine w/ returns to form dao object
    print "Merging with Returns File (Loaded above)"
    tstart = time.time()

    # Merge the BoW df with the returns using gvkey and date
    bow_df_merged = bow_df.merge(ret_data, on = ['gvkey', 'date'], how = 'left')

    print "Loaded and merged returns file in %0.1f secs" % (time.time() - tstart)

    ## saving the combined returns and bow df - to run from here if needed
    outname = os.path.join(results_dir, 'bow_tfidf_returns_' + str(i) + '.csv')
    bow_df_merged.to_csv(outname)

    del bow_df_merged, bow_df


    # # workin with new labels but old bow_df_merged
    # new_labels = ['gvkey', 'date', 'fwd_xmkt_projnorm_sec_0_10', 'fwd_xmkt_projnorm_sec_0_1', 'fwd_xmkt_0_10', 'fwd_xmkt_0_1']
    # tf = os.path.join(results_dir, 'bow_tfidf_returns_' + str(i) + '.csv')
    # bow_df_merged = pd.read_csv(tf, index_col=0)
    # bow_df_merged['date'] = pd.to_datetime(bow_df_merged['date'].values).date
    #
    # bow_df_merged = bow_df_merged.drop(new_labels[2:], axis=1)
    #
    # bow_df_merged2 = bow_df_merged.merge(ret_data[new_labels], on = ['gvkey', 'date'], how = 'left')
    # bow_df_merged2.to_csv(tf)
    # print bow_df_merged2.shape

    # ##### normal operation from here

    ## loading the saved bow and return df
    tf = os.path.join(results_dir, 'bow_tfidf_returns_' + str(i) + '.csv')
    dao = da.TrainingDataAccess(tf)

    ## This dao object is ready - need to add features to the config

    features = []
    for c in feature_names:
        if 'feature_' in c:
            features.append(c)

    print "Total no. of features (vocab words) considered are", len(features)

    config = config_set(features)
    print config.config_dict['run_params']['ycol_train']
    print config.config_dict['run_params']['ycol_eval']
    print config.config_dict['run_params']['binary_threshold']
    print config.config_dict['run_params']['filters']
    print config.config_dict['model']['class_name']

    ##### Step 6 Call the model building and eval functions - from fitting.py
    if (cv_periods.at[i, 'train2-start'] == "None"):   #%Y-%m-%d
        tup1 = (datetime.strptime(cv_periods.at[i, 'cv-start'], "%Y-%m-%d").date(), datetime.strptime(cv_periods.at[i, 'cv-end'], "%Y-%m-%d").date())
        tup2 = (datetime.strptime(cv_periods.at[i, 'train1-start'], "%Y-%m-%d").date(), datetime.strptime(cv_periods.at[i, 'train1-end'], "%Y-%m-%d").date())
        # tup2 = tup1
        # tup2 = (datetime.strptime(cv_periods.at[i, 'train1-start'], "%Y-%m-%d").date(), datetime.strptime(cv_periods.at[i, 'train1-end'], "%Y-%m-%d").date())

        arg = (dao, config, tup1, tup2, None)  #cv_periods.at[i, 'train2-start']

    else:
        tup1 = (datetime.strptime(cv_periods.at[i, 'cv-start'], "%Y-%m-%d").date(), datetime.strptime(cv_periods.at[i, 'cv-end'], "%Y-%m-%d").date())
        tup2 = (datetime.strptime(cv_periods.at[i, 'train1-start'], "%Y-%m-%d").date(), datetime.strptime(cv_periods.at[i, 'train1-end'], "%Y-%m-%d").date())
        tup3 = (datetime.strptime(cv_periods.at[i, 'train2-start'], "%Y-%m-%d").date(), datetime.strptime(cv_periods.at[i, 'train2-end'], "%Y-%m-%d").date())

        arg = (dao, config, tup1, tup2, tup3)

    tstart = time.time()

    print "Started NB Model building and prediction"
    model_results.append(fitting.train_worker_cv(arg))   # can parallelize this separately by keeping all the dao and config objects created and sending them out!
    print "Completed NB in %0.1f secs" % (time.time() - tstart)

    print "Loop %d completed in %0.1f secs" % (i, time.time() - tloop)

    print "Out of the loop"

    icr_list = []
    icr_list_tr = []
    model_dict = {}
    for result in model_results:
        icr_list.append(result[0])
        icr_list_tr.append(result[2])
        cv_start_date = result[0]['cv-start-date'].values[0]
        cv_end_date = result[0]['cv-end-date'].values[0]
        model = result[1]  # this is an NB model
        datestr = cv_end_date.strftime("%Y%m%d")
        model_dict[(cv_start_date, cv_end_date)] = model
        model.persist(results_dir, datestr)

    results_df = pd.concat(icr_list)
    results_df = results_df.sort_values(by='date')
    assert not results_df.duplicated(subset=['gvkey', 'date']).any()
    # results_df.to_csv(os.path.join(results_dir, 'IC_results.csv'))
    cv_periods.to_csv(os.path.join(results_dir, 'cv_periods.csv'))

    # results_dir = '/research/home/rakesh/results/WalkForward/new_WF/TW_FS_PRL/check_day0_10rets2'
    outname = 'results_raw' + str(i) + '.csv'
    fname = os.path.join(results_dir, outname)
    results_df.to_csv(fname)

    # saving training results
    results_df2 = pd.concat(icr_list_tr)
    results_df2 = results_df2.sort_values(by='date')
    assert not results_df2.duplicated(subset=['gvkey', 'date']).any()

    outname = 'results_train' + str(i) + '.csv'
    fname = os.path.join(results_dir, outname)
    results_df2.to_csv(fname)

    # save config file for each run!
    outname = 'config_selected' + '.pkl'
    fname = os.path.join(results_dir, outname)
    output = open(fname, 'wb')
    pickle.dump(config, output)
    output.close()

    # remove all the pkls
    os.remove(os.path.join(results_dir, 'negcounts_list_' + str(i) + '.pkl'))
    os.remove(os.path.join(results_dir, 'poscounts_list_' + str(i) + '.pkl'))
    os.remove(os.path.join(results_dir, 'negcounts_' + str(i) + '.pkl'))
    os.remove(os.path.join(results_dir, 'poscounts_' + str(i) + '.pkl'))
    # os.remove(os.path.join(results_dir, 'negfilter_list_' + str(i) + '.pkl'))
    # os.remove(os.path.join(results_dir, 'posfilter_list_' + str(i) + '.pkl'))


    # results_df = []
    if return_models:
        return results_df, model_dict
    else:
        return results_df




# ###### Section for filtering out features
#
# print "Filtering the features out"
#
# # load the bow_df from TW_FS_NB_scl01_chi
# fname = os.path.join(results_dir, 'features_' + str(i) + '.csv')
# BNS_select = pd.read_csv(fname)
#
# fname = os.path.join(results_dir, 'bow_tfidf_returns_' + str(i) + '.csv')
# bow_df = pd.read_csv(fname)
#
# # load the BNS_select from TW_FS_NB_scl01_chi - get list of features with count > 2.25%
# max_counts_total = int(round(0.0225*(Npos+Nneg)))
# BNS_select = BNS_select[BNS_select['counts_total'] > max_counts_total]
# BNS_select_features = BNS_select['features']
# features_remove = BNS_select_features.values.tolist()
# feature_names_remove = ['feature_' + x for x in features_remove]
#
# # drop from bow_df those columns
# bow_df = bow_df.drop(feature_names_remove,axis=1)
#
# # save to file and load the new - in TW_FS_NB_scl01_chi/cutfeatures2_thresh05
# results_dir2 = '/research/home/rakesh/results/WalkForward/TW_FS_NB_scl01_chi/cutfeatures2_thresh05'
# outname = os.path.join(results_dir2, 'bow_tfidf_returns_' + str(i) + '.csv')
# bow_df.to_csv(outname)
#
# print bow_df.shape
# del bow_df
#
# fname = os.path.join(results_dir, 'features_' + str(i) + '.csv')
# BNS_select = pd.read_csv(fname)
# BNS_select = BNS_select[BNS_select['counts_total'] <= max_counts_total]
# fname = os.path.join(results_dir2, 'features_' + str(i) + '.csv')
# BNS_select.to_csv(fname)
#
# BNS_select_features = BNS_select['features']
# features = BNS_select_features.values.tolist()
# feature_names = ['feature_' + x for x in features]
