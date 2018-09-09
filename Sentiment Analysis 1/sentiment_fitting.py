import cPickle as pickle
import pandas as pd
import numpy as np
import re
import string
import datetime as dt
import multiprocessing as mp
import json

from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

import os, sys
import time

sys.path.append(os.path.expanduser('/research/home/rakesh/research/scripts/_modules/python'))

import pybt.data_access as da
from pybt import modeling

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

from scipy.stats import pearsonr, spearmanr

# load config file
with open('/research/home/rakesh/MyCode/sentiment_config.json') as f:
    config = json.load(f)

# Step 1: Load the filter phrases
df_temp = pd.read_csv("/research/home/rakesh/results/Article_Cleaning/relevant_1words.csv", header=None)
rel_wrds_load = df_temp[0].values.tolist()
rel_wrds = [e.lower() for e in rel_wrds_load]

rel_wrds = list(set(rel_wrds))

print len(rel_wrds), ": Total no. of relevant words"

phrase_fname = config['filter_phrase_list'] + '.csv'
fname = os.path.join('/research/home/rakesh/results/Article_Cleaning/', phrase_fname)
df_temp = pd.read_csv(fname, header=None)
rel_wrds_load = df_temp[0].values.tolist()
rel_phrases = [e.lower() for e in rel_wrds_load]

rel_phrases = list(set(rel_phrases))

print len(rel_phrases), ": Total no. of relevant phrases"

# Load LM dictionaries
df_negative = pd.read_csv("/research/home/rakesh/LM_negative.csv",header=0)
df_positive = pd.read_csv("/research/home/rakesh/LM_positive.csv",header=0)

# Converting df to list to feed CountVectorizer
positive_vocab = []
for k in range(len(df_positive)):
    positive_vocab.append(df_positive.values[k][0].lower())

negative_vocab = []
for k in range(len(df_negative)):
    negative_vocab.append(df_negative.values[k][0].lower())

def sents_to_tokens(sentence):

    tokens = [e.lower() for e in map(string.strip, re.split("(\W+)", sentence)) if len(e) > 1 and not re.match("[\W]",e)]

    return tokens

def intersection(lst1, lst2):

    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def phrase_filter(sent1, rel_phrases, ss_article):

    # common phrases - and sentiment scores
    common_phrases = []
    sentim_score = 0
    sent = sent1.lower()

    j = 0
    ph_index = []
    for ph in rel_phrases:
        if ph in sent:
            common_phrases.append(ph)
            ph_index.append(j)
        j += 1

    # removing the phrases from the sentence
    flag = 0
    if common_phrases:
        flag = 1
        sents_new = sent
        # for cp in common_phrases:
        #     sents_new = sents_new.replace(cp, "")

        # score the phrase removed sentence
        ss = analyser.polarity_scores(sents_new)
        sentim_score = ss['compound']
        ss_article[ph_index] += sentim_score

    return ss_article, sentim_score, flag

def scores_worker(event_keys_split):

    dict_return = {}
    Nph = len(rel_phrases)

    for ring in event_keys_split:

        article_select = event_corpus[ring]
        sents = tokenize.sent_tokenize(article_select)

        Ns = len(sents)
        if (Ns < 5):
            continue

        ss_words, ss_phrases, ss_total = 0, 0, 0
        Nw, Np = 0, 0

        pwords = np.zeros(Ns)
        nwords = np.zeros(Ns)
        sent_with_pwords, sent_with_nwords = 0, 0

        ss_article = np.zeros(Nph)

        for i in range(Ns):

            sents_toks = sents_to_tokens(sents[i])

            ss_article, sentim_score, filter_flag = phrase_filter(sents[i], rel_phrases, ss_article)

            if (filter_flag == 0):
                # sentiment analysis without filter
                ss = analyser.polarity_scores(sents[i])
                sentim_score = ss['compound']
            else:
                Np += 1
                ss_phrases += sentim_score

            ss_total += sentim_score

            # common words
            common_words = intersection(rel_wrds, sents_toks)
            if common_words:
                ss_words += sentim_score
                Nw += 1

            # common with pos/neg vocab
            common_pwords = intersection(positive_vocab, sents_toks)
            if common_pwords:
                pwords[i] = len(common_pwords)
                sent_with_pwords += 1

            common_nwords = intersection(negative_vocab, sents_toks)
            if common_nwords:
                nwords[i] = len(common_nwords)
                sent_with_nwords += 1

        score_posneg_words = (sent_with_pwords - sent_with_nwords)*1.0/Ns
        if (Np == 0) or (Nw == 0):
            if (Np == 0) and (Nw == 0):
                other_scores = np.array([0.0, 0.0, ss_total*1.0/Ns, pwords.sum()/Ns, nwords.sum()/Ns, score_posneg_words, len(sents)])
            elif (Nw == 0):
                other_scores = np.array([0.0, ss_phrases*1.0/Np, ss_total*1.0/Ns, pwords.sum()/Ns, nwords.sum()/Ns, score_posneg_words, len(sents)])
            else:
                other_scores = np.array([ss_words*1.0/Nw, 0.0, ss_total*1.0/Ns, pwords.sum()/Ns, nwords.sum()/Ns, score_posneg_words, len(sents)])
        else:
            other_scores = np.array([ss_words*1.0/Nw, ss_phrases*1.0/Np, ss_total*1.0/Ns, pwords.sum()/Ns, nwords.sum()/Ns, score_posneg_words, len(sents)])

        scores = np.append(ss_article, other_scores)

        dict_return[ring] = scores

    return dict_return


def scores_calc():

    event_keys = event_corpus.keys()
    # event_keys = event_keys[:1000]

    # parallelize here and call a scoring function
    ncomb = 1000  #1000
    nevents = len(event_keys)
    nsplit = np.divide(nevents, ncomb) + (np.remainder(nevents, ncomb) > 0)
    inps = [event_keys[q*ncomb:(q+1)*ncomb] for q in range(nsplit)]

    pool = mp.Pool(processes = 25)
    results = pool.map(scores_worker, inps)

    # combine the results
    dict_scores = {}
    for d in results:
        dict_scores.update(d)

    # dict_scores = scores_worker(event_keys[:10])

    # convert dict to df - or combine all dicts into a df
    df_scores = pd.DataFrame.from_dict(dict_scores)
    index_names = list(rel_phrases)
    index_names += ["word_filt_ss", "phrase_filt_ss", "article_ss", "pwords_per_sent", "nwords_per_sent", "posneg_sent_score", "sent_length"]
    df_scores.index = index_names

    df_scores = df_scores.transpose()

    return df_scores


def run_wf(argins):

    global event_corpus
    event_corpus, event_art_num, returns_select = argins

    tst = time.time()
    print "Started scores_calc"
    df_scores = scores_calc()
    print "Completed %f" %(time.time() - tst)

    # map to the mapping here - only needed when config['corpus_type'] == 'article'

    if config['corpus_type'] == 'article':
        # save scores and exit
        df_scores['transcript_id'] = df_scores.index
        df_scores_save = df_scores[["transcript_id", "word_filt_ss", "phrase_filt_ss", "article_ss", "pwords_per_sent", "nwords_per_sent", "posneg_sent_score", "sent_length"]]
        df_scores_save.index = np.arange(len(df_scores))
        fname = os.path.join(config['results_directory'], "sentiment_scores_transcripts.csv")
        df_scores_save.to_csv(fname)

        exit()

    # bring the gvkey + event to be able to merge with returns
    df_scores['gvkey + event_date'] = df_scores.index
    df_scores.index = np.arange(len(df_scores))
    fname = os.path.join(config['results_directory'], "scores.csv")
    df_scores.to_csv(fname)

    # add feature_ tag here?
    feature_names = ['feature_' + x for x in df_scores.columns]
    df_scores.columns = feature_names

    # ### scaling the scores here - this has some forward bias
    # df_scores2 = df_scores
    # scaler = MinMaxScaler(feature_range=(0,1))   # or Standard Scaler
    # df_scores.loc[:, feature_names] = scaler.fit_transform(df_scores2)


    df_XY = pd.merge(df_scores, returns_select, how='inner', on=['gvkey + event_date'],
                     sort=False,                                         # left_index=True,
                     suffixes=('_x', '_y'), copy=True, indicator=False)
    print len(df_XY)
    # save df_scores to merge with other returns? maybe I can merge all returns here...?
    fname = os.path.join(config['results_directory'], "scores_returns.csv")
    df_XY.to_csv(fname)

    # # Load an already saved scores + returns matrix
    # XY_dir = '/research/home/rakesh/results/WalkForward/Sentiment/phrase1_check'
    # fname = os.path.join(XY_dir, "scores_returns.csv")
    # df_XY = pd.read_csv(fname, index_col=0)
    # fn = [x.startswith('feature_') for x in df_XY.columns]
    # feature_names = df_XY.columns[fn]

    # split model data based on event_date - and the WF periods
    # if (config['WF_or_CV'] == 'WF'):
    cv_periods = pd.read_csv(config['periods_fname'], index_col=0, header=0, parse_dates=True)
    rets_train_label = config['returns_choice'][0]
    rets_test_label = config['returns_choice'][1]

    print config['returns_choice'], ": Choice of Returns"

    for i in cv_periods.index:

        # i = 0   # a loop here?
        mask1 = (df_XY['date'] > cv_periods.at[i, 'train1-start']) & (df_XY['date'] < cv_periods.at[i, 'train1-end'])
        mask2 = (df_XY['date'] > cv_periods.at[i, 'cv-start']) & (df_XY['date'] < cv_periods.at[i, 'cv-end'])
        df_XY_train = df_XY[mask1]
        df_XY_eval = df_XY[mask2]

        print len(df_XY_train), len(df_XY_eval)

        if (config['objective'] == 'binary'):
            binary_thresh = config['binary_thresh']

            mask = (df_XY_train[rets_train_label] >= binary_thresh) | (df_XY_train[rets_train_label] < -binary_thresh)
            df_XY_tr_select = df_XY_train[mask]

            rets_train = (df_XY_tr_select[rets_train_label] >= binary_thresh)*1.0 + (df_XY_tr_select[rets_train_label] < binary_thresh)*0.0
            feats_train = df_XY_tr_select[feature_names]

        elif (config['objective'] == 'regression'):
            rets_train = df_XY_train[rets_train_label]
            feats_train = df_XY_train[feature_names]

        rets_eval = df_XY_eval[rets_test_label]
        feats_eval = df_XY_eval[feature_names]

        ### scaling the scores here
        feats_train2 = feats_train
        scaler = MinMaxScaler(feature_range=(0,1))   # or Standard Scaler
        feats_train2 = feats_train2.fillna(value = 0)
        feats_train.loc[:, feature_names] = scaler.fit_transform(feats_train2)

        feats_eval2 = feats_eval
        feats_eval2 = feats_eval2.fillna(value = 0)
        feats_eval.loc[:, feature_names] = scaler.transform(feats_eval2)   # but these values have to be between zero and one! no guarantee - so maybe another transform?

        ##### Model training and eval
        class_obj = getattr(sys.modules['__main__'], config['model_class_name'])
        model = class_obj()

        print "Applying ", config['model_class_name']

        print feats_train.shape
        model.train(feats_train, rets_train)

        # # calculate accuracy and IC in training data - how good is the fit in training
        train_res = pd.DataFrame(index=rets_train.index)
        train_res['actual'] = rets_train
        train_res['predicted'] = model.predict(feats_train)
        train_res['predicted2'] = model.predict2(feats_train)
        train_res['gvkey'] = df_XY_train['gvkey']
        train_res['date'] = df_XY_train['date']
        train_res['cv-start-date'] = cv_periods.at[i, 'cv-start']
        train_res['cv-end-date'] = cv_periods.at[i, 'cv-end']

        outname = 'results_train' + str(i) + '.csv'
        fname = os.path.join(config['results_directory'], outname)
        train_res.to_csv(fname)

        ## model eval
        test_res = pd.DataFrame(index=rets_eval.index)
        test_res['actual'] = rets_eval
        test_res['predicted'] = model.predict(feats_eval)
        test_res['predicted2'] = model.predict2(feats_eval)
        test_res['gvkey'] = df_XY_eval['gvkey']
        test_res['date'] = df_XY_eval['date']
        test_res['cv-start-date'] = cv_periods.at[i, 'cv-start']
        test_res['cv-end-date'] = cv_periods.at[i, 'cv-end']

        test_res = test_res.sort_values(by='date')
        assert not test_res.duplicated(subset=['gvkey', 'date']).any()

        outname = 'results_raw' + str(i) + '.csv'
        fname = os.path.join(config['results_directory'], outname)
        test_res.to_csv(fname)

        # save config file
        fname = os.path.join(config['results_directory'], 'config.json')
        with open(fname, 'w') as outfile:
            json.dump(config, outfile)


# ### To consider pwords and nwords in only he word-selected sentences
#
# def scores_worker(event_keys_split):
#
#     dict_return = {}
#     Nph = len(rel_phrases)
#
#     for ring in event_keys_split:
#
#         article_select = event_corpus[ring]
#         sents = tokenize.sent_tokenize(article_select)
#
#         Ns = len(sents)
#         if (Ns < 5):
#             continue
#
#         ss_words, ss_phrases, ss_total = 0, 0, 0
#         Nw, Np = 0, 0
#
#         pwords = np.zeros(Ns)
#         nwords = np.zeros(Ns)
#
#         ss_article = np.zeros(Nph)
#
#         for i in range(Ns):
#
#             sents_toks = sents_to_tokens(sents[i])
#
#             ss_article, sentim_score, filter_flag = phrase_filter(sents[i], rel_phrases, ss_article)
#
#             if (filter_flag == 0):
#                 # sentiment analysis without filter
#                 ss = analyser.polarity_scores(sents[i])
#                 sentim_score = ss['compound']
#             else:
#                 Np += 1
#                 ss_phrases += sentim_score
#
#             ss_total += sentim_score
#
#             # common words
#             common_words = intersection(rel_wrds, sents_toks)
#             if common_words:
#                 ss_words += sentim_score
#                 Nw += 1
#
#                 # common with pos/neg vocab
#                 common_pwords = intersection(positive_vocab, sents_toks)
#                 if common_pwords:
#                     pwords[i] = len(common_pwords)
#
#                 common_nwords = intersection(negative_vocab, sents_toks)
#                 if common_nwords:
#                     nwords[i] = len(common_nwords)
#
#         if (Np == 0) or (Nw == 0):
#             if (Np == 0) and (Nw == 0):
#                 other_scores = np.array([0.0, 0.0, ss_total*1.0/Ns, 0.0, 0.0, len(sents)])
#             elif (Nw == 0):
#                 other_scores = np.array([0.0, ss_phrases*1.0/Np, ss_total*1.0/Ns, 0.0, 0.0, len(sents)])
#             else:
#                 other_scores = np.array([ss_words*1.0/Nw, 0.0, ss_total*1.0/Ns, pwords.sum()/Nw, nwords.sum()/Nw, len(sents)])
#         else:
#             other_scores = np.array([ss_words*1.0/Nw, ss_phrases*1.0/Np, ss_total*1.0/Ns, pwords.sum()/Nw, nwords.sum()/Nw, len(sents)])
#
#         scores = np.append(ss_article, other_scores)
#
#         dict_return[ring] = scores
#
#     return dict_return
