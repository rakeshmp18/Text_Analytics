import cPickle as pickle
import pandas as pd
import numpy as np
import re
import string
from nltk import tokenize
import datetime as dt
import json

import os, sys
import time

sys.path.append(os.path.expanduser('/research/home/rakesh/research/scripts/_modules/python'))

import pybt.data_access as da
from pybt import modeling
from model_classes import *
import sentiment_fitting
import sentiment_fitting_combine
import sentiment_fitting_groups

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

from scipy.stats import pearsonr, spearmanr

tstart_full = time.time()

# load config file
with open('/research/home/rakesh/MyCode/sentiment_config.json') as f:
    config = json.load(f)

results_dir = config['results_directory']
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print "Saving Results to", results_dir

print "Loading Transript Text Data"

# Load data, returns and mapping
with open(os.path.join(config['corpus_folder'], config['corpus_fname']), 'rb') as ofp:
    event_corpus = pickle.load(ofp)

if (config['corpus_type'] == 'event'):
    with open(os.path.join(config['corpus_folder'], config['event_num_fname']), 'rb') as ofp:
        event_art_num = pickle.load(ofp)
else:
    event_art_num = {}

print "Completed Loading Text Data"

# Load returns
start_date = '2006-01-01'
end_date = '2018-12-31'

tstart = time.time()

ret_file = os.path.join(config['returns_folder'], "post1_earnings.csv")
tdao = da.TrainingDataAccess(ret_file)

start_dt_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
end_dt_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

inds, rets = tdao.training_data_range(start_dt_date, end_dt_date, include_date=True)

# this in effect merges the returns and indices
rets['gvkey'] = inds['gvkey']
rets['date'] = pd.to_datetime(rets['date'])
inds['date'] = pd.to_datetime(inds['date'])

print "Returns for %d events loaded in %0.1f seconds" % (len(rets), (time.time()-tstart))

returns_select = pd.merge(rets, inds, how='inner', on=['gvkey', 'date'],
                         sort=False,                                         # left_index=True,
                         suffixes=('_x', '_y'), copy=True, indicator=False)

returns_select['date'] = returns_select['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
returns_select['gvkey + event_date'] = returns_select['gvkey'] + ' ' + returns_select['date']
returns_select['date'] = pd.to_datetime(returns_select['date'])

print returns_select.shape, ": Returns Shape"

# event_corpus = {}
# event_art_num = {}
# returns_select = []

sentiment_fitting.run_wf([event_corpus, event_art_num, returns_select])
# sentiment_fitting_combine.run_wf([event_corpus, event_art_num, returns_select])
# sentiment_fitting_groups.run_wf([event_corpus, event_art_num, returns_select])
print "completed in %0.1f" %(time.time() - tstart_full)
