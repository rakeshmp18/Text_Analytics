### This code calls the CV Framework

import json
import sys, os, shutil
import numpy as np
from scipy.stats import norm, spearmanr, ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.switch_backend('agg')
import multiprocessing as mp
import cPickle as pickle
import collections
import argparse

import time
import datetime as dt
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE

sys.path.append(os.path.expanduser('/research/home/rakesh/research/scripts/_modules/python'))

import pybt.data_access as da
from pybt.config_helper import ConfigHelper
from pybt import modeling

import fitting_FS_WF

results_dir = '/research/home/rakesh/results/WalkForward/new_WF/TW_FS_PRL_day0_1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print results_dir

assert(os.path.isdir(results_dir))

print "Loading event corpus"

tstart = time.time()
with open('/research/home/rakesh/results/Article_Cleaning/new2_PRLstemnounclean2_events_corpus.pkl', 'rb') as ofp:
    event_corpus = pickle.load(ofp)
# event_corpus = {}
pkl_file = open('/research/home/rakesh/results/Article_Cleaning/new2_PRL_art_num.pkl', 'rb')
event_art_num = pickle.load(pkl_file)
pkl_file.close()

event_keys = event_corpus.keys()

print "Loaded corpus for %d events loaded in %0.1f seconds" % (len(event_keys), (time.time()-tstart))


# Load returns - of each event, returns not related to each article directly

start_date = '2006-01-01'
end_date = '2015-12-31'

tstart = time.time()
temp = '/research/home/ryan/Dev/research-analysis/rgreen/amer-indicator-sets/packages/amer_v1/earnings/post-earnings/post1_earnings.csv'
ret_file = temp   #os.path.join(NETAPP_DIR, temp)
tdao = da.TrainingDataAccess(ret_file)

start_dt_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
end_dt_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

inds, rets = tdao.training_data_range(start_dt_date, end_dt_date, include_date=True)

# this in effect merges the returns and indices
rets['gvkey'] = inds['gvkey']
rets['date'] = pd.to_datetime(rets['date'])
inds['date'] = pd.to_datetime(inds['date'])

print "Returns for %d events loaded in %0.1f seconds" % (len(rets), (time.time()-tstart))

label_choice = 'fwd_xmkt_0_1'   # 'fwd_xmkt_1_10'  'ret_demean_sec_1'
if label_choice.startswith('fwd_'):
    event_dates = rets[['date', 'gvkey', label_choice]]  # just making a copy
    print "Used rets"
else:
    event_dates = inds[['date', 'gvkey', label_choice]]
    print "Used inds"
event_dates['date'] = event_dates['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
event_dates['gvkey + event_date'] = event_dates['gvkey'] + ' ' + event_dates['date']
event_dates['date'] = pd.to_datetime(event_dates['date'])
event_dates = event_dates.rename(columns = {label_choice: "returns_select"})


# # This loads the mapping between articles and events (which is gvkey + date)
DataFolder = '/research/data/tmp/Dow_Jones_Global/amer-v1-filtered'
mapping_file = os.path.join(DataFolder, "post1_article_mapping.csv")
mapping = pd.read_csv(mapping_file, index_col = 0, parse_dates = ['display_date_EST', 'event_date'])

mapping['date'] = pd.to_datetime(mapping['date'])
mask = (mapping['date'] > start_date) & (mapping['date'] <= end_date)
mapping_select = mapping.loc[mask]

# event_date is what has to be matched with the returns where the event date is under the 'date' column - RG
mapping_select = mapping_select.drop(['date'], axis=1)
mapping_select['date'] = mapping_select['event_date']

print "No. of articles loaded between %s and %s are %d" %(start_date, end_date, len(mapping_select))


class NaiveBayesModel(modeling.ForecastModel):

    @staticmethod
    def from_config(config):
        return NaiveBayesModel()

    @staticmethod
    def model_dates(model_dir):
        pass

    def __init__(self):
        self.model = MultinomialNB()
        # self.model = GaussianNB()

    def train(self, indicators, returns, validation_indicators=None, validation_returns=None, weights=None):
        # print weights[:3]
        self.model.fit(indicators, returns, weights)
        self.trained = True

    def predict(self, indicators):
        yhat = self.model.predict_proba(indicators)
        return yhat[:,1]

    def predict2(self, indicators):
        yhat = self.model.predict(indicators)
        return yhat

    def persist(self, out_dir, datestr):
        pass

    def load(self, model_dir, datestr):
        pass

# class SVCModel(modeling.ForecastModel):
#
#     @staticmethod
#     def from_config(config):
#         return SVCModel()
#
#     @staticmethod
#     def model_dates(model_dir):
#         pass
#
#     def __init__(self):
#         # self.model = SVC(kernel='rbf', gamma=1e-8, C=12, cache_size=1000)
#         # self.model = SVC(kernel='poly', degree=3 , C=12, cache_size=1000, probability=True)
#         # base_model = SVC(kernel='rbf', gamma=1e-8, C=12, cache_size=1000)
#         # base_model = MultinomialNB()
#         # base_model = LinearSVC(C=12)
#         base_model = SVC(kernel='poly', degree=2 , C=12, cache_size=1000)
#         self.model = CalibratedClassifierCV(base_model, method='isotonic')
#         self.model = RFE(base_model, 4000, step = 4000)
#
#     def train(self, indicators, returns, validation_indicators=None, validation_returns=None, weights=None):
#         self.model.fit(indicators, returns)
#         self.trained = True
#
#     def predict(self, indicators):
#         yhat = self.model.predict_proba(indicators)
#         return yhat[:,1]
#
#     def predict2(self, indicators):
#         yhat = self.model.predict(indicators)
#         return yhat
#
#     def persist(self, out_dir, datestr):
#         pass
#
#     def load(self, model_dir, datestr):
#         pass

tstart = time.time()

odir = results_dir   #'/research/home/rakesh/results/temp'
parser = argparse.ArgumentParser()
parser.add_argument('--wf_period', required=True, type=int, help='WF period')
walk_fwd_period = parser.parse_args().wf_period

# results_temp = fitting.run_cv(dao, config, odir, 4)
event_info = [event_corpus, event_dates, event_art_num]
results_temp = fitting_FS_WF.run_cv_features(event_info, mapping_select, odir, walk_fwd_period)    #, config

print "Complete CV Framework run in %0.1f secs" % (time.time()-tstart)

# # results_dir = '/research/home/rakesh/results/TwoWord/complete_results/CVnew_5K_SVC/PolySVR12_deg3_proba'
# fname = os.path.join(results_dir, 'results_raw0.csv')
# results_temp.to_csv(fname)
#
# fname = os.path.join(results_dir, 'model_dict0.pkl')
# output = open(fname, 'wb')
# pickle.dump(models_dict, output)
# output.close()
