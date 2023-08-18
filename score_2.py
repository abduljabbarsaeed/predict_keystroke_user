import json

import numpy as np

import pandas as pd

import os

import pickle

import joblib

import sklearn

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from azureml.core import Model

def init():

    global model

    model_name = 'RF'

    path = Model.get_model_path(model_name)

    model = joblib.load(path)

def run(data):

    try:

        data_json = json.loads(data)
        print(data_json) 
        # create input data frame 
        df = pd.DataFrame({ 
                   'mean_HT': pd.Series(dtype='float'),
                   'mean_PPT': pd.Series(dtype='float'),
                   'mean_RRT': pd.Series(dtype='float'),
                   'mean_RPT': pd.Series(dtype='float'),
                   'sd_HT': pd.Series(dtype='float'),
                   'sd_PPT': pd.Series(dtype='float'),
                   'sd_RRT': pd.Series(dtype='float'),
                   'sd_RPT': pd.Series(dtype='float')})  

        # appending input row
        df = df.append({
                        'mean_HT':data_json["HT"]["Mean"],
                        'mean_PPT':data_json["PPT"]["Mean"],
                        'mean_RRT':data_json["RRT"]["Mean"],
                        'mean_RPT':data_json["RPT"]["Mean"],
                        'sd_HT':data_json["HT"]["STD"],
                        'sd_PPT':data_json["PPT"]["STD"],
                        'sd_RRT':data_json["RRT"]["STD"],
                        'sd_RPT':data_json["RPT"]["STD"]},ignore_index=True)      

        result = model.predict(df)

        return {'user predicted' : int(result) , 'message' : "Successfully classified User"}

    except Exception as e:

        error = str(e)

        return {'user predicted' : error , 'message' : 'Failed to classify user'}


