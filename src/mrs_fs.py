import random
import numpy as np
import pandas as pd

from modules.mrs_id import MRSIdentifier

class MRSFeatureSelection(object):
    
    def run(self, features, labels, mode=None, k=None):
        if type(features) is not pd.core.frame.DataFrame:
            features = pd.DataFrame(features)
        
        selected_plot_nums, selected_features = [], []

        # ----- STEP.1 -----
        if mode == 'random':
            if not isinstance(k, int):
                print("選出する特徴量の数をkにint型で入れてください．")
            else:
                sf, rf = self._set_of_selected_random_features(features, k)
        elif mode == 'specific':
            if not (isinstance(k, list) or isinstance(k, np.ndarray)):
                print("指定したい特徴量をkにlist型で入れてください．")
            else:
                sf, rf = self._separate_feature_set(features, k)
        # ----- STEP.2 -----
        size_of_mrs, feature_set = self._mrs_feature_selection_algorithm(sf, rf, labels)

        return size_of_mrs, feature_set
    
    def _set_of_selected_random_features(self, features, k):
        sf = features.sample(n=k, axis=1)#, random_state=0
        rf = features.drop(sf.columns, axis=1)
        return sf, rf
    
    def _separate_feature_set(self, features, k):
        sf = features[k]
        rf = features.drop(sf.columns, axis=1)
        return sf, rf
    
    def _mrs_feature_selection_algorithm(self, sf, rf, labels):
        sf_use = sf.copy()
        rf_use = rf.copy()
        
        mrs_id = MRSIdentifier()
        s = mrs_id.run(sf_use.values, labels)
        
        for i in range(len(sf.iloc[0])):
            for j in range(len(rf.iloc[0])):
                # ----- swap -----
                sf_swap = sf_use.copy()
                sf_swap.iloc[:,i] = rf_use.iloc[:,j]
                # ----- perform MRS_ID for samples with feature set SF -----
                s_1 = mrs_id.run(sf_swap.values, labels)
                
                if s_1 < s:
                    s = s_1
                    rf_columns = rf_use.columns[j]
                    sf_columns = sf_use.columns[i]
                    # rfの 値の代入 と 名前の入れ替え
                    rf_use.iloc[:,j] = sf_use.iloc[:,i]
                    rf_use.rename(columns={rf_columns: sf_columns}, inplace=True)
                    # sfの 値の代入 と 名前の入れ替え
                    sf_use = sf_swap
                    sf_use.rename(columns={sf_columns: rf_columns}, inplace=True)
        return s, sf_use