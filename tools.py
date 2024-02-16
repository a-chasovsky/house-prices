import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import time
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OrdinalEncoder
from feature_engine.outliers import OutlierTrimmer

class OverviewTransformer():

    def __init__(self, features_na):
        self.features_na = features_na


    def fit(self, X):
        pass


    def transform(self, X):
        X_ = X.copy()
        X_.columns = map(str.lower, X_.columns)
        features_na = self.features_na
        X_ = X_.rename(
            columns={
                'condition1': 'condition_first',
                'condition2': 'condition_second',
                'exterior1st': 'exterior_first',
                'exterior2nd': 'exterior_second',
                'bsmtfintype1': 'bsmtfintype_first',
                'bsmtfinsf1': 'bsmtfinsf_first',
                'bsmtfintype2': 'bsmtfintype_second',
                'bsmtfinsf2': 'bsmtfinsf_second',
                '1stflrsf': 'first_flrsf',
                '2ndflrsf': 'second_flrsf',
                '3ssnporch': 'three_ssnporch'
                }
        )
        X_[features_na] = X_[features_na].fillna('NA')
        garage_fill = X_.loc[X_['garageyrblt'].isna(), 'yearbuilt']
        loc = (X_['garageyrblt'].isna(), 'garageyrblt')
        X_.loc[loc] = X_.loc[loc].fillna(garage_fill)
        return X_


    def fit_transform(self, X, y):
        X_ = X.copy()
        X_.columns = map(str.lower, X_.columns)
        features_na = self.features_na
        X_ = X_.rename(
            columns={
                'condition1': 'condition_first',
                'condition2': 'condition_second',
                'exterior1st': 'exterior_first',
                'exterior2nd': 'exterior_second',
                'bsmtfintype1': 'bsmtfintype_first',
                'bsmtfinsf1': 'bsmtfinsf_first',
                'bsmtfintype2': 'bsmtfintype_second',
                'bsmtfinsf2': 'bsmtfinsf_second',
                '1stflrsf': 'first_flrsf',
                '2ndflrsf': 'second_flrsf',
                '3ssnporch': 'three_ssnporch'
                }
        )
        X_[features_na] = X_[features_na].fillna('NA')
        garage_fill = X_.loc[X_['garageyrblt'].isna(), 'yearbuilt']
        loc = (X_['garageyrblt'].isna(), 'garageyrblt')
        X_.loc[loc] = X_.loc[loc].fillna(garage_fill)
        return X_


class LabelTransformer():

    def __init__(self, features_labels_lists, features_labels_dicts):
        self.features_labels_lists = features_labels_lists
        self.features_labels_dicts = features_labels_dicts


    def fit(self, X):
        pass


    def transform(self, X):
        X_ = X.copy()
        
        zip_ = zip(self.features_labels_lists, self.features_labels_dicts)
        for lst, dct in zip_:
            for f in lst:
                X_[f] = X_[f].replace(dct)

        X_['mssubclass'] = X_['mssubclass'].apply(lambda x: str(x))
        
        return X_


    def fit_transform(self, X, y):
        X_ = X.copy()
        zip_ = zip(self.features_labels_lists, self.features_labels_dicts)
        for lst, dct in zip_:
            for f in lst:
                X_[f] = X_[f].replace(dct)
        return X_


class FeaturesCreator():

    def __init__(self, factors=True):
        self.factors = factors


    def fit(self, x, y):
        x = x.copy()
        pass


    def transform(self, x):
        factors = self.factors
        
        x['flrsfmean'] = ((x['first_flrsf']
                       + 0.7*x['second_flrsf']) / 1.7)
        x['bedroomsize'] = (x['bedroomabvgr'] / x['grlivarea'])
        x['kitchensize'] = (x['kitchenabvgr'] / x['grlivarea'])
        
        x['bedroomfracrms'] = (x['bedroomabvgr'] / x['totrmsabvgrd'])
        # max value of 'bedroomfracrms' except inf
        loc_value = (~np.isinf(x['bedroomfracrms']), 'bedroomfracrms')
        value = x.loc[loc_value].max()
        # fill inf values with max value
        loc_r = np.isinf(x['bedroomfracrms'])
        x.loc[loc_r, 'bedroomfracrms'] = value
    
        x['kitchenfracrms'] = (x['kitchenabvgr'] / x['totrmsabvgrd'])
        # max value of 'kitchenfracrms' except inf
        loc_value = (~np.isinf(x['kitchenfracrms']), 'kitchenfracrms')
        value = x.loc[loc_value].max()
        # fill inf values with max value
        loc_r = np.isinf(x['kitchenfracrms'])
        x.loc[loc_r, 'kitchenfracrms'] = value
        # fill NaN values by 0
        x['kitchenfracrms'] = x['kitchenfracrms'].fillna(0)
    
        x['bathsfracbedr'] = (x['fullbath'] / x['bedroomabvgr'])
        # max value of 'bathsfracbedr' except inf
        loc_value = (~np.isinf(x['bathsfracbedr']), 'bathsfracbedr')
        value = x.loc[loc_value].max()
        # fill inf values with max value
        loc_r = np.isinf(x['bathsfracbedr'])
        x.loc[loc_r, 'bathsfracbedr'] = value
        # fill NaN values by 0
        x['bathsfracbedr'] = x['bathsfracbedr'].fillna(0)

        for f in ['bedroomfracrms', 'kitchenfracrms', 'bathsfracbedr']:
            x[f] = np.round(x[f], 4)

        if factors:
            x['yearremodadd_exst'] = (x['yearremodadd']!=x['yearbuilt']).astype(int)
            # factor_labels_dict = {0: 'N', 1: 'Y'}
            # x['yearremodadd_exst'] = x['yearremodadd_exst'].map(factor_labels_dict)
            
            features_to_factor = [
                'masvnrarea', 'bsmtfinsf_first', 'bsmtfinsf_second', 
                'totalbsmtsf', 'bsmtunfsf', 'lowqualfinsf', 'second_flrsf', 'garagearea',
                'wooddecksf', 'openporchsf', 'enclosedporch', 'three_ssnporch',
                'screenporch', 'poolarea', 'miscval'
            ]
            
            for feature in features_to_factor:
                new_feature_name = feature + '_exst'
                x[new_feature_name] = (x[feature] != 0).astype(int)
                # x[new_feature_name] = x[new_feature_name].map(factor_labels_dict)
          
        cond = (x['yearremodadd_exst']==1)
        outcome1 = (x['yrsold'] - x['yearremodadd'])
        outcome0 = (x['yrsold'] - x['yearbuilt'])
        x['modage'] = np.where(cond, outcome1, outcome0)
        # x = x.drop('yearremodadd_exst', axis=1)
        
        x['houseage'] = x['yrsold'] - x['yearbuilt']
        x['garageage'] = x['yrsold'] - x['garageyrblt']

        return x


    def fit_transform(self, x, y):
        factors = self.factors
        
        x['flrsfmean'] = ((x['first_flrsf']
                       + 0.7*x['second_flrsf']) / 1.7)
        x['bedroomsize'] = (x['bedroomabvgr'] / x['grlivarea'])
        x['kitchensize'] = (x['kitchenabvgr'] / x['grlivarea'])
        
        x['bedroomfracrms'] = (x['bedroomabvgr'] / x['totrmsabvgrd'])
        # max value of 'bedroomfracrms' except inf
        loc_value = (~np.isinf(x['bedroomfracrms']), 'bedroomfracrms')
        value = x.loc[loc_value].max()
        # fill inf values with max value
        loc_r = np.isinf(x['bedroomfracrms'])
        x.loc[loc_r, 'bedroomfracrms'] = value
    
        x['kitchenfracrms'] = (x['kitchenabvgr'] / x['totrmsabvgrd'])
        # max value of 'kitchenfracrms' except inf
        loc_value = (~np.isinf(x['kitchenfracrms']), 'kitchenfracrms')
        value = x.loc[loc_value].max()
        # fill inf values with max value
        loc_r = np.isinf(x['kitchenfracrms'])
        x.loc[loc_r, 'kitchenfracrms'] = value
        # fill NaN values by 0
        x['kitchenfracrms'] = x['kitchenfracrms'].fillna(0)
    
        x['bathsfracbedr'] = (x['fullbath'] / x['bedroomabvgr'])
        # max value of 'bathsfracbedr' except inf
        loc_value = (~np.isinf(x['bathsfracbedr']), 'bathsfracbedr')
        value = x.loc[loc_value].max()
        # fill inf values with max value
        loc_r = np.isinf(x['bathsfracbedr'])
        x.loc[loc_r, 'bathsfracbedr'] = value
        # fill NaN values by 0
        x['bathsfracbedr'] = x['bathsfracbedr'].fillna(0)

        for f in ['bedroomfracrms', 'kitchenfracrms', 'bathsfracbedr']:
            x[f] = np.round(x[f], 4)

        if factors:
            x['yearremodadd_exst'] = (x['yearremodadd']!=x['yearbuilt']).astype(int)
            # factor_labels_dict = {0: 'N', 1: 'Y'}
            # x['yearremodadd_exst'] = x['yearremodadd_exst'].map(factor_labels_dict)
    
            features_to_factor = [
                'masvnrarea', 'bsmtfinsf_first', 'bsmtfinsf_second', 
                'totalbsmtsf', 'bsmtunfsf', 'lowqualfinsf', 'second_flrsf', 'garagearea',
                'wooddecksf', 'openporchsf', 'enclosedporch', 'three_ssnporch',
                'screenporch', 'poolarea', 'miscval'
            ]
            
            for feature in features_to_factor:
                new_feature_name = feature + '_exst'
                x[new_feature_name] = (x[feature] != 0).astype(int)
                # x[new_feature_name] = x[new_feature_name].map(factor_labels_dict)
            
        cond = (x['yearremodadd_exst']=='Y')
        outcome1 = (x['yrsold'] - x['yearremodadd'])
        outcome0 = (x['yrsold'] - x['yearbuilt'])
        x['modage'] = np.where(cond, outcome1, outcome0)
        # x = x.drop('yearremodadd_exst', axis=1)
        
        x['houseage'] = x['yrsold'] - x['yearbuilt']
        x['garageage'] = x['yrsold'] - x['garageyrblt']

        return x


class FeaturesLogger():

    def __init__(self, features_log):
        self.features_log = features_log


    def fit(self, x, y):
        x = x.copy()
        pass


    def transform(self, x):
        features_log = self.features_log
        for feature in features_log:
            const = 1
            x[feature] = np.log(x[feature] + const)
            x = x.rename(columns={feature: 'lg_'+feature})
        
        return x


    def fit_transform(self, x, y):
        features_log = self.features_log
        for feature in features_log:
            const = 1
            x[feature] = np.log(x[feature] + const)
            x = x.rename(columns={feature: 'lg_'+feature})
        
        return x


class Scaler():

    def __init__(self, features_transform):
        self.features = features_transform


    def fit(self, X):
        features = self.features
        self.scaler_sk = StandardScaler()
        # self.scaler_sk = MinMaxScaler()
        self.scaler_sk.fit(X[features])


    def transform(self, X):
        X_ = X.copy()
        features = self.features
        scaler_sk = self.scaler_sk
        X_[features] = scaler_sk.transform(X_[features])
        return X_


    def fit_transform(self, X):
        X_ = X.copy()
        features = self.features
        self.scaler_sk = StandardScaler()
        X_[features] = self.scaler_sk.fit_transform(X_[features])
        return X_


class Encoder():

    def __init__(self, features_transform):
        self.features = features_transform


    def fit(self, X, y):
        features = self.features
        self.encoder = OrdinalEncoder(
            encoding_method='ordered',
            variables=features,
            missing_values='ignore',
            unseen='encode'
        )
        self.encoder.fit(X[features], y)


    def transform(self, X):
        X_ = X.copy()
        features = self.features
        X_[features] = self.encoder.transform(X_[features])
        return X_


    def fit_transform(self, X, y):
        X_ = X.copy()
        features = self.features
        self.encoder = OrdinalEncoder(
            encoding_method='ordered',
            variables=features,
            missing_values='ignore',
            unseen='encode'
        )
        self.encoder.fit(X_[features], y)
        X_[features] = self.encoder.transform(X_[features])
        return X_


class Stopwatch():

    def start(self):
        time_start = time.time()
        return time_start

    
    def stop_sec(self, start=None):
        if start is None:
            time_sec = time.time()
        else:
            time_sec = time.time() - start
        return time_sec


    def stop(self, start=None):
        if start is None:
            time_format = time.time()
        else:
            time_sec = time.time() - start
            time_sec = dt.timedelta(seconds=np.round(time_sec))
            time_format = str(time_sec)
        return time_format


class ModelsEvaluator():

    def __init__(
            self, target_variable, models, models_names,
            sample_frac=0.3, replace=True, full_results=False, n_folds=1000):
        self.target_ = target_variable
        self.models_ = models
        self.models_names_ = models_names
        self.sample_frac_ = sample_frac
        self.replace_ = replace
        self.full_results_ = full_results
        self.n_folds_ = n_folds


    def fit(self, X, y):
        # time 
        t_fit_st = time.time()
        # copying X, y
        self.X_ = X.copy()
        self.y_ = y.copy()
        # features attribute
        self.features_ = X.columns
        # create data variable
        self.data_ = pd.concat([self.X_, self.y_], axis=1)
        # variables attribute
        self.variables_ = self.data_.columns
        # dict with samples
        self.samples_ = {i:None for i in self.n_folds_}
        # dcit with results for metrics dataframe
        results_metrics_dict = {i:[] for i in self.models_names_}
        # list with names of fold in format (example: 'fold_0' for 0th fold)
        # fold_names_list = []
        # calculate size of samples
        self.sample_size_ = len(self.data_) * self.sample_frac_
        self.sample_size_ = int(np.round(self.sample_size_))
        # count models
        self.models_count_ = len(self.models_names_)
        # if full results necessary
        if self.full_results_:
            # create array for predictions
            array_size = self.sample_size_ * self.models_count_
            self.predictions_ = np.zeros(shape=(array_size, self.n_folds_))
            # create array for y_true
            self.y_true_full_ = np.zeros(shape=(self.sample_size_, self.n_folds_))
        # for every fold of n_folds folds
        for i in range(0, self.n_folds_):
            # create variable with fold name
            # fold_name = 'fold_{0}'.format(i)
            # append it to fold names list
            # fold_names_list.append(fold_name)
            # create sample
            sample = self.data_.sample(frac=self.sample_frac_, replace=self.replace_)
            # add sample to samples attribute
            self.samples_[i] = sample.values
            # create X and y from sample
            X_ = sample.loc[:, sample.columns!=target].copy()
            y_ = sample[target].copy()
            if self.full_results_:
                # create array with predictions of all models for current folds
                predictions_fold = np.array([])
                # create 'column' in y_true array wtih y_true of current fold
                self.y_true_full_[:, i] = y_
            # for every model
            for model, name in zip(self.models_, self.models_names_):
                # predict y
                y_pred = model.predict(X_)
                # calculate metric for current model in current fold
                metric = mean_squared_error(y_, y_pred, squared=False)
                # add it to metrics dict
                results_metrics_dict[name].append(metric)
                if self.full_results_:
                    # add prediction of current model to array with predictions
                    predictions_fold = np.append(predictions_fold, np.array(y_pred))
            if self.full_results_:
                # create 'column' for current fold in full predictions array
                self.predictions_[:, i] = predictions_fold
        # create df with metrics
        # self.metrics_folds_ = pd.DataFrame(results_metrics_dict, index=fold_names_list)
        self.metrics_folds_ = pd.DataFrame(results_metrics_dict)
        
        if self.full_results_:
            # create dict with full results and predictions
            self.results_dict_ = {
                'sample_size': self.sample_size_,
                'n_folds': self.n_folds_,
                'target_predictions': self.predictions_,
                'target_actual': self.y_true_full_
            }
        
        # calculate CI with bootstrap
        ci_dict = ci_bootstrap(self.metrics_folds_)
        # create df with mean scores and ci
        self.metrics_scores_ = pd.DataFrame(
            data=ci_dict,
            index=self.models_names_
        )
        rename_dict = {
            'ci_min': 'ci_min_mean',
            'ci_max': 'ci_max_mean',
            'statistic': 'rmse'
        }
        self.metrics_scores_ = self.metrics_scores_.rename(columns=rename_dict)
        self.metrics_scores_['ci_min'] = (self.metrics_scores_['rmse']
                                          - 2*self.metrics_scores_['std'])
        self.metrics_scores_['ci_max'] = (self.metrics_scores_['rmse']
                                          + 2*self.metrics_scores_['std'])

        # change columns order
        scores_cols = [
            'rmse', 'std', 'ci_min', 'ci_max',
            'ci_min_mean', 'ci_max_mean'
        ]
        self.metrics_scores_ = self.metrics_scores_[scores_cols].copy()
        # calculate duration
        self.fit_time_sec_ = time.time() - t_fit_st
        self.fit_time_format_ = dt.timedelta(seconds=np.round(self.fit_time_sec_))
        self.fit_time_format_ = str(self.fit_time_format_)

        models_names_str = ','.join(self.models_names_)
        return_label = (
            f'ModelsEvaluator(models=({models_names_str}), '
            f'sample_size={self.sample_size_}, '
            f'n_folds={self.n_folds_})'
            f'\nFit Time: {self.fit_time_format_}'
        )
        return print(return_label)


class HousePricePredictor():

    def __init__(
            self, residuals_estimator,
            predictors, features, target, outliers_scale=1.5):
        
        self.residuals_estimator = residuals_estimator
        self.predictors = predictors
        self.features = features
        self.target = target
        self.outliers_scale = outliers_scale
        self.is_fitted = False


    def fit(self, X, y):
        
        # preparing data for linear regression
        # create attribute with data for linear regression
        self.X_linear = X[self.predictors].copy()
        # create dataset for linear regression
        data_linear = pd.concat([self.X_linear, y], axis=1)
        # create trimmer to cut outliers of target variable
        target_trim = OutlierTrimmer(
            capping_method='iqr',
            tail='both',
            fold=self.outliers_scale,
            variables=self.target)
        # cut outliers of target variable
        data_linear = target_trim.fit_transform(data_linear)

        # create formula
        self.linear_formula = self.target + ' ~ ' + ' + '.join(self.predictors)
        # fit linear regression
        self.linear_estimator = smf.ols(
            formula=self.linear_formula, data=data_linear).fit(cov_type='HC3')

        # preparing data for residuals estimator
        # create attribute with data for residuals prediction
        self.X_residuals = X[self.features].copy()
        # create dataset for residuals prediction
        X_residuals_fit = X[self.features].copy()
        # calculate preditions of linear model
        y_pred_linear = self.linear_estimator.predict(self.X_linear)
        # calculate residuals
        residuals = y - y_pred_linear
        # transform predictions of linear model to use them as predictor
        y_pred_linear = y_pred_linear.values.reshape(-1, 1)
        self.scaler_target = StandardScaler()
        y_pred_linear = self.scaler_target.fit_transform(y_pred_linear)
        # add predictions of linear model as new predictor
        X_residuals_fit['y_pred_linear'] = y_pred_linear
        
        # fit residuals estimator
        self.residuals_estimator.fit(X_residuals_fit, residuals)

        # mark HHP as fitted
        self.is_fitted = True


    def check_fitted(self):
        if self.__sklearn_is_fitted__:
            print('Estimator is fitted')
        else:
            print('Estimator not fitted')

    
    def get_params(self, deep=True):
        parameters = {
            'linear_estimator_parameters': self.linear_estimator.params,
            'residuals_estimator_parameters': self.residuals_estimator.get_params()
        }
        return 

    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    def __sklearn_clone__(self):
        return self

    def __sklearn_is_fitted__(self):
        if self.is_fitted:
            return True
        else:
            return False
        
    
    def get_linear_predictions(self, X):
        if self.__sklearn_is_fitted__:
            X_linear = X[self.predictors].copy()
            predictions_linear = self.linear_estimator.predict(X_linear)
            return predictions_linear
        else:
            print('Estimator not fitted')

    
    def get_residuals_predictions(self, X):
        if self.__sklearn_is_fitted__:
            # linear predictons
            predictions_linear = self.get_linear_predictions(X)
            predictions_linear = predictions_linear.values.reshape(-1, 1)
            predictions_linear = self.scaler_target.transform(predictions_linear)
            # resuduals predictions
            X_residuals = X[self.features].copy()
            X_residuals['y_pred_linear'] = predictions_linear
            predictions_residuals = self.residuals_estimator.predict(X_residuals)
            predictions_residuals = pd.Series(
                data=predictions_residuals,
                index=X_residuals.index
            )
            return predictions_residuals
        else:
            print('Estimator not fitted')

    
    def predict(self, X):
        
        if self.__sklearn_is_fitted__:
            # predict
            predictions_linear = self.get_linear_predictions(X)
            predictions_residuals = self.get_residuals_predictions(X)
            predictions = predictions_linear + predictions_residuals
            return predictions
        else:
            print('Estimator not fitted')


class Cleaner():

    def __init__(self, idxs_drop):
        self.idxs_drop = idxs_drop


    def fit(self, X):
        pass


    def transform(self, X):
        X_ = X.copy()
        X_ = X_.drop(self.idxs_drop, axis=0)
        masvnrarea_nan_idxs = [529, 1243, 936, 973, 650, 1278, 234, 977]
        X_.loc[masvnrarea_nan_idxs, 'masvnrarea'] = 0
        X_.loc[X_.index.isin([773, 1230]), 'masvnrarea'] = 0
        X_.loc[1241, 'masvnrarea'] = np.round(200, 1)
        X_.loc[688, 'masvnrarea'] = np.round(196, 1)
        X_.loc[X_.index.isin([624, 1334]), 'masvnrarea'] = 0
        X_.loc[X_.index.isin([1300]), 'masvnrtype'] = 'Stone'
        
        return X_


    def fit_transform(self, X, y):
        X_ = X.copy()
        X_ = X_.drop(self.idxs_drop, axis=0)
        masvnrarea_nan_idxs = [529, 1243, 936, 973, 650, 1278, 234, 977]
        X_.loc[masvnrarea_nan_idxs, 'masvnrarea'] = 0
        X_.loc[X_.index.isin([773, 1230]), 'masvnrarea'] = 0
        X_.loc[1241, 'masvnrarea'] = np.round(200, 1)
        X_.loc[688, 'masvnrarea'] = np.round(196, 1)
        X_.loc[X_.index.isin([624, 1334]), 'masvnrarea'] = 0
        X_.loc[X_.index.isin([1300]), 'masvnrtype'] = 'Stone'
        
        return X_



