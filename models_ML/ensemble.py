from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics

import numpy as np 
import pandas as pd 
import seaborn as sns
importcsv

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                print(0)
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

dataset = pd.read_csv('/content/gdrive/MyDrive/bp/cleaned_further.csv',names = ['alpha','PIR', 'ptt', 'bpmax' ,'bpmin', 'hrfinal', 'ih', 'il', 'meu', 'j', 'k','l','m','n','o','p','q','r','s'])

X = dataset[['alpha','PIR', 'ptt','hrfinal', 'ih', 'il', 'meu', 'j', 'k','l','m','n','o','p','q','r','s']]

y = dataset[['bpmin','bpmax']]

sbp = list()
dbp = list()
real_BP = list()
with open('/content/gdrive/MyDrive/bp/cleaned_further.csv', 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter = ',')
	for row in csv_reader:
		#ptt.append(float(row[2]))
		sbp.append(float(row[3]))
		dbp.append(float(row[4]))

	real_BP = list()
	for i in range(len(sbp)):
		BP_actual = (2*dbp[i] + sbp[i])/3
		real_BP.append(BP_actual)

y = np.array(real_BP)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

rf = RandomForestRegressor()
knn = KNeighborsRegressor()
ridge = Ridge()
svr = SVR()
xgb = XGBRegressor()
lg = LinearRegression()
adb = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=np.random.RandomState(1))

models = [rf, knn, ridge, svr, xgb]

for model in models:
    model.fit(X_train, y_train)

stacked_averaged_models = StackingAveragedModels(base_models = (knn, svr, xgb, ridge),
                                                 meta_model = rf)

stacked_averaged_models.fit(X_train, y_train)


y_pred = stacked_averaged_models.predict(X_train)

print('Train - Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))  
print('Train - Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))  
print('Train - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


y_pred = stacked_averaged_models.predict(X_test)


print('Test - Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Test - Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Test - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
