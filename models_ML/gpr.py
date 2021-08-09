import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, DotProduct, RationalQuadratic, Exponentiation, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

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

kernel = 6.92**2 * RationalQuadratic(length_scale= 133.2, alpha = 4.64 ) + WhiteKernel(noise_level= 40.3)
gp = GaussianProcessRegressor(kernel=kernel, alpha=5e-9, optimizer='fmin_l_bfgs_b', 
                                n_restarts_optimizer=0, normalize_y=False, copy_X_train=True,
                                random_state=2016)

y_log = np.log1p(y_train)
clf = Pipeline([('scaler', StandardScaler()), ('gp', gp)]) 
y_log_centered = y_log - y_log.mean()
y = np.expm1(y_log)


clf.fit(X_train,y_log_centered)
prediction = clf.predict(X_test)
prediction1 = clf.predict(X_train)
prediction = np.expm1(prediction + y_log.mean())
prediction1 = np.expm1(prediction1 + y_log.mean())
y_pred = prediction
y_pred1 = prediction1

print('Train - Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred1))  
print('Train - Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred1))  
print('Train - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred1)))

print('Test - Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Test - Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Test - Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))