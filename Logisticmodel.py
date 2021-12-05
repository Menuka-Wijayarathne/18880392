# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:04:09 2021

@author: Menuka_08214
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data=pd.read_csv("input2.csv")

X = data.loc[:, data.columns != 'y']
y = data.loc[:, data.columns == 'y']

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
# print("length of oversampled data is ",len(os_data_X))
# print("Number of disconnected customers in oversampled data",len(os_data_y[os_data_y['y']==0]))
# print("Number of connected customers",len(os_data_y[os_data_y['y']==1]))
# print("Proportion of disconnections in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
# print("Proportion of connections in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#Recursive Feature Elimination
data_final_vars=data.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg,n_features_to_select=10, step=1)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)

#Element disqualified features and choose qualified features for training data set
cols=['ONT','RX','ONTperport','Thirdpartyontperport','OriginalOntperport','IPTV','VOICE','HIS','TotalServices','MutipleService'] 
X=os_data_X[cols]
y=os_data_y['y']

#observing the p value of the features 
import statsmodels.api as sm
# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary2())

#observing the p value of the features after removing the feature where p values >0.05

cols1=['ONT','RX','ONTperport','Thirdpartyontperport','OriginalOntperport','MutipleService'] 
X=os_data_X[cols1]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
#print(result.summary2())


#Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
