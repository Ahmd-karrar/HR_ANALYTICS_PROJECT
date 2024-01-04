# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 04:53:29 2023

@author: ahmed.abdelmonim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score,roc_curve,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
df = pd.read_csv("C:/Users/ahmed.abdelmonim/HR_1.csv")

# choose relevant columns
df.columns

df_model = df[[ 'DistanceFromHome', 'Education', 'EducationField',
       'Gender', 'MaritalStatus', 'Attrition', 'Department_x',
        'PercentSalaryHike',
       'StockOptionLevel', 'BusinessTravel',  'JobLevel',
       'JobRole', 'JobSatisfaction', 'NumCompaniesWorked', 'PerformanceRating',
        'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager', 'OverTime',
       'Age_group', 'Income_group']]

# get dummy data
df_model['Attrition'] = df_model['Attrition'].map({'Yes':1 , 'No': 0})
df_model['Attrition'] = df_model['Attrition'].astype(int)

df_model[['Education','StockOptionLevel','JobLevel','JobSatisfaction','PerformanceRating',
                      'TrainingTimesLastYear','WorkLifeBalance']] = df_model[['Education','StockOptionLevel','JobLevel','JobSatisfaction','PerformanceRating',
                      'TrainingTimesLastYear','WorkLifeBalance']].astype('category')
df_model.dtypes
df_model1= pd.get_dummies(df_model)
df_model1.head()

# Train and test split
x = df_model1.loc[:,df_model1.columns != 'Attrition']
y = df_model1.loc[:,'Attrition']


x_train, x_test , y_train , y_test = train_test_split(x,y, test_size=0.30 , random_state= 999)

# Buld NB Model
NBmodel= GaussianNB()
NBmodel.fit(x_train,y_train)    

NB_predprob = NBmodel.predict_proba(x_test)
NB_predprob

#cutoff
cutoff = 0.5
NBpred_test = np.where(NB_predprob[:,1] > cutoff , 1,0)

# Confustion  Matrix

confusion_matrix(y_test, NBpred_test,labels= [0,1]) 

accuracy_score(y_test, NBpred_test)
precision_score(y_test, NBpred_test)
recall_score(y_test, NBpred_test)

print(classification_report(y_test, NBpred_test))
# Area Under Roc Curve
auc = roc_auc_score(y_test, NB_predprob[:,1])
print('AUC: %.3f' % auc)

# ROC Curve 

NBfpr ,NBtpr,thresholds = roc_curve(y_test,NB_predprob[:,1])

# Find the optimal threshold index that maximizes the difference between true positive rate (tpr) and false positive rate (fpr).
optimal_idx = np.argmax(NBtpr - NBfpr)
# Get the corresponding optimal threshold from the thresholds array.
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
# plot the roc curve for the model 
plt.figure()
lw = 2
plt.plot(NBfpr,NBtpr, color = 'darkorange',lw = lw ,label = 'ROC Curve (area = %0.3f)' % auc)
plt.plot([0,1],[0,1], color = 'navy', lw = lw , linestyle = '--')
plt.axes('tight')
plt.xlabel('False Postive Rate ');plt.ylabel('True Positive Rate')
plt.title('Recever operating characteristic')
plt.legnd(loc ="lower right")
plt.show()
        
# Decision Tree
DTmodel = DecisionTreeClassifier(criterion= 'entropy', min_samples_split= int(len(x_train)*.10))
DTmodel.fit(x_train,y_train)
DT_pred = DTmodel.predict(x_test)
DT_pred_probs = DTmodel.predict_proba(x_test)

cutoff = 0.5
dtmodel_pred = np.where(DT_pred_probs[:,1] > cutoff,1,0)


# Confustion  Matrix

confusion_matrix(y_test, dtmodel_pred,labels= [0,1]) 

accuracy_score(y_test, dtmodel_pred)
precision_score(y_test,dtmodel_pred)
recall_score(y_test,dtmodel_pred )

print(classification_report(y_test, DT_pred))
# Area Under Roc Curvedtmodel_pred
auc = roc_auc_score(y_test, DT_pred_probs[:,1])
print('AUC: %.3f' % auc)

# ROC Curve 

DTfpr ,DTtpr,thresholds = roc_curve(y_test,DT_pred_probs[:,1])

# plot the roc curve for the model 
plt.figure()
lw = 2
plt.plot(DTfpr,DTtpr, color = 'darkorange',lw = lw ,label = 'ROC Curve (area = %0.3f)' % auc)
plt.plot([0,1],[0,1], color = 'navy', lw = lw , linestyle = '--')
plt.axes('tight')
plt.xlabel('False Postive Rate ');plt.ylabel('True Positive Rate')
plt.title('Recever operating characteristic')
plt.legnd(loc ="lower right")
plt.show()





# Build Random Forest Model

rf = RandomForestClassifier(random_state= 999 , n_estimators = 100 , oob_score= True , max_features= 'sqrt')

rf.fit(x_train,y_train)

# calculating prediction for the model

y_pred = rf.predict(x_test)
y_pred_probs = rf.predict_proba(x_test)

cutoff = 0.5
pred_test = np.where(y_pred_probs[:,1] > cutoff , 1,0)

pred_test 

# Confustion  Matrix
confusion_matrix(y_test, pred_test,labels= [0,1]) 

accuracy_score(y_test, pred_test)

recall_score(y_test, pred_test)

print(classification_report(y_test,y_pred ))

# Area Under Roc Curve

auc = roc_auc_score(y_test, y_pred_probs[:,1])
print('AUC: %.3f' % auc)

# OOB Score
rf.oob_score_

rf.feature_importances_


# ROC Curve 

RFfpr ,RFtpr,thresholds = roc_curve(y_test,y_pred_probs[:,1])

# plot the roc curve for the model 
plt.figure()
lw = 2
plt.plot(RFfpr,RFtpr, color = 'darkorange',lw = lw ,label = 'ROC Curve (area = %0.3f)' % auc)
plt.plot([0,1],[0,1], color = 'navy', lw = lw , linestyle = '--')
plt.axes('tight')
plt.xlabel('False Postive Rate ');plt.ylabel('True Positive Rate')
plt.title('Recever operating characteristic')
plt.legnd(loc ="lower right")
plt.show()

# Importance Matrix 

features = list(x.columns)
importances = rf.feature_importances_
indices = np.argsort(importances)
num_features = 14
plt.figure(figsize=(100,100))
plt.title('Feature Importances')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.barh(range(num_features), importances[indices[-num_features:]], color='b', align='center')
plt.yticks(range(num_features),[features[i] for i in indices[-num_features:]])
plt.xlabel('Relative Importance')
plt.show();




model_params = {
    'svm': {
        'model': GaussianNB(),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}


























