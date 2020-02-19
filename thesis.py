########################## IMPORTS ###############################################################

from typing import List

import pandas as pd
from pandas import DataFrame
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_learning_curves
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import itertools
import random


import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.metrics import classification_report_imbalanced
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import figure

from sklearn.model_selection import TimeSeriesSplit

################################################# END OF IMPORTS#######################################################


df = pd.read_excel("C:\\Users\\feret\\Desktop\\dilomatiki\\Data_Recession.xlsx")  # import arxika data
print(df.head())
# print(df.shape) # 657 rows - 17 columns


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

for i in df:
    print(i)

y = df.iloc[:, 1]
# print(y) # series NBER-RECESSION

X = df.iloc[:, 2:]
# print(X) #dataframe

pd.options.mode.chained_assignment = None  # stops warning

df_2 = df[["Year", "NBER-Recession"]]
#print(df_2.head())


for column in X:
    df_2["prev_" + column] = df[column].shift(12)


df_2.dropna(inplace=True)
# print(df_2["Year"]) # neo dataframe me y kai Xt-12

X.rename(columns={"prev_YC 10y-3m (%)": "YC 10y-3m", "prev_Rate of unemployment (%)": "Rate_of_Ump", "prev_NAIRU (%)": "NAIRU",
                  "prev_Unemployment Gap (%)":"Ump_gap", "prev_Real Effective Federal Funds rate (%)": "Real_Eff_FundsRate",
                  "prev_Moodys BAA Yield (%)": "MoodyBAA_Yield",
                  "prev_Moodys BAA Yield over 10-year Treasury Yield (Credit Spread)": "Credit_Spread",
                  "prev_Leading Indicator": "Leading_Indicator", "prev_Real Money Supply M1 (%)": "Real_MoneySup_M1",
                  "prev_Real Money Supply M2 (%)": "Real_MonSup_M2", "prev_Non-Farm Payrolls (#)": "NonFarm_Payrolls",
                  "prev_S&P 500 Index monthly return (%)": "S&P 500",
                  "prev_3-month Commercial Paper - 3-month T-bill spread (%) Money Market Spread": "3-month_Comm_Paper",
                  "prev_GDI monthly (%)": "GDI_monthly", "prev_GDP monthly (%)": "GDP_monthly"}, inplace=True)



names_X=[]

# for i in X:
#    names_X.append(i)
# print(names_X)

def summary_model(x, y):
    x = sm.add_constant(x)
    logit_model = sm.Logit(y, x)
    result = logit_model.fit()
    print(result.summary())


#----------------------------------------------------
#####################################15 logit models each one with one variable:##################################################

# for i in range(0, 15):
#     print(summary_model(X.iloc[:, i], y))
#----------------------------------------------------

#X2: df which contains the statistical important independent variables

X2 = X.drop({'Rate_of_Ump', 'Ump_gap', 'MoodyBAA_Yield', 'S&P 500',
             'GDP_monthly', 'NonFarm_Payrolls'}, axis=1)

print(summary_model(X2, y))

X2_corr = X2.corr()
#print(X2.corr())

plt.figure(figsize=(10, 10))
sns.heatmap(X2_corr, xticklabels=X2_corr.columns.values, yticklabels=X2_corr.columns.values)
#plt.show()

#print(summary_model(X2,y))

X3 = X2.drop({'Credit_Spread', 'Leading_Indicator','Real_MoneySup_M1', 'Real_MonSup_M2'}, axis= 1)

#print(summary_model(X3,y))

X4 = X3.drop({'3-month_Comm_Paper'}, axis=1)
#print(summary_model(X4,y))

#---------------------STEPWISE REGRESSION--------------------------------

for i in X4:
    print(i)
X4 = X[["YC 10y-3m", "NAIRU", "Real_Eff_FundsRate", "GDI_monthly"]]

for i in range(1, 5):
    print(summary_model(X4.iloc[:, 0:i], y))
    i += 1


logistic_model = LogisticRegression()
logistic_model.fit(X4, y)
predictions = logistic_model.predict(X4)
# the probability of being y=1
prob = logistic_model.predict_proba(X4)[:, 1]
probability = [1 if i > 0.40 else 0 for i in prob]
# print(probability)
# print(predictions)
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic_model.score(X4, y)))
conf_mat = confusion_matrix(y, predictions)
#print(conf_mat)
#print(classification_report(y, predictions, digits=3))


#--------------------------TRAIN SPLIT DATA-------------------------------

tss = TimeSeriesSplit(n_splits=3)   ##### n-splits determine the number folds. N-1 in this case we split our # data into two sets. (3-1)

y_year = df_2.iloc[:, 0]
#print(y_year.head())
train_size = int(len(X) * 0.70)
X_train, X_test = X4[0:train_size], X4[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]
y_year_train, y_year_test = y_year[0:train_size], y_year[train_size:len(y_year)]

#print(y_year_train)
#print('Observations: %d' % (len(X4))) #645 obs
#print('Training Observations: %d' % (len(X_train))) #451 of 645 obs srart : 1965-06-01 until: 2002-12-01 70% training set
#print('Testing Observations: %d' % (len(X_test))) #194 of 645 obs start:  01-01-2003 : 01-02-2019

probability_data = pd.DataFrame()


class prediction():

    def __init__(self, xtrain, ytrain, xtest, ytest):
        self.xtrain = X_train
        self.ytrain = y_train
        self.xtest = X_test
        self.ytest = y_test

    def classification_report(self):
        logit_model = LogisticRegression()
        logit_model.fit(self.xtrain, self.ytrain)
        pred = logit_model.fit(self.xtrain, self.ytrain).predict(self.xtest)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            logistic_model.score(self.xtest, self.ytest)))
        print(logit_model.score(self.xtest, self.ytest))
        print(classification_report(self.ytest, pred, digits=3))
        con_mat = confusion_matrix(self.ytest, pred)
        print(con_mat)
        print("Recall score: {}".format(recall_score(self.ytest, pred)))
        print("Precision score: {}".format(precision_score(self.ytest, pred)))
        print("F1 Score: {}".format(f1_score(self.ytest, pred)))

        model_prob = logit_model.predict_proba(self.xtest)[:, 1]
        model_probability = [1 if i > 0.40 else 0 for i in model_prob]
        #return probability
        probability_data["recession_probability"] = np.array(model_probability)

    def ROC(self):
        roc_auc = roc_auc_score(self.ytest, logistic_model.predict(self.xtest))
        fpr, tpr, thresholds = roc_curve(self.ytest, logistic_model.predict_proba(self.xtest)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()

r = prediction(X_train, y_train, X_test, y_test)

print(r.classification_report())
#print(y_test)
print(probability_data)
#print(r.ROC())


#######################################################CHARTS##########################################



df_2.to_csv(r'C:\\Users\\feret\\Desktop\\df_last.csv', index=False)
probability_data.to_csv(r'C:\\Users\\feret\\Desktop\\probability_data.csv', index=False)

# fig = go.Figure()
#
#
# fig.add_trace(go.Scatter(
#     x=df.Year,
#     y=df["YC 10y-3m (%)"],
#     name="YC 10y-3m",
#     line_color='firebrick',
#     opacity=0.8))
#
# fig.add_trace(go.Scatter(
#      x=df.Year,
#      y=probability_data['recession_probability'],
#      name="recession_probability",
#      line_color="orange",
#      opacity=0.8))
#
#
# # Use date string to set xaxis range
# fig.update_layout(xaxis_range=['2003-01-01', '2019-02-28'])
#
# fig.show()
#
#
#
# fig.add_trace(go.Scatter(
#     x=df_last.Year,
#     y=df_last["GDI monthly (%)"],
#     name="GDI_monthly",
#     line_color='purple',
#     opacity=0.8))
#
#
#
#
# #fig.add_trace(go.Scatter(
#  #   x=last_data_test.Year,
#   #  y=last_data_test["Real_Eff_FundsRate"],
#    # name="Real Effective Federal Funds rate (%)",
#     #line_color='black',
#
# #   opacity=0.8))
# #
#
# #fig.add_trace(go.Scatter(
#   #  x=last_data_test.Year,
#    # y=last_data_test['NAIRU'],
#     #name="NAIRU (%)",
#     #line_color='mediumblue',
#     #opacity=0.8))




