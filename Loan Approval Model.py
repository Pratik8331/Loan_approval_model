# LOGISTIC REGRESSION

import pandas as pd

data = pd.read_csv(r"D:\Trainings\Courses\CTTC\AIML\Datasets\Churn_Modelling Data.csv")

data2 = data.copy() # for reference and safety

# data cleaning 

data.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)
data.isnull().sum().sum()

# label Encoding
from sklearn.preprocessing import LabelEncoder

# instance of label encoder
le1 = LabelEncoder()
le2 = LabelEncoder()
data.Gender = le1.fit_transform(data.Gender)
data.Geography = le2.fit_transform(data.Geography)

ip = data.drop(['Exited','Tenure','NumOfProducts','CreditScore','HasCrCard',
                'IsActiveMember'],axis=1)
op = data['Exited']



# OneHotEncoding - it differentiates, not rank
from sklearn.preprocessing import OneHotEncoder

# instance of label encoder
#which column contains categorical features
ohe = OneHotEncoder() 
ip = ohe.fit_transform(ip).toarray()

# Scaling the data
from sklearn.model_selection import train_test_split as tts
xtr,xts,ytr,yts = tts(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)

from sklearn.linear_model import LogisticRegression
alg = LogisticRegression()
alg.fit(xtr,ytr)
pred  = alg.predict(xts)
alg.score(xts,yts) # accuracy

# Confusion Matrix - tells how my algorithm is performing
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yts,pred)
print(cm)


corr = data.corr()

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.show()

# accuracy and recall

accuracy = alg.score(xts,yts)
print(accuracy)

from sklearn import metrics
recall = metrics.recall_score(yts,pred)
print(recall)