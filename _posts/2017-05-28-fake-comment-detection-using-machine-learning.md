---
layout: post
title: Fake Comment Detection Using Machine Learning 
subtitle: Used Simple Classifiers SVM, RandomForests, LogsiticRegression

---
In this project, we use Support Vector Machine classifier to classify the fake comments using the dataset given by the client. You can download the dataset from the following [link](https://github.com/farooqkhadim/Fake-Comment-Detection-using-ML/data).

Steps:

<ul>
  <li>Preprocessing</li>
  <li>Model Training</li>
  <li>Model Prediction</li>
</ul>

Here is some of the code:

## Important Dependencies to Import

~~~
importsys import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import Imputer 
from sklearn import cross_validation 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
%matplotlib inline
~~~

## Function for Dataset Reading

~~~
def read_datasets():  
  genuine_comments = pd.read_csv("data/not_fake_comments.csv") 
  fake_comments = pd.read_csv("data/fake_comments.csv")
  #print genuine_comments.columns
  #print genuine_comments.describe()
  #print fake_comments.describe()
  x=pd.concat([genuine_comments,fake_comments]) y=len(fake_comments)*[0] + len(genuine_comments)*[1] 
  return x,y
~~~

## Function for Feature Engineering

~~~
defextract_features(x):
lang_list = list(enumerate(np.unique(x['lang'])))
lang_dict = { name : i for i, name in lang_list }
x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(i nt)
x.loc[:,'sex_code']=predict_sex(x['name'])
feature_columns_to_use = ['statuses_count','followers_count','friend           s_count','favourites_count','listed_count','sex_code','lang_code']
x=x.loc[:,feature_columns_to_use] 
return x
~~~

## Function for plotting confusion matrix

~~~
defplot_confusion_matrix(cm,title='Confusionmatrix',cmap=plt.cm.Blue s):
target_names=['Fake','Genuine']
plt.imshow(cm, interpolation='nearest', cmap=cmap) plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(target_names)) plt.xticks(tick_marks, target_names, rotation=45) 
plt.yticks(tick_marks, target_names) 
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
~~~

## Function for training data using Support Vector Machine

~~~
def train(X_train,y_train,X_test):
    """ Trains and predicts dataset with a SVM classifier """ # Scaling features
    X_train=preprocessing.scale(X_train) 
    X_test=preprocessing.scale(X_test)
    Cs = 10.0 ** np.arange(-2,3,.5)
    gammas = 10.0 ** np.arange(-2,3,.5)
    param = [{'gamma': gammas, 'C': Cs}]
    cvk = StratifiedKFold(y_train,n_folds=5)
    classifier = SVC()
    clf = GridSearchCV(classifier,param_grid=param,cv=cvk)
    clf.fit(X_train,y_train)
    print("The best classifier is: ",clf.best_estimator_) 
    clf.best_estimator_.fit(X_train,y_train)
    # Estimate score
    scores = cross_validation.cross_val_score(clf.best_estimator_, X_train,y_train, cv=5)
    print (scores)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    title = 'Learning Curves (SVM, rbf kernel, $\gamma=%.6f$)' %clf.best_estimator_.gamma
    plot_learning_curve(clf.best_estimator_, title, X_train, y_train, cv=5)
    plt.show()
    # Predict class
    y_pred = clf.best_estimator_.predict(X_test) 
    return y_test,y_pred
~~~

## Splitting datasets in train and test dataset

~~~
 X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, ran dom_state=44)
~~~

**Classificatin Accuracy: 0.904255319149**

**Confusion Matrix without normalization**

~~~
[[262 6]
 [48 248]]
~~~

**Confusion Matrix with normalization**

~~~
[[0.97761194 0.02238806]
 [0.16216216 0.83783784]]
~~~

The complete code is available at the following [link](https://github.com/farooqkhadim/Fake-Comment-Detection-using-ML) .
Thanks for reading.
