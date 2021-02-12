#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,roc_auc_score
################For plotting the graph##################
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import numpy as np


# In[83]:


print ("Loading the dataset, please wait")
data = pd.read_csv('winequality.csv')
print ("Cleaning the dataset and generating the Selected column")


# In[84]:


X = data.drop(['quality'],axis=1)


# In[85]:


data.info()


# In[86]:


X


# In[87]:


data


# In[88]:


Y = data['quality']


# In[89]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


for col in X.columns: 
    print(col) 


# In[91]:


col = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']


# In[92]:


from sklearn.preprocessing import MinMaxScaler


# In[93]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[94]:


for i in col:
    X[i] = scaler.fit_transform(X[i].values.reshape(-1, 1))


# In[95]:


X


# In[96]:


Y


# In[97]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=123)


# # Linear Regression

# In[98]:


lm = LinearRegression()
model = lm.fit(X_train, Y_train)


# In[99]:


Y_pred = model.predict(X_test)


# In[100]:


#To retrieve the intercept:
print(model.intercept_)
#For retrieving the slope:
print(model.coef_)


# In[101]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[104]:



accuracy = model.score(X_test,Y_test)
print(accuracy*100,'%')


# # Logistic Regression

# In[105]:


#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[106]:


logreg = LogisticRegression(C=1e10)
fit = logreg.fit(X_train, Y_train)
Y_pred_LR = logreg.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred_LR)
print('accuracy',accuracy)
print("Precision Score : ",precision_score(Y_test, Y_pred_LR, 
                                           pos_label='positive',
                                           average='weighted'))
print("Recall Score : ",recall_score(Y_test, Y_pred_LR, 
                                           pos_label='positive',
                                           average='weighted'))
f1=f1_score(Y_test, Y_pred_LR,pos_label='positive',
                                           average='weighted')
print ('f1 =',f1)


# # Lasso

# In[107]:


from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#Initializing the Lasso Regressor with Normalization Factor as True
lasso_reg = Lasso(alpha = 0.25)

#Fitting the Training data to the Lasso regressor
lasso_reg.fit(X_train,Y_train)

#Predicting for X_test
Y_pred_lass =lasso_reg.predict(X_test)

#Printing the Score with RMLSE
#a=accuracy_score(Y_test,y_pred_lass)
from sklearn.metrics import mean_squared_error

k=mean_squared_error(Y_test, Y_pred_lass)
k


# # Ridge

# In[108]:


from sklearn.linear_model import Ridge
import numpy as np

clf = Ridge(alpha=1.265464)
clf.fit(X_train, Y_train)
Ridge()


# In[109]:


pred_train_rr= clf.predict(X_train)
print(np.sqrt(mean_squared_error(Y_train,pred_train_rr)))
print(r2_score(Y_train, pred_train_rr))

pred_test_rr= clf.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test,pred_test_rr))) 
print(r2_score(Y_test, pred_test_rr))


# In[110]:


model_enet = ElasticNet(alpha = 0.025691)
model_enet.fit(X_train, Y_train) 
pred_train_enet= model_enet.predict(X_train)
print(np.sqrt(mean_squared_error(Y_train,pred_train_enet)))
print(r2_score(Y_train, pred_train_enet))

pred_test_enet= model_enet.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test,pred_test_enet)))
print(r2_score(Y_test, pred_test_enet))


# In[111]:


model_lasso = Lasso(alpha=0.025691)
model_lasso.fit(X_train, Y_train) 
pred_train_lasso= model_lasso.predict(X_train)
print(np.sqrt(mean_squared_error(Y_train,pred_train_lasso)))
print(r2_score(Y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print(np.sqrt(mean_squared_error(Y_test,pred_test_lasso))) 
print(r2_score(Y_test, pred_test_lasso))


# # ElasticNet

# In[112]:


from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression


# In[113]:


X, Y = make_regression(n_features=2, random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(X, Y)
ElasticNet(random_state=0)
print(regr.coef_)



# In[114]:


print(regr.intercept_)

print(regr.predict([[0, 0]]))


# In[115]:


data.plot(x='fixed acidity', y='quality', style='o')  
plt.title('fixed acidity VS quality')  
plt.xlabel('fixed acidity')  
plt.ylabel('quality')  
plt.show()


# In[116]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(data['quality'])


# In[117]:


x = data['fixed acidity'].values.reshape(-1,1)
y = data['quality'].values.reshape(-1,1)


# In[118]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[119]:


regressor = LinearRegression()  
regressor.fit(x_train, y_train)


# In[120]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[121]:


y_pred = regressor.predict(x_test)


# In[122]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[123]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[124]:


plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()


# In[125]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Decision Tree

# In[126]:


from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score




tree1 = tree.DecisionTreeClassifier()

tree1 = tree1.fit(X_train, Y_train)

print('the accuracy rate on training data is', tree1.score(X_train, Y_train))

print('the accuracy rate on testing data is', tree1.score(X_test, Y_test))




Y_pred_DT = tree1.predict(X_test)

print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))


# In[127]:


print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))



# # knn 

# In[128]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)


# In[3]:


y_pred_knn = classifier.predict(X_test)


print('score = ',classifier.score(X_test, Y_test))



# In[1162]:


accuracy = accuracy_score(Y_test,y_pred_knn)
print('accuracy',accuracy)


# In[1163]:


print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))
print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))


# # SVM

# In[129]:


from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

lsvm = svm.SVC(kernel='linear', C=1)
lsvm.fit(X_train,Y_train)

print("training classification rate = ", lsvm.score(X_train, Y_train))
print("testing classification rate = ", lsvm.score(X_test, Y_test))

Y_pred_svm = lsvm.predict(X_test)

Y_pred_svm

df = pd.DataFrame(Y_pred_svm, columns = ['quality'])

export_csv = df.to_csv (r'svemo.csv', index = None, header=True) 

accuracy = accuracy_score(Y_test, Y_pred_svm)
print('Accuracy = ', accuracy)

print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))
print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))



# # Polynomial SVM

# In[130]:


psvm = svm.SVC(kernel='poly', degree=2)
psvm.fit(X_train, Y_train)
print("training classification rate = ", psvm.score(X_train, Y_train))
print("testing classification rate = ", psvm.score(X_test, Y_test))


# In[131]:


Y_pred_svm = psvm.predict(X_test)

Y_pred_svm

df = pd.DataFrame(Y_pred_svm, columns = ['quality'])

export_csv = df.to_csv (r'svemo.csv', index = None, header=True) 

accuracy = accuracy_score(Y_test, Y_pred_svm)
print('Accuracy = ', accuracy)

print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))
print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))


# # Random Forest

# In[132]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
forest = RandomForestClassifier(n_estimators=1, max_depth=100, random_state=345)

forest = forest.fit(X_train, Y_train)
print("training data accuracy = ",forest.score(X_train, Y_train))
print("testing data accuracy = ",forest.score(X_test, Y_test))

Y_pred_RF = forest.predict(X_test)
df = pd.DataFrame(Y_pred_RF, columns = ['quality'])

export_csv = df.to_csv (r'RaFo.csv', index = None, header=True) 
print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))
print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))


# # Bagging

# In[133]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier


dtc = DecisionTreeClassifier()
bag_model=BaggingClassifier()
bag_model=bag_model.fit(X_train,Y_train)

ytest_pred=bag_model.predict(X_test)

df = pd.DataFrame(ytest_pred, columns = ['quality'])

export_csv = df.to_csv (r'BC.csv', index = None, header=True) 

print(bag_model.score(X_test, Y_test))

print(confusion_matrix(Y_test, ytest_pred)) 

accuracy = accuracy_score(Y_test, ytest_pred)
print('accuracy',accuracy)
print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))
print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))


# # Gradient Boosting

# In[134]:


clf_GB = GradientBoostingClassifier()
clf_GB.fit(X_train, Y_train)

predictions_GB = clf_GB.predict(X_test)
df = pd.DataFrame(predictions_GB, columns = ['quality'])

export_csv = df.to_csv (r'GB.csv', index = None, header=True) 

accuracy = accuracy_score(Y_test, predictions_GB)
print('accuracy',accuracy)
print("Precision Score : ",precision_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))

print("Recall Score : ",recall_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))
print("f1 Score : ",f1_score(Y_test, Y_pred_DT, 
                                           pos_label='positive',
                                           average='micro'))


# In[135]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
n_groups = 5
means_frank = (56.12,57,41.23,30.9,18.6)
means_guido = (56.74,57.6,68.9,70.7,64.8)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,alpha=opacity,color='green',label='Precision')
rects2 = plt.bar(index + bar_width, means_guido, bar_width,alpha=opacity,color='orange',
                 label='Recall')

plt.xlabel('Methods')
plt.ylabel('Scores')
plt.title('Score')
plt.xticks(index + bar_width, ('Linear ', 'Logistic', 'Lasso', 'Ridge', 'ElasticNET'))
plt.legend()

plt.tight_layout()
plt.show()


# In[136]:


import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 5
means_frank = (33.0,57.6,63.9,70.0,64.5)
means_guido = (54.74,55.6,60.9,43.7,64.08)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Accuracy')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='pink',
label='f1')

plt.xlabel('Methods')
plt.ylabel('Scores')
plt.title('Scores')
plt.xticks(index + bar_width, ('Linear ', 'Logistic', 'Lasso', 'Ridge', 'ElasticNET'))
plt.legend()

plt.tight_layout()
plt.show()


# In[137]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
n_groups = 7
means_frank = (56.12,56.0,59.23,64.9,50.6,60.0,63.0)
means_guido = (100,80,56,57,81,43,56)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,alpha=opacity,color='red',label='Test accuracy')
rects2 = plt.bar(index + bar_width, means_guido, bar_width,alpha=opacity,color='grey',
                 label='Train Accuracy')

plt.xlabel('Methods')
plt.ylabel('Scores')
plt.title('Score')
plt.xticks(index + bar_width, ('DT', 'KNN', 'LSVM', 'PSVM', 'RF','BG','XGB'))
plt.legend()

plt.tight_layout()
plt.show()




# In[138]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
n_groups = 7
means_frank = (56.12,56.0,59.93,66.9,40.6,45.0,63.0)
means_guido = (56,80,68,60,71,69,56)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,alpha=opacity,color='gold',label='Precision')
rects2 = plt.bar(index + bar_width, means_guido, bar_width,alpha=opacity,color='orange',
                 label='f1')

plt.xlabel('Methods')
plt.ylabel('Scores')
plt.title('Score')
plt.xticks(index + bar_width, ('DT', 'KNN', 'LSVM', 'PSVM', 'RF','BG','XGB'))
plt.legend()

plt.tight_layout()
plt.show()



# In[ ]:




