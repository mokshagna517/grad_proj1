#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[2]:


data_file="customers.xlsx"
customer_data=pd.read_excel(data_file)

df=pd.DataFrame()
mortgage=[]

customer_data.head()
sex=pd.get_dummies(customer_data['Sex'], drop_first=True)

customer_data.drop('Post code', axis=1, inplace=True)
customer_data.drop(['Sex', 'Current Account', 'Location','Savings acc', 'ISA'], axis=1, inplace=True)
customer_data=pd.concat([customer_data, sex], axis=1)
customer_data.head()


# In[3]:


for i in range(len(customer_data)):
    if((customer_data.loc[i,'Mortgage balance'])>0):
        mortgage.append(1)
    else:
        mortgage.append(0)

customer_data['has_Mortgage']=pd.Series(mortgage)
customer_data.head()
customer_data.tail(15)
customer_data.drop('Mortgage balance', axis=1, inplace=True)


        


# In[4]:


customer_data.head()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(customer_data.drop('has_Mortgage',axis=1), 
           customer_data['has_Mortgage'], test_size=0.20, 
            random_state=51)


# In[6]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[16]:


train_pred=logmodel.predict(X_train)
Predictions = logmodel.predict(X_test)


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(y_test,Predictions))
print(classification_report(y_train, train_pred))


# In[18]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, Predictions))
print(confusion_matrix(y_train, train_pred))


# In[19]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Predictions))
print(accuracy_score(y_train, train_pred))


# In[15]:


sns.lineplot(data=y_test)
sns.lineplot(data=Predictions)


# In[ ]:


logmodel.predict(X_train)

