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


# In[54]:


data_file="customers.xlsx"
customer_data=pd.read_excel(data_file)
sex=pd.get_dummies(customer_data['Sex'], drop_first=True)
savings_acc=pd.get_dummies(customer_data['Savings acc'], drop_first=True)

customer_data.drop(['Post code','Sex','Location','Current Account', 'Savings acc', 'ISA'], axis=1, inplace=True)
customer_data=pd.concat([customer_data, savings_acc], axis=1)
creditcard=[]


# In[55]:


for i in range(len(customer_data)):
    if((customer_data.loc[i,'Credit card balance'])>0):
        creditcard.append(1)
    else:
        creditcard.append(0)
customer_data['has_Creditcard']=pd.Series(creditcard)
customer_data.head()


# In[56]:


customer_data.drop('Credit card balance',axis=1, inplace=True)


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(customer_data.drop('has_Creditcard',axis=1), 
           customer_data['has_Creditcard'], test_size=0.20, 
            random_state=100)


# In[65]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[66]:


train_pred=logmodel.predict(X_train)
Predictions = logmodel.predict(X_test)


# In[67]:


from sklearn.metrics import classification_report
print(classification_report(y_test,Predictions))
print(classification_report(y_train, train_pred))


# In[68]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, Predictions))
print(confusion_matrix(y_train, train_pred))


# In[69]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Predictions))
print(accuracy_score(y_train, train_pred))


# In[ ]:




