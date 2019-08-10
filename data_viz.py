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


# In[10]:


data_file="customers.xlsx"
customer_data=pd.read_excel(data_file)

df=pd.DataFrame()
mortgage=[]

customer_data.head()
sex=pd.get_dummies(customer_data['Sex'], drop_first=True)
current_account=pd.get_dummies(customer_data['Current Account'], drop_first=True)
customer_data.drop('Post code', axis=1, inplace=True)
customer_data.drop(['Sex', 'Current Account', 'Location','Savings acc', 'ISA'], axis=1, inplace=True)
customer_data=pd.concat([customer_data, sex, current_account], axis=1)
customer_data.head()


# In[13]:


for i in range(len(customer_data)):
    if((customer_data.loc[i,'Mortgage balance'])>0):
        mortgage.append(1)
    else:
        mortgage.append(0)

customer_data['has_Mortgage']=pd.Series(mortgage)
customer_data.head()
customer_data.tail(15)
customer_data.drop('Mortgage balance', axis=1, inplace=True)


        


# In[14]:


customer_data.head()


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(customer_data.drop('has_Mortgage',axis=1), 
           customer_data['has_Mortgage'], test_size=0.20, 
            random_state=50)


# In[35]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[37]:


Predictions = logmodel.predict(X_test)


# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_test,Predictions))


# In[39]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, Predictions))


# In[ ]:




