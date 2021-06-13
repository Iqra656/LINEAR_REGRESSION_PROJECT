#!/usr/bin/env python
# coding: utf-8

# # IQRA MALIK

# In[80]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[81]:


df=pd.read_csv("C:/Users/Imran/Desktop/My Projects/My_project.csv",header=0)
df.head()


# In[82]:


df.shape


# In[83]:


df.describe()


# In[84]:


sns.jointplot(x="n_hot_rooms",y='price',data=df)


# In[85]:


sns.jointplot(x="rainfall",y="price",data=df)


# In[86]:


df.head()


# In[87]:


sns.countplot(x="airport",data=df)


# In[88]:


sns.countplot(x="waterbody",data=df)


# In[89]:


sns.countplot(x="bus_ter",data=df)


# In[90]:


df.info()


# In[91]:


#Outlier Treatment
np.percentile(df.n_hot_rooms,[99])


# In[92]:


uv=np.percentile(df.n_hot_rooms,[99])[0]
uv


# In[93]:


df[(df.n_hot_rooms>uv)]


# In[94]:


#identify Outlier
df.n_hot_rooms[(df.n_hot_rooms>3*uv)]=3*uv


# In[95]:


df.describe()


# In[96]:


lv=np.percentile(df.rainfall,[1])[0]
lv


# In[97]:


df[(df.rainfall<lv)]


# In[98]:


df.rainfall[(df.rainfall<lv*0.3)]=lv*3


# In[99]:


sns.jointplot(x="crime_rate",y="price",data=df)


# In[100]:


df.describe()


# In[101]:


#Missing values with n_hos_bed variabe
df.info()


# In[102]:


df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())


# In[103]:


df.info()


# In[104]:


#Seasonality in DATA
sns.jointplot(x="crime_rate",y="price",data=df)


# In[105]:


sns.jointplot(x='n_hos_beds',y='price',data=df)


# In[106]:


df.crime_rate=np.log(1+df.crime_rate)


# In[ ]:





# In[107]:


sns.jointplot(x="crime_rate",y="price",data=df)


# In[108]:


df['avg_dist']=(df.dist1+df.dist2+df.dist3+df.dist4)/4


# In[109]:


df.head()


# In[110]:


del df['dist1']


# In[111]:


del df['dist2']


# In[112]:


del df['dist3']


# In[113]:


del df['dist4']


# In[114]:


df.head()


# In[115]:


del df['bus_ter']


# In[120]:


df=pd.get_dummies(df,drop_first=True)


# In[121]:


df.head()


# In[ ]:





# In[122]:


del df['waterbody_None']


# In[123]:


df.head()


# In[124]:


df.corr()


# In[125]:


import statsmodels.api as sn


# In[126]:


X =sn.add_constant(df['room_num'])


# In[127]:


lm=sn.OLS(df['price'],X).fit()


# In[128]:


lm.summary()


# In[46]:


from sklearn.linear_model import LinearRegression


# In[47]:


y=df['price']


# In[48]:


x=df[['room_num']]


# In[49]:


lm2=LinearRegression()


# In[50]:


lm2.fit(x,y)


# In[51]:


print(lm2.intercept_,lm2.coef_)


# In[52]:


lm2.predict(x)


# In[53]:


sns.jointplot(y=df['price'],x=df['room_num'],data=df,kind='reg')


# In[54]:


#Multiple linear regression model
x_multi= df.drop('price',axis=1)


# In[55]:


x_multi.head()


# In[56]:


y_multi=df['price']


# In[57]:


x_multi_cons=sn.add_constant(x_multi)


# In[58]:


x_multi_cons.head()


# In[59]:


lm_multi=sn.OLS(y_multi,x_multi_cons).fit()


# In[60]:


lm_multi.summary()


# In[61]:


lm3=LinearRegression()


# In[62]:


lm3.fit(x_multi,y_multi)


# In[63]:


print(lm3.intercept_,lm3.coef_)


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train,X_test,Y_train,Y_test= train_test_split(x_multi,y_multi,test_size=0.2,random_state=0)


# In[66]:


print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[67]:


lm_a=LinearRegression()


# In[68]:


lm_a.fit(X_train,Y_train)


# In[69]:


y_test_a=lm_a.predict(X_test)


# In[71]:


y_train_a=lm_a.predict(X_train)


# In[72]:


from sklearn.metrics import r2_score


# In[74]:


r2_score(Y_test,y_test_a)


# In[75]:


r2_score(Y_train,y_train_a)


# In[ ]:




