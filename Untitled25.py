#!/usr/bin/env python
# coding: utf-8

# # Prediction and Analysis

# In[2]:


import pandas as pd
from sklearn import linear_model


# In[3]:


df=pd.read_csv(r"G:\COVID-19 Project\Coronaprojectdata.csv")


# In[4]:


df.head(2)


# # Data Analysis

# In[5]:


from matplotlib import pyplot as plt 


# In[6]:


plt.xlabel('Days')
plt.ylabel('')
plt.title('Total Test Cases')
plt.scatter(df.Days,df.T_T,color='black',linewidth=5, linestyle='dotted')


# In[7]:


plt.bar(df.Days,df.T_C,color='black',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('Total Positive Cases')


# In[8]:


plt.plot(df.Days,df.T_D,color='Red',linewidth=5, linestyle='dotted')
plt.xlabel('Days')
plt.ylabel('')
plt.title('Total Death')


# In[9]:


plt.plot(df.Days,df.T_R,color='Green',linewidth=5)
plt.xlabel('Days')
plt.ylabel('')
plt.title('Total Recovery cases')


# In[10]:


plt.bar(df.Days,df.N_T,color='blue',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('New Test Cases')


# In[11]:


plt.bar(df.Days,df.N_C,color='gray',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('New Cases')


# In[12]:


plt.bar(df.Days,df.N_D,color='Red',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('New Death cases ')


# In[13]:


plt.bar(df.Days,df.N_R,color='Green',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('Recovery cases')


# In[14]:


plt.xlabel('Days')
plt.ylabel('')
plt.title('Death Cases and Recovery Cases Analysis')
plt.plot(df.Days, df.N_D, label="Death Cases")
plt.plot(df.Days, df.N_R, label="Recovery cases")

plt.legend(loc='best')


# In[15]:


plt.xlabel("Cases")
plt.ylabel("Days")
plt.title("New Recovary and Death Cases Analysis")



plt.hist([df.N_D,df.N_R], rwidth=0.95, color=['red','green'],label=['Death Cases','Recovery Cases'])
plt.legend()


# In[16]:



plt.xlabel("Cases")
plt.ylabel("Days")
plt.title("Total Recovary and Death Cases Analysis")

plt.hist([df.T_D,df.T_R], rwidth=0.95, color=['red','green'],label=['Total Death Cases','Total Recovery Cases'])
plt.legend()


# # Linear Regression Model

# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


reg=LinearRegression()


# In[19]:


reg.fit(df[['Days','T_T']],df[['T_C','T_D','T_R','N_C','N_D','N_R']])


# In[20]:


reg.predict([[90,400000]])


# In[21]:


df.head(2)


# In[22]:


y = df[['T_C','T_D','T_R','N_T','N_C','N_D','N_R']]
x = df.drop(['T_C','T_D','T_R','N_T','N_C','N_D','N_R','Date'],axis=1)
x=x.dropna()


# In[23]:


x.head(2)


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


logmodel=LinearRegression()


# In[28]:


logmodel.fit(x_train, y_train)


# In[29]:


logmodel.predict([[90,400000]])


# In[30]:


logmodel.score(x_test,y_test)


# In[31]:


logmodel.coef_


# In[32]:


logmodel.intercept_


# In[33]:


logmodel.score(x_train,y_train)


#  # k-nearest neighbors algorithm

# In[34]:


from sklearn.neighbors import KNeighborsRegressor


# In[35]:


log = KNeighborsRegressor(n_neighbors=2)


# In[36]:


log.fit(x_train, y_train)


# In[37]:


log.predict([[100,200000]])


# In[38]:


log.score(x_test,y_test)


# # Cities

# In[39]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[40]:


df=pd.read_csv(r"G:\COVID-19 Project\coronacity.csv")


# In[41]:


df.head(2)


# In[42]:


xpos = np.arange(len(df.Division))
xpos


# In[43]:


plt.bar(df.Division,df.d_17_5_2020,color='black',)
plt.xlabel('Divisons')
plt.ylabel('Cases')
plt.title('Divison Cases')
plt.tick_params(axis='x', rotation=80)


# In[44]:


from sklearn import preprocessing


# In[45]:


label_encoder = preprocessing.LabelEncoder()


# In[46]:


df['Cities']= label_encoder.fit_transform(df['Cities']) 


# In[47]:


df['Cities'].unique()


# # K means Algorithm

# In[48]:


plt.scatter(df.d_17_5_2020,df['Cities'])
plt.xlabel('Cities')
plt.ylabel('d_29_4_20')
plt.title('Cities Condition')


# In[49]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['d_17_5_2020']])
y_predicted


# In[50]:


df['cluster']=y_predicted
df.head(2)


# In[51]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]


plt.scatter(df1.Cities,df1['d_17_5_2020'],color='green')
plt.scatter(df2.Cities,df2['d_17_5_2020'],color='red')
plt.scatter(df3.Cities,df3['d_17_5_2020'],color='blue')
plt.scatter(df4.Cities,df4['d_17_5_2020'],color='yellow')



plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Cities')
plt.legend()


# In[52]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]


plt.scatter(df1.d_17_5_2020,df1['Cities'],color='green')
plt.scatter(df2.d_17_5_2020,df2['Cities'],color='red')
plt.scatter(df3.d_17_5_2020,df3['Cities'],color='blue')
plt.scatter(df4.d_17_5_2020,df4['Cities'],color='yellow')



plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Cities')
plt.legend()


# In[53]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[54]:


df=pd.read_csv(r"G:\COVID-19 Project\dhakacitydata.csv")


# In[55]:


df.head(2)


#  # Data Analysis

# In[56]:


xpos = np.arange(len(df.Cities))
xpos


# In[57]:


plt.bar(df.Cities,df.d_15_5_2020,color='black',)
plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Cities Cases')


# In[58]:


from sklearn import preprocessing


# In[59]:


label_encoder = preprocessing.LabelEncoder() 


# In[60]:


df['Cities']= label_encoder.fit_transform(df['Cities'])


# In[61]:


df['Cities'].unique()


# In[62]:


df.head(2)


#  # K means  Algorithm

# In[63]:


plt.scatter(df.Cities,df['d_17_5_2020'])
plt.xlabel('Cities')
plt.ylabel('d_17_5_2020')


# In[64]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['d_17_5_2020']])
y_predicted


# In[65]:


df['cluster']=y_predicted
df.head(2)


# In[66]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]


plt.scatter(df1.Cities,df1['d_17_5_2020'],color='green')
plt.scatter(df2.Cities,df2['d_17_5_2020'],color='red')
plt.scatter(df3.Cities,df3['d_17_5_2020'],color='yellow')


plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Dhaka City Area')
plt.legend()


#  # Agglomerative Clustering

# In[67]:


plt.scatter(df.d_21_4_20,df['d_17_5_2020'])
plt.xlabel('Cities')
plt.ylabel('d_29_4_20')
plt.title("Situation of Cities")


# In[68]:


km = AgglomerativeClustering(n_clusters=3)
y_predicted = km.fit_predict(df[['d_17_5_2020']])
y_predicted


# In[69]:


df['cluster']=y_predicted
df.head(2)


# In[71]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3= df[df.cluster==2]
plt.scatter(df1.d_21_4_20,df1['d_17_5_2020'],color='red')
plt.scatter(df2.d_21_4_20,df2['d_17_5_2020'],color='yellow')
plt.scatter(df3.d_21_4_20,df3['d_17_5_2020'],color='green')
plt.xlabel('Cities')
plt.ylabel('Cases')
plt.title('Situation Of Dhaka City Area')
plt.legend()


#  # Date

# In[82]:


df=pd.read_csv(r"G:\COVID-19 Project\datasetdate.csv")


# In[83]:



df.head(2)


# In[84]:


plt.bar(df.date,df.f_d,color='Red',)
plt.xlabel('Date')
plt.ylabel('')
plt.title('Female Death Cases')
plt.tick_params(axis='x', rotation=90)


# In[85]:


plt.bar(df.date,df.m_d,color='Red',)
plt.xlabel('Date')
plt.ylabel('')
plt.title('Male Death Cases')
plt.tick_params(axis='x', rotation=90)


# In[86]:


plt.bar(df.date,df.dha_d,color='Red',)
plt.xlabel('Date')
plt.ylabel('')
plt.title('Dhaka Death Cases')
plt.tick_params(axis='x', rotation=90)


# In[87]:


plt.bar(df.date,df.syl_d,color='Red',)
plt.xlabel('Days')
plt.ylabel('')
plt.title('Sylet Death Cases')
plt.tick_params(axis='x', rotation=80)


# In[88]:


plt.xlabel('Days')
plt.ylabel('')
plt.title('Percentage Of Male and Female cases Analysis')


plt.plot(df.date, df.c_m_p, label="Male Cases")
plt.plot(df.date, df.c_f_p, label="Female Cases")

plt.legend(loc='best')
plt.tick_params(axis='x', rotation=80)


# In[89]:


y = df[['c_a_p_30']]
x = df.drop(['c_f_p','c_m_p','a_31_40','dha_h_q','syl_h_q','bar_h_q','ran_h_q','chi_h_q','raj_h_q','khu_h_q','dha_c','syl_c','bar_c','ran_c','chi_c','raj_c','khu_c','dha_d','syl_d','bar_d','ran_d','chi_d','raj_d','khu_d','f_d','m_d','a_0_10','a_11_20','a_21_30','a_41_50','a_51_60','a_60','t_q','c_a_p_10','c_a_p_20','c_a_p_40','c_a_p_30','c_a_p_50','c_a_p_60','c_a_p_60+'],axis=1)
x=x.dropna()


# In[90]:


x


# In[91]:


from sklearn.preprocessing import LabelEncoder


# In[92]:


le=LabelEncoder()


# In[93]:


le.fit(x['date'])


# In[94]:


x['date']=le.transform(x['date'])


# In[95]:


x


# # Linear Regression Model 

# In[96]:


from sklearn.model_selection import train_test_split


# In[97]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[98]:


from sklearn.linear_model import LinearRegression


# In[99]:


reg=LinearRegression()


# In[100]:



reg.fit(x_train, y_train)


# In[101]:


reg.predict([[100]])


#  # k-nearest neighbors algorithm

# In[102]:


from sklearn.neighbors import KNeighborsRegressor


# In[103]:


neigh = KNeighborsRegressor(n_neighbors=2)


# In[104]:


neigh.fit(x_train, y_train)


# In[105]:


neigh.predict([[100]])


# In[72]:


a=99


# In[73]:


b=98


# In[81]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = [ 'Linear Regression', 'KNN']
students = [98,99]
ax.bar(langs,students)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Of Algorithm')
plt.show()


# In[ ]:





#  # Data Analysis

# # Data Vizulaization

# # Data cleaning

# # Machine learning 

# # Supervised and Unsupervised Learning by using multiple libraries in Python.

# # Machine Learning for Data Science

# # Tool : Python

# # THE END
