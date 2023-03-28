#!/usr/bin/env python
# coding: utf-8
# # Oasis Infobyte (Data Science Internship)
# 
# ## Task - 3 : Car Price Prediction With Machine Learning
# 
# ## Author : Patil Saloni Ravindra
# 
# ## March - P2 Batch Oasis Infobyte SIP

# ## Importing Libraries
# In[44]:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# In[9]:
data=pd.read_csv("C:/Users/admin/Desktop/CarPrice_Assignment.csv")
data
# In[10]:
data.tail()
# In[11]:
data.head()
# In[14]:
data.info()
# In[15]:
data.describe()
# In[16]:
data.isnull().sum()
# In[17]:
data.duplicated().sum()
# In[18]:
data.shape
# In[19]:
print(data.price.describe(percentiles=[0.225,0.50,0.75,0.85,0.98,1]))
# # Exploratory Data Analysis
# In[20]:
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("Door Number Histogram")
sns.countplot(data.doornumber,palette=("plasma"))
plt.subplot(1,2,2)
plt.title('Door Number vs Price')
sns.boxplot(x=data.doornumber,y=data.price,palette=("plasma"))
plt.show()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title('Aspiration Histogram')
sns.countplot(data.aspiration, palette=("plasma"))
plt.subplot(1,2,2)
plt.title("Aspiration vs Price")
sns.boxplot(x=data.aspiration,y=data.price,palette=("plasma"))
plt.show()
# In[21]:
# fuel type
colors=sns.color_palette('pastel')
labels=data['fueltype'].dropna().unique()
plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
plt.title('fueltype_Percentage')
plt.pie(data['fueltype'].value_counts(),labels=labels,colors=colors,autopct='%.2f%%')
plt.subplot(1,2,2)
plt.title('fueltype Bar chart')
sns.countplot(x="fueltype",data=data,palette=colors)
data.fueltype.value_counts(dropna=False)
# In[25]:
df=pd.DataFrame(data.groupby(["fueltype"])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title("Fuel Type vs Average Price")
plt.show()
df=pd.DataFrame(data.groupby(["carbody"])["price"].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('car Type vs Average price')
plt.show()
# In[29]:
predict="price"
data=data[["symboling","wheelbase","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]]
# In[30]:
x=np.array(data.drop([predict],1))
y=np.array(data[predict])
# In[31]:
print(x)
print(y)
# In[32]:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
# # Random Forest Model
# In[33]:
from sklearn.ensemble import RandomForestRegressor
# In[35]:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
print('training data shape is :{}.'.format(x_train.shape))
print('training label shape is:{}.'.format(y_train.shape))
print('testing data shape is:{}.'.format(x_test.shape))
print('testing labelshape is:{}.'.format(y_test.shape))
# In[36]:
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
# In[37]:
regressor.fit(x,y)
# In[38]:
regressor.score(x_train,y_train)
# In[39]:
regressor.score(x_test,y_test)
# In[40]:
from sklearn.metrics import accuracy_score
predictions=regressor.predict(x_test)
# In[41]:
percentage=regressor.score(x_test,y_test)
percentage
# In[43]:
# Check the accuracy on the training set
print(regressor.score(x_train,y_train))
print(f"Test set:{len(x_test)}")
print(f"Accuracy={percentage*100}%")
# # Thank You!
