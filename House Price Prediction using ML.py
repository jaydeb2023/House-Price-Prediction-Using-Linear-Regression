#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


data = pd.read_csv('BostonHousing.csv')


# In[5]:


data.head()


# In[7]:


data.info()


# In[8]:


print(data.isnull().sum())


# In[9]:


X = data.drop("medv", axis=1)  # Features (independent variables)
y = data["medv"]               # Target (dependent variable: Median value of homes)


# In[10]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[12]:


# Predict on the test set
y_pred = model.predict(X_test)


# In[13]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[14]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# In[15]:


# Visualize the predicted vs actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()


# In[ ]:




