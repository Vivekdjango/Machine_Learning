import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
df = pd.read_csv('advertising.csv')
df.head()


# In[14]:


sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')


# In[15]:


df.info()


# In[16]:


sns.distplot(df.Age,kde=False)


# In[17]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=df)


# In[19]:


sns.jointplot(x='Age',y='Daily Internet Usage',data=df,kind='kde',color='red')


# In[20]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=df,color='g')


# In[21]:


sns.pairplot(df,hue='Clicked on Ad',palette='RdBu_r')


# In[22]:


from sklearn.cross_validation import train_test_split


# In[23]:


X = df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = df['Clicked on Ad']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


lr = LogisticRegression()


# In[27]:


lr.fit(X_train,y_train)


# In[28]:


pred = lr.predict(X_test)


# In[29]:


#Check Accuracy
from sklearn.metrics import classification_report
print classification_report(y_test,pred)

plt.show()

# In[30]:


#So the model has 92% accuracy which is very good

