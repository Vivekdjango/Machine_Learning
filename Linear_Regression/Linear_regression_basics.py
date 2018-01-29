
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("USA_Housing.csv")
print data.head()
sns.pairplot(data)
sns.distplot(data['Price'])
sns.heatmap(data.corr())   #What is correlation?
print data.columns
X = data[[u'Avg. Area Income', u'Avg. Area House Age',
       u'Avg. Area Number of Rooms', u'Avg. Area Number of Bedrooms',
       u'Area Population']]
y = data['Price']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
accuracy = lm.score(X_test,y_test)  #to calculate the accuracy of data
print accuracy
print lm.intercept_  #Its co-efficent value
print lm.coef_  #It returns Co-eff of each column
cdf = pd.DataFrame(lm.coef_,X_train.columns,columns=['Coeff'])
print cdf
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions))   # To check the price diff between test and coorect values
from sklearn import metrics
print metrics.mean_absolute_error(y_test,predictions)
print metrics.mean_squared_error(y_test,predictions)
print np.sqrt(metrics.mean_squared_error(y_test,predictions))
