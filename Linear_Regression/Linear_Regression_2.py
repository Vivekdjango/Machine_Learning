import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


customers = pd.read_csv("Ecommerce Customers")

print customers.head()
print customers.info()

print customers.describe()


sns.jointplot(customers["Time on Website"],customers["Yearly Amount Spent"])
sns.jointplot(customers["Time on App"],customers["Yearly Amount Spent"])

sns.pairplot(customers)

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

from sklearn.cross_validation import train_test_split

print customers.columns

X  = customers[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]
y = customers["Yearly Amount Spent"]

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)


print lm.coef_

#print y_test.head()

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
from sklearn import metrics

print "MAE: %f"%(metrics.mean_absolute_error(y_test,predictions))
print "MSE: %f"%(metrics.mean_squared_error(y_test,predictions))
print "RMSE: %f"%(np.sqrt(metrics.mean_squared_error(y_test,predictions)))

plt.hist(y_test-predictions)
sns.distplot(y_test-predictions)
plt.show()
cdf = pd.DataFrame(lm.coef_,X_train.columns,columns=['Coeff'])
print cdf

accuracy = lm.score(X_test,y_test)
print accuracy
