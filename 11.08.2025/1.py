import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data={'abs_days':[2,5,1,8,4,6,3,7,0,9],'marks':[88,75,92,60,80,70,85,65,95,55]}
df=pd.DataFrame(data);
x=df[['abs_days']]
y=df['marks']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
def predict_marks(days):
  ans=lr.predict(days)
  return ans

user_input=int(input("enter no of days absent"))
predicted=predict_marks([[user_input]])
print(f"predicted marks for absent days:{user_input}::{predicted}")
print("scatter plot")
plt.scatter(x,y,color='red',label='scatterpoints')
plt.plot(x,lr.predict(x),color='blue',label='regressionline')
plt.xlabel('abs_days')
plt.ylabel('marks')
plt.title('marks vs absent days')
plt.legend()
plt.show()
