import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import random
random.seed(0)
attend=np.random.randint(1,100,100)
activity=np.random.randint(1,10,100)
internal=np.random.randint(1,50,100)
totmarks=(attend*0.2+activity*0.3+internal*0.5)*2
df=pd.DataFrame({'attendance':attend,'activity':activity,'internal_marks':internal,'end_sem':totmarks})
print(df)
x=df[['attendance','activity','internal_marks']]
y=df['end_sem']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)

def predict_marks(test_data):
  ans=lr.predict(test_data)
  return ans
att=int(input("enter attendance"))
actmark=int(input("enter activity marks"))
intmark=int(input("enter internal marks"))
predicted=predict_marks([[att,actmark,intmark]])
print(predicted)
y_pred=lr.predict(x_test)
plt.scatter(y_test,y_pred,color='blue',label='scatterpoints')
plt.xlabel('actual vs predicted')
plt.ylabel('end_sem')
plt.title('marks vs absent days')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red',label='line')  # perfect fit line
plt.legend()
plt.show()
