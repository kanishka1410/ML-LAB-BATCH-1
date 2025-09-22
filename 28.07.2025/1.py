import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data={'week':[1,2,3,4,5],'studyhours':[12,18,22,28,35]}
df=pd.DataFrame(data)
print(df)
x=df[['week']]
y=df['studyhours']
model=LinearRegression()
model.fit(x,y);
future_weeks=[[7],[9]]
y_pred=model.predict(future_weeks)
print(y_pred)
