import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression


data = pd.read_csv('pong_data.csv')
train_x = data[['ball_x', 'ball_y']]
print(train_x.shape)
train_y = data['paddle_y']
print(train_y.shape)

model = LinearRegression(fit_intercept=True)
model.fit(train_x, train_y)
dump(model, 'model.joblib')










