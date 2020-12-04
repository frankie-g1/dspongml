from sklearn.linear_model import ElasticNet
import pandas as pd
from joblib import dump

data = pd.read_csv('pong_data.csv')
train_x = data[['ball_x', 'ball_y']]
train_x_mask = train_x[train_x.iloc[:, 0] > 400]
print(train_x.shape)
train_y = data['paddle_y']
print(train_y.shape)

model = ElasticNet()
model.fit(train_x, train_y)
dump(model, 'modelE.joblib')
