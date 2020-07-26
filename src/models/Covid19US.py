# IMport the necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters

keras.backend.set_floatx('float64')

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

#Import dataset
data = pd.read_csv('data/us-data.csv', thousands=',')

# Drop the unnecessary columns
data_deaths = data[['Date', 'Positive', 'Hospitalized – Cumulative','In ICU – Cumulative','On Ventilator – Cumulative', 'Deaths']]

# Covert the data to right format
data_deaths['Date'] = pd.to_datetime(data_deaths['Date'].astype(str), format='%Y%m%d')
data_deaths = data_deaths.sort_values(by=['Date'], ascending=True)

# Convert the date column to datetime
data_deaths['Date'] = pd.to_datetime(data_deaths['Date'])

# Set the index of the DataFrame to the date column
data_deaths.set_index('Date', inplace = True)
data_deaths.sort_index(ascending=True)

# Drop NAs
data_deaths.fillna(0, inplace=True)

# Set the Train and Test set
train_size = int(len(data_deaths) * 0.9)
test_size = len(data_deaths) - train_size
train, test = data_deaths[0:train_size], data_deaths[train_size:len(data_deaths)]

# Scalar the data
from sklearn.preprocessing import RobustScaler

f_columns = ['Positive', 'Hospitalized – Cumulative','In ICU – Cumulative','On Ventilator – Cumulative']

f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['Deaths']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['Deaths'] = cnt_transformer.transform(train[['Deaths']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['Deaths'] = cnt_transformer.transform(test[['Deaths']])

# Prep the data to put into the model
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

X_train, y_train = create_dataset(train, train.Deaths, time_steps)
X_test, y_test = create_dataset(test, test.Deaths, time_steps)

# Set the model
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128, 
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=24, 
    validation_split=0.1,
    shuffle=False,
    verbose=0
)

# Predict 
y_pred = model.predict(X_test)

y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

# Show the prediction
# calculate RMSE
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv.reshape(1, -1)))

print('Mean Prediction Deaths: %.0f' % rmse)
print('Accuracy: %.2f' % (100 - rmse / np.mean(y_test_inv)))

