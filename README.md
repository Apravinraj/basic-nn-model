# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Pravin Raj A
### Register Number: 212222240079
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

AI_squad=Sequential([
    Dense(units=9,activation='relu',input_shape=[8]),
    Dense(units=9,activation='relu'),
    Dense(units=9,activation='relu')
])

AI_squad.summary()

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('excel').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'mark':'int'})
df.head()

import pandas as pd
from sklearn.model_selection import train_test_split

x=df[['roll_no']].values
y=df[['mark']].values

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train=Scaler.transform(x_train)
x_test=Scaler.transform(x_test)

AI_squad=Sequential([
    Dense(units=9,activation='relu',input_shape=[1]),
    Dense(units=9,activation='relu'),
    Dense(units=9,activation='relu')
])

AI_squad.compile(optimizer = 'rmsprop', loss= 'mse')
AI_squad.fit(x_train,y_train,epochs=100)

loss_df = pd.DataFrame(AI_squad.history.history)
loss_df.plot()

x_test1 = Scaler.transform(x_test)
AI_squad.evaluate(x_test1,y_test)
n_n1_1=Scaler.transform([[1]])
AI_squad.predict(n_n1_1)
```

## OUTPUT

## Dataset Information
![Screenshot 2024-09-01 144812](https://github.com/user-attachments/assets/2f142502-bba2-4831-9a3b-1bbadf95cf2d)

![Screenshot 2024-09-01 160900](https://github.com/user-attachments/assets/6cb84bbc-6ca7-4604-9d6e-1488dcc36d67)


### Training Loss Vs Iteration Plot
![Screenshot 2024-09-01 160941](https://github.com/user-attachments/assets/4effaad7-2ed5-4efe-ae4e-aa5a2c6f4975)


### Test Data Root Mean Squared Error
![Screenshot 2024-09-01 161014](https://github.com/user-attachments/assets/a7a23616-5b64-4da1-b054-884cb303d683)


### New Sample Data Prediction

![Screenshot 2024-09-01 161027](https://github.com/user-attachments/assets/8ae674b4-d5df-4222-897b-f1a91b9b3608)

## RESULT

Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
