# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Developing a neural network regression model entails a structured process, encompassing phases such as data acquisition, preprocessing, feature selection, model architecture determination, training, hyperparameter optimization, performance evaluation, and deployment, followed by ongoing monitoring for refinement.

## Neural Network Model

![image](https://github.com/user-attachments/assets/b868e4cd-4c1a-432b-a8d2-af7f44e57f15)


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
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

ws = gc.open('data1').sheet1

rows = ws.get_all_values()
```
```python
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'sno':'float'})
df = df.astype({'marks':'float'})
df.head()

x = df[["sno"]].values
y = df[["marks"]].values
```
```python
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train1 = scaler.transform(x_train)
```
```python
marks_data = Sequential([Dense(6,activation='relu'),Dense(7,activation='relu'),Dense(1)])
marks_data.compile(optimizer = 'rmsprop' , loss = 'mse')

marks_data.fit(x_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)
loss_df.plot()

x_test1 = scaler.transform(x_test)
marks_data.evaluate(x_test1,y_test)

X_n1 = [[30]]
X_n1_1 = scaler.transform(X_n1)
marks_data.predict(X_n1_1)
```
## Dataset Information

![Screenshot 2024-09-22 175552](https://github.com/user-attachments/assets/86599e84-b686-4683-bee0-bfd9c8e4b6f4)

## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-09-22 175648](https://github.com/user-attachments/assets/c949fb6a-576b-47b3-996c-86dcbd944ac2)


### Test Data Root Mean Squared Error

![Screenshot 2024-09-22 175853](https://github.com/user-attachments/assets/6951b9d9-a2ab-445a-b7a8-a6a7fe404bee)


### New Sample Data Prediction

![Screenshot 2024-09-22 175927](https://github.com/user-attachments/assets/a63764fe-2015-4a1c-8cbd-af301e5947b6)


## RESULT

Thus the program executed successfully
