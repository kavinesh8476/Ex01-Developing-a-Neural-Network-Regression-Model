# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1) Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

2) Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

3) First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/user-attachments/assets/3d7adc83-202f-4c73-b802-278b49253ace)


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
### Name:Kavinesh M
### Register Number:212222230064
## Importing Required packages
```python
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default

```
## Authenticate the Google sheet
```py
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet = gc.open('experiment1').sheet1
data = worksheet.get_all_values()
```
## Construct Data frame using Rows and columns
```py
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()
x=df[['input']].values
y=df[['output']].values
x
```
## Split the testing and training data
```py
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)
```
## Build the Deep learning Model
```py
ai=models.Sequential([
    layers.Dense(units=9,input_shape=[1]),
    layers.Dense(units=9),
    layers.Dense(units=1)
])
ai.summary()
ai.compile(optimizer = 'rmsprop', loss = 'mse')
ai.fit(X_train1,y_train,epochs=1000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
X_test1=Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
X_1=[[5]]
X_1_1 = Scaler.transform(X_1)
ai.predict(X_1_1)
```

## Dataset Information

![image](https://github.com/user-attachments/assets/5dd1c27b-f597-41a3-86a3-2f0797355a13)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/96b5dcf7-38c5-440a-b0ea-7e41d2372a99)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/6b3367c8-5a21-4468-b330-ad02fd534209)
![image](https://github.com/user-attachments/assets/593bbca6-3daa-4046-b1d2-1fb52cef37ea)
![image](https://github.com/user-attachments/assets/54484e53-23bc-424b-8daa-08e4a174fdd0)
![image](https://github.com/user-attachments/assets/f5612098-4399-42fc-9c0a-a1518a92866a)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/7e83d2e8-6481-4584-86d4-f88bb3e79925)


## RESULT

Thus a Neural network for Regression model is Implemented Successfully.
