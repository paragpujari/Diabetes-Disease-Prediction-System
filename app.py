# import all the necessary libraries

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from   sklearn.model_selection   import train_test_split
from   sklearn                   import svm
from   sklearn.metrics           import accuracy_score
from   sklearn.preprocessing     import StandardScaler
from   PIL                       import Image


# load the diabetes dataset
diabetes_df  = pd.read_csv('Projects\datasets\diabetes.csv')

# load the first five rows of the dataset
# print(diabetes_df.head())


# group the data in terms of Outcome to get the mean distibution of the data
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()
# print(diabetes_mean_df)


# Split the data into independent and dependent variables
X = diabetes_df.drop(columns='Outcome', axis = 1)
Y = diabetes_df['Outcome']

# print(X)
# print(Y)

# scale the input variables using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# print(X)
# print(Y)

# Split the dataset into independent and dependent variables
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


# create an SVM Model with a linear kernel
model = svm.SVC(kernel='linear')


# train the model using the model of SVM
model.fit(X_train, Y_train)


# make the predictions on the training and testing sets
train_y_pred = model.predict(X_train)

test_y_pred = model.predict(X_test)


# Calculate the accuracy on the training and testing sets
acc_train = accuracy_score(train_y_pred, Y_train)
acc_test = accuracy_score(test_y_pred, Y_test)

print('Accuracy of the training data is:', (acc_train)*100.0)
print('Accuracy of the testing data is:', (acc_test)*100.0)



# Create the streamlit app

def app():
    # open the Image
    img = Image.open(r'Projects\Image\img.jpeg')
    # resize the image
    img = img.resize((200,200))
    # store the image in the streamlit app
    st.image(img,caption='Diabetes Image', width=200)

    # set the tile of the Image
    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')

    # enter all the variables with their given input range
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on given input data
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf,age]
    # convert the input data into numpy array
    input_data_nparray = np.asarray(input_data)
    # reshape the given input data
    reshaped_input_data = input_data_nparray.reshape(1,-1)
    # predict the model
    prediction = model.predict(reshaped_input_data)

    #display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if(prediction == 1):
        st.warning('This person has diabetes.')
    else:
        st.warning('This person does not have diabetes.')

      # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    # display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {acc_train:.2f}')
    st.write(f'Test set accuracy: {acc_test:.2f}')



if __name__ == '__main__':
    app()