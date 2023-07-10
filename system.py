import numpy as np
import pickle

# # Load the saved model

loaded_model = pickle.load(open('E:/Projects/Jupyter/ML/ML Projects/Diabetes/Web app/trained_model.sav', 'rb'))

input_data = (10,139,80,0,0,27.1,1.441,57)

# changing data a numpy array
input_datarry = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_dataa = input_datarry.reshape(1, -1)

prediction = loaded_model.predict(input_dataa)
print(prediction)

if (prediction[0] == 0):
    print ('The person is not diabetic')
else: 
    print('This person is diabetic')
