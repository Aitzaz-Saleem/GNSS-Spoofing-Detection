# # -*- coding: utf-8 -*-
# """
# Created on Sun Apr 16 07:11:48 2023

# @author: aitza
# """

# import joblib
# import tensorflow

# x = int(input("Enter Satellite PRN : "))
# y = int(input("Select model for prediction \nPress 1: Naive Bayes \nPress 2: Support Vector Machine \nPress 3: Artifical Neural Network\n"))
# if x == 3 and y ==1: 
#     loaded_model = joblib.load('Satellite_3_1.sav')
# elif x == 6 and y ==1:
#     loaded_model = joblib.load('Satellite_6_1.sav')
# elif x == 7 and y ==1:
#     loaded_model = joblib.load('Satellite_7_1.sav')
# elif x == 10 and y ==1:
#     loaded_model = joblib.load('Satellite_10_1.sav')
# elif x == 13 and y ==1:
#     loaded_model = joblib.load('Satellite_13_1.sav')
# elif x == 16 and y ==1:
#     loaded_model = joblib.load('Satellite_16_1.sav')
# elif x == 19 and y ==1:
#     loaded_model = joblib.load('Satellite_19_1.sav')    
# elif x == 23 and y ==1:
#       loaded_model = joblib.load('Satellite_23_1.sav')
# elif x == 3 and y ==2: 
#     loaded_model = joblib.load('Satellite_3_2.sav')
# elif x == 6 and y ==2:
#     loaded_model = joblib.load('Satellite_6_2.sav')
# elif x == 7 and y ==2:
#     loaded_model = joblib.load('Satellite_7_2.sav')
# elif x == 10 and y ==2:
#     loaded_model = joblib.load('Satellite_10_2.sav')
# elif x == 13 and y ==2:
#     loaded_model = joblib.load('Satellite_13_2.sav')
# elif x == 16 and y ==2:
#     loaded_model = joblib.load('Satellite_16_2.sav')
# elif x == 19 and y ==2:
#     loaded_model = joblib.load('Satellite_19_2.sav')    
# elif x == 23 and y ==2:
#       loaded_model = joblib.load('Satellite_23_2.sav')
# elif x == 3 and y ==3: 
#     loaded_model = tensorflow.keras.models.load_model('Satellite_3_3.h5')
# elif x == 6 and y ==3:
#     loaded_model = tensorflow.keras.models.load_model('Satellite_6_3.h5')
# elif x == 7 and y ==3:
#     loaded_model = tensorflow.keras.models.load_model('Satellite_7_3.h5')
# elif x == 10 and y ==3:
#     loaded_model = tensorflow.keras.models.load_model('Satellite_10_3.h5')
# elif x == 13 and y ==3:
#     loaded_model = tensorflow.keras.models.load_model('Satellite_13_3.h5')
# elif x == 16 and y ==3:
#     loaded_model = tensorflow.keras.models.load_model('Satellite_16_3.h5')
# elif x == 19 and y ==3:
#     loaded_model = tensorflow.keras.models.load_model('Satellite_19_3.h5')    
# else:
#       loaded_model = tensorflow.keras.models.load_model('Satellite_23_3.h5')

# y_pred = loaded_model.predict([[12222,15555,10000,-577,-76,12345]])
# print("The prediction is  : ", y_pred)

x = int(input("Enter Movement :\
              \nPress 1: 5% Maintained Control Group\
              \nPress 2: 5% Maintained Intervention Group\
              \nPress 3: 10% Maintained Control Group\
              \nPress 4: 10% Maintained Intervention Group\
              \nPress 5: MVC Control Group\
              \nPress 1: MVC Intervention Group\
              \nPress 2: 5% Ramp Control Group\
              \nPress 3: 5% Ramp Intervention Group\
              \nPress 4: 10% Ramp Control Group\
              \nPress 5: 10% Ramp Intervention Group\
              \n"))
y = int(input("Select Option \
              \nPress 1: Machine Learning \
              \nPress 2: Deep Learning\
              \n"))
# If user select Machine Leraning
z = int(input("Select model for prediction \
              \nPress 1: Light Gradient Boosting Machine\
              \nPress 2: Extreme Gradient Boosting\
              \nPress 3: Adaptive Boosting\
              \nPress 4: Decision Tree Classifier\
              \nPress 5: Random Forest Classifier\
              \nPress 6: Gradient Boosting\n"))
# If user select Deep Leraning
q = int(input("Select model for prediction \
              \nPress 1: Long Short-Term Memory \
              \nPress 2: Transformers \
              \n"))
