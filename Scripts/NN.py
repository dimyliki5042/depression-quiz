import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier


def Conf_Bin(x):
    if x == 'Male' or x == 'Yes':
        return 1
    else:
        return 0
    
def Conf_Diet(x):
    if x == 'Healthy':
        return 3
    elif x == 'Moderate':
        return 2
    elif x == 'Unhealthy':
        return 1
    else:
        return 0
    
def Conf_Sleep(x):
    x = x.replace('\'', '')
    if x == '5-6 hours':
        return 2
    elif x == 'Less than 5 hours':
        return 1
    elif x == '7-8 hours':
        return 3
    elif x == 'More than 8 hours':
        return 4
    else:
        return 0

def Config_Data(data):
    data.rename(columns={'Have you ever had suicidal thoughts ?': 'Suicidal', 'Family History of Mental Illness': 'Mental'}, inplace=True)
    data['Gender'] = data['Gender'].apply(Conf_Bin)
    data['Suicidal'] = data['Suicidal'].apply(Conf_Bin)
    data['Mental'] = data['Mental'].apply(Conf_Bin)
    data['Sleep Duration'] = data['Sleep Duration'].apply(Conf_Sleep)
    data['Dietary Habits'] = data['Dietary Habits'].apply(Conf_Diet)
    data['Financial Stress'] = data['Financial Stress'].apply(lambda x: 0.0 if x == '?' else x).astype(float)
    result = data['Depression']
    del data['City'], data['Profession'], data['Degree'], data['id'], data['Depression']
    return data, result

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def Show_Confusion_Matrix(cm, data_type):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.title(f'Матрица ошибок {data_type} выборки', pad=15)
    plt.show()

def Train():
    raw_data = pd.read_csv('Sources\\dataset.csv')
    data, result = Config_Data(raw_data)
    x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=0)
    param = {
         'objective': 'binary:logistic',
         'n_estimators': 100
    }
    model = XGBClassifier(
        **param,
        eval_metric=['logloss', 'error'],
        learning_rate = 0.05
    )
    train_data, test_data = (x_train, y_train), (x_test, y_test)
    model.fit(x_train, y_train, eval_set=[train_data, test_data], verbose=False)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    evals_result = model.evals_result()
    return model, evals_result, acc, cm

def Get_Answer(answers):
    model, evals_result, acc, cm = Train()
    result = model.predict(Create_DF(answers))
    if result == 0:
        txt = 'Вы здоровы'
    else:
        txt = 'Наблюдаются признаки депрессии'
    return txt, evals_result, acc, cm

def Create_DF(answers):
    data = pd.DataFrame([answers], columns=['Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                                            'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Suicidal', 'Work/Study Hours', 'Financial Stress', 'Mental'])
    return data