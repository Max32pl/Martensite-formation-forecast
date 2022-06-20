import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import os

def f(X):

    #ga variiert ganze Zahlen, 2. Stelle nach Komma wird benötigt
    X[0] = X[0]/100
    X[2] = X[2]/100

    #Eingangsvariable für Prediction von Rz
    input_pred_rz = X[0]
    input_pred_rz = np.reshape(input_pred_rz,(-1,1))

    #Eingangsvariable für Prediction von Martensit
    #3. und 4. Stelle zum Einstellen von ap und vc nutzen
    input_pred_m = [[X[0], X[1], X[2], X[3]]]

    #Laden des gespeicherten Modells AdaBoost/Martensit
    filename = ["2_Model_AdaBoost_vc_ap_f_T_Nr1", "2_Model_AdaBoost_vc_ap_f_T_Nr2", "2_Model_AdaBoost_vc_ap_f_T_Nr3"
        , "2_Model_AdaBoost_vc_ap_f_T_Nr4", "2_Model_AdaBoost_vc_ap_f_T_Nr5"]
    file = os.path.join(path, filename[3])
    loaded_model = pickle.load(open(file, 'rb'))

    #Prediction Martensit
    m_pred = loaded_model.predict(input_pred_m)

    #Laden des gespeicherten Modells LinReg/Rauheit
    loaded_model = pickle.load(open('model_lin_Rz', 'rb'))

    #Prediction Rauheit
    r_pred = loaded_model.predict(input_pred_rz)

    """Gewichtung theoretische Rauheit & Martensitbildung"""
    y = (w1*abs(m_target-m_pred))+(w2*abs(r_target-r_pred))

    return y

if __name__ == '__main__':

    """Optimierung"""

    #Speicherort von Modell und Daten
    global path
    path = r"C:\Path"

    """Gewichtung, Zielgrößen"""
    global w1
    w1 = 0.5
    global w2
    w2 = 0.5
    global m_target
    m_target = 4
    global r_target
    r_target = 20

    #genetic algorithm
    X = []
    varbound = np.array([[10, 70], [-150, -50],[20, 50],[20, 70]])
    vartype = np.array([['int'], ['int'], ['int'], ['int']])

    algorithm_param = {'max_num_iteration': None,
                       'population_size': 100,
                       'mutation_probability': 0.5,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 200}

    model = ga(function=f,
               dimension=4,
               variable_type_mixed=vartype,
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)


    model.run()

    #Formatieren des Lösungsoutputs in numpy
    solution_list = model.output_dict['variable']
    #ga variiert ganze Zahlen, 2. Stelle nach Komma wird benötigt
    solution_list[0] = solution_list[0] / 100
    solution_list[2] = solution_list[2] / 100
    print('\nLösung: ', solution_list)

    #Ausgabe der pred. Rauheit & Differenz zum Soll
    loaded_model = pickle.load(open('model_lin_Rz', 'rb'))
    r_pred = loaded_model.predict(solution_list[0].reshape(1,-1))
    print("\nRauheit pred: ",r_pred)
    print('Rauheit soll: ', r_target)

    r_diff = abs(r_target-r_pred)
    print("Rauheit Diff: ",r_diff)

    #Ausgabe des pred Martensit & Differenz zum Soll & Varianz der 5 Modellierungen
    # Laden des gespeicherten Modells AdaBoost/Martensit
    filename = ["2_Model_AdaBoost_vc_ap_f_T_Nr1", "2_Model_AdaBoost_vc_ap_f_T_Nr2", "2_Model_AdaBoost_vc_ap_f_T_Nr3"
        , "2_Model_AdaBoost_vc_ap_f_T_Nr4", "2_Model_AdaBoost_vc_ap_f_T_Nr5"]
    martensit_list = []

    for i in filename:
        file = os.path.join(path, i)
        loaded_model = pickle.load(open(file, 'rb'))

        martensit_list.append(loaded_model.predict(solution_list.reshape(1,-1)))

    print('\nMartensit pred: ', martensit_list[3])
    print('Martensit soll: ',m_target)

    m_diff = abs(m_target-martensit_list[3])
    print("Martensit Diff: ",m_diff)

    print('Martensit pred der 5 Modellierungen: ', martensit_list)

    varianz = np.var(martensit_list)
    print("Varianz: ", varianz)
