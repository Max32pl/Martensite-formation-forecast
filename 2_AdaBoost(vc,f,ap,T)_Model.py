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

def dataEinlesen():

    data_df = pd.read_csv(r"C:\PathVersuche_Mod_Mar.csv",sep=";")
    #print(data_df)

    data_X = data_df[["f [mm]","T [C]","ap [mm]","vc [m/min]"]]
    data_y = data_df[["Ferritsc [%]"]]
    print("\nEingang:",data_X)
    print("\nAusgang:",data_y)

    return data_X,data_y

def adaBoostRegressor(data_X, data_y):

    #Speicherort von Modell und Daten
    path = r"C:\Path"
    #Liste zum Abspeichern des trainierten Modells der einzelnen Durchgänge der Kreuzvalidierung
    filename = ["2_Model_AdaBoost_vc_ap_f_T_Nr1","2_Model_AdaBoost_vc_ap_f_T_Nr2","2_Model_AdaBoost_vc_ap_f_T_Nr3"
        ,"2_Model_AdaBoost_vc_ap_f_T_Nr4","2_Model_AdaBoost_vc_ap_f_T_Nr5"]
    #Liste zum Abspeichern der der Testdaten
    testdata_x_list = ["2_testdata_X_AdaBoost_vc_ap_f_T_Nr1","2_testdata_X_AdaBoost_vc_ap_f_T_Nr2","2_testdata_X_AdaBoost_vc_ap_f_T_Nr3"
        ,"2_testdata_X_AdaBoost_vc_ap_f_T_Nr4","2_testdata_X_AdaBoost_vc_ap_f_T_Nr5"]
    testdata_y_list = ["2_testdata_y_AdaBoost_vc_ap_f_T_Nr1", "2_testdata_y_AdaBoost_vc_ap_f_T_Nr2","2_testdata_y_AdaBoost_vc_ap_f_T_Nr3"
        ,"2_testdata_y_AdaBoost_vc_ap_f_T_Nr4","2_testdata_y_AdaBoost_vc_ap_f_T_Nr5"]
    # Liste zum Abspeichern der der Trainingsdaten
    traindata_x_list = ["2_traindata_X_AdaBoost_vc_ap_f_T_Nr1","2_traindata_X_AdaBoost_vc_ap_f_T_Nr2","2_traindata_X_AdaBoost_vc_ap_f_T_Nr3"
        ,"2_traindata_X_AdaBoost_vc_ap_f_T_Nr4","2_traindata_X_AdaBoost_vc_ap_f_T_Nr5"]
    traindata_y_list = ["2_traindata_y_AdaBoost_vc_ap_f_T_Nr1","2_traindata_y_AdaBoost_vc_ap_f_T_Nr2","2_traindata_y_AdaBoost_vc_ap_f_T_Nr3"
        ,"2_traindata_y_AdaBoost_vc_ap_f_T_Nr4","2_traindata_y_AdaBoost_vc_ap_f_T_Nr5"]
    
    #Variable Durchgänge 
    i = 0
    
    Kreuzvalidierung
    rn = range(1, 72)

    kf5 = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf5.split(rn):

        print("\nDurchlauf ", i+1, ":")

        # print("Train:",train_index,"\nTest:",test_index)
        X_train = data_X.iloc[train_index]
        X_train = X_train.to_numpy()
        X_test = data_X.iloc[test_index]
        X_test = X_test.to_numpy()
        y_train = data_y.iloc[train_index]
        y_train = y_train.to_numpy()
        y_train = y_train.flatten()
        y_test = data_y.iloc[test_index]
        y_test = y_test.to_numpy()
        y_test = y_test.flatten()

        # print("Test:",y_test)
        # print("Train X:",len(X_train),"Train y:",len(y_train))

        #Trainieren des Modells
        regrM = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), random_state=None, n_estimators=20)
        regrM.fit(X_train, y_train)

        y_pred_test = regrM.predict(X_test)
        y_pred_train = regrM.predict(X_train)

        #Güte für Testdaten
        abserror = mean_absolute_error(y_test, y_pred_test)
        print("Abs Error Test: ", abserror)
        rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        print("RMSE Test:", rmse)
        r2_Score = r2_score(y_test, y_pred_test)
        print("r2_score Test:", r2_Score)

        #Güte für Trainingsdaten
        abserror = mean_absolute_error(y_train, y_pred_train)
        print("Abs Error Train: ", abserror)
        rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        print("RMSE Train:", rmse)
        r2_Score = r2_score(y_train, y_pred_train)
        print("r2_score Train:", r2_Score)

        #Modell speichern in .sav Datei
        file = os.path.join(path, filename[i])
        pickle.dump(regrM, open(file, 'wb'))
        
        #Datensatz speichern in .sav Datei
        file = os.path.join(path, testdata_y_list[i])
        pickle.dump(y_test, open(file, 'wb'))
        file = os.path.join(path, testdata_x_list[i])
        pickle.dump(X_test, open(file, 'wb'))
        file = os.path.join(path, traindata_y_list[i])
        pickle.dump(y_train, open(file, 'wb'))
        file = os.path.join(path, traindata_x_list[i])
        pickle.dump(X_train, open(file, 'wb'))

        i += 1

if __name__ == '__main__':

    data_X,data_y = dataEinlesen()

    adaBoostRegressor(data_X,data_y)
