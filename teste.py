from Reg_Models import RegModels

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors

from sklearn import model_selection
import xgboost

data = pd.read_csv("./dados/FRA3-FRA6_cleaned_feature_engineered_reduzido.csv")
# data = data.sample(frac=1).reset_index(drop=True)
# data = data[:-100000] # redução da quantidade de linhas se preciso para execução mais rápida


def create_scaled_dataset(X, y):
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    std_scaler = StandardScaler()
    X_scaled_train = pd.DataFrame(std_scaler.fit_transform(
        X_train), columns=X_train.columns, index=X_train.index)
    X_scaled_test = pd.DataFrame(std_scaler.transform(
        X_test), columns=X_train.columns, index=X_test.index)
    return X_scaled_train, X_scaled_test, y_train, y_test


# Vert_irreg_left_rail


# data = data[real_columns]
# print(data.columns)
target = "Vert_irreg_right_rail"

y = data[target]
X = data.drop(columns=target)


X_train, X_test, y_train, y_test = create_scaled_dataset(X, y)


# modelsList = {'Lasso': obj_lasso,
#               'Random_Forest':  obj_random_forest,
#               'Elastic_Net':  obj_elastic_net,
#               'Ada_Boost':  obj_ada_boost,
#               'Ridge':  obj_ridge,
#               'XGBR': obj_XGBRegressor,
#               'Extra_Trees': obj_extra_trees,
#               'Cat_Boost': obj_catBoostRegressor,
#               'Light_Boost': obj_LightBoost,
#                'KNN_Regressor' : obj_KNeighborsRegressor }

a = RegModels(modo="load", X_trainS=X_train,
                X_testS=X_test, y_trainS=y_train, y_testS=y_test)

minhaLista = {'Lasso': 15,
              'Ridge': 15, 
              'Elastic_Net': 10 }


print(a.getStandardModels())

a.fit_models(minhaLista)

model = a.stack_models()

print(model)

# print(a.models_performace())

# a.getStandardModels()

# #Colocar o nome do modelo e a quantidade de trials respectiva 



# # Treina a lista de modelos especifica
# # por padrao, coloca os modelos no diretorio "modelosTreinados", mas pode ter o caminho especificado em path
# a.fit_models(minhaLista, "modelosTreinados2")


# # model = a.stack_models()
# table = a.models_performace()


