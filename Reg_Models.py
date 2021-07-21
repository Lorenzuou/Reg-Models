from pickle import TRUE
import optuna
import xgboost
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn import linear_model
from sklearn import ensemble
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
import numpy as np
from source.source import calculate_metrics


class RegModels:
    def __init__(self, modo, X_trainS, X_testS, y_trainS, y_testS) -> None:
        self.modo = modo
        self.X_train = X_trainS
        self.y_train = y_trainS
        self.X_test = X_testS
        self.y_test = y_testS
        self.models_fit = False

        self.standard_models = {'Lasso': obj_lasso,
                                'Random_Forest':  obj_random_forest,
                                'Elastic_Net':  obj_elastic_net,
                                'Ada_Boost':  obj_ada_boost,
                                'Ridge':  obj_ridge,
                                'XGBR': obj_XGBRegressor,
                                'Extra_Trees': obj_extra_trees,
                                'Cat_Boost': obj_catBoostRegressor,
                                'Light_Boost': obj_LightBoost,
                                'KNN_Regressor': obj_KNeighborsRegressor}

    # Retorna a lista de modelos dispomíveis
    # verbose == True, a funcao printa o nome dos modelos
    def get_Standard_Models(self, verbose=True):
        if verbose:
            print("Modelos disponíveis: ")
            for e in self.standard_models:
                print(e)
            print("................")
        return self.standard_models

    def get_Fitted_Models(self, verbose=True):
        if(self.models_fit):
            return self.models_fit
        else:
            return False

    def fit_models(self, modelsList=False, path="fitted_models"):
        self.X_train = self.X_train
        self.y_train = self.y_train
        self.path = path
        self.stacking_regressor = False
        global X_train
        X_train = self.X_train
        global y_train
        y_train = self.y_train
        global X_test
        X_test = self.X_test
        global y_test
        y_test = self.y_test

        self.modelsList = modelsList

        if(self.modo == "optimize"):
            self.optimize_models(opt=True)
        else:
            self.optimize_models(opt=False)

    def optimize_models(self, opt):
      

        if self.modelsList:
            modelsList = self.modelsList  # lista de modelos definidas pelo usuario
        else:  # Modelos padrões
            modelsList = {'Lasso': 15,
                          'Random_Forest':   10,
                          'Elastic_Net':   15,
                          'Ada_Boost':   15,
                          'Ridge':   15,
                          'XGBR':  1,
                          'Extra_Trees':  15,
                          'Cat_Boost':  15,
                          'Light_Boost':  5}

        self.models_fit = []  # cria um dict para armazenar os modelos treinados pelo optuna
        k = 0
        if(opt):
            for m in modelsList:
                model = self.standard_models[m]
                self.models_fit.append((m, set_model(model, modelsList[m])))
                save_model(self.models_fit[k][1], (m + ".sav"), path=self.path)
                k += 1
            for m in self.models_fit:
                print("modelo {} treinado".format(m))
        else:
            import os
            savedModels = os.listdir(self.path)

            for m in modelsList:
                for e in savedModels:
                    if((m + '.sav') == e):
                        self.models_fit.append(
                            (e, load_model(e, self.path).fit(X_train, y_train)))
                        print(e)

    # faz o ensemble de todos os modelos de regressão para resultar em uma só previsao
    def stack_models(self):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import RidgeCV

        estimators = []

        if(len(self.models_fit) > 0):
            estimators = self.models_fit
        else:
            import os

            savedModels = os.listdir(self.path)

            if(len(savedModels) < 1):
                print("There are no saved models on: {}".format(self.path))
                return False

            for e in savedModels:
                estimators.append((e, load_model(e, self.path)))

        self.stacking_regressor = StackingRegressor(
            estimators=estimators, final_estimator=RidgeCV())
        self.stacking_regressor.fit(X_train, y_train)

        return self.stacking_regressor

    def get_Stacked_Model(self):
        if(self.stacking_regressor):
            return self.stacking_regressor
        else:
            print("Stacked model was not assembled")
            return False

    # retorna uma tabela com a performace de todos os modelos

    def models_performace(self):

        test_metrics = []
        scoreList = []
        for m in self.models_fit:
            test_pred = m[1].predict(X_test)
            scoreList.append(m[1].score(X_test, y_test))
            test_metrics.append(calculate_metrics(y_test, test_pred, m[0]))

        data = pd.concat(test_metrics)
        data['R2'] = scoreList
        return data


def set_model(model, n_trials):
    study = optuna.create_study(direction="maximize")
    study.optimize(model, n_trials=n_trials, callbacks=[callback])
    print(study.best_trial)
    best_model = study.user_attrs["best_model"]

    return best_model


def save_model(model, name, path):
    import pickle
    filename = path + '/' + name
    pickle.dump(model, open(filename, 'wb'))


def load_model(name, path):
    import pickle
    filename = path + '/' + name
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def score_method(model, trial):
    for step in range(100):
        model.fit(X_train, y_train)

        # Report intermediate objective value.
        intermediate_value = model.score(X_test, y_test)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        return intermediate_value


# funcoes com os modelos de regressao a serem otimizados.
# Mais modelos podem ser adicionados, só é necessário que o retorno e a estrutura geral seja a mesma
# Esses modelos ja estao com algumas sugestoes de otimização, contudo mais opções podem ser adicionadas


def obj_lasso(trial):
    h_alpha = trial.suggest_float("alpha", 0.1, 1.0, log=True)
    model = linear_model.Lasso(alpha=h_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_random_forest(trial):
    h_n_estimators = trial.suggest_int("n_estimators", 100, 500)  # descrição
    h_max_depth = trial.suggest_categorical(
        "max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None])  # descrição
    h_min_samples_leaf = trial.suggest_categorical("leaf", [1, 2, 4])
    h_min_samples_split = trial.suggest_categorical(
        "samples_split", [2, 5, 10])
    model = ensemble.RandomForestRegressor(min_samples_split=h_min_samples_split,
                                           max_depth=h_max_depth, n_estimators=h_n_estimators, min_samples_leaf=h_min_samples_leaf)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_elastic_net(trial):
    h_n_alphas = trial.suggest_int("n_alphas", 10, 1000, log=True)
    h_random_state = trial.suggest_int("random_state", 1, 50, log=True)
    h_cv = trial.suggest_int("cv", 3, 5, log=True)
    model = linear_model.ElasticNetCV(
        n_alphas=h_n_alphas,  cv=h_cv, random_state=h_random_state)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_ada_boost(trial):
    h_n_estimators = trial.suggest_int("n_estimators", 10, 1000)  # descrição
    h_random_state = trial.suggest_int("random_state", 1, 50, log=True)
    h_learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    h_loss = trial.suggest_categorical(
        "loss", ['linear', 'square', 'exponential'])
    model = ensemble.AdaBoostRegressor(random_state=h_random_state, learning_rate=h_learning_rate, loss=h_loss,
                                       n_estimators=h_n_estimators)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_ridge(trial):
    h_alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    model = linear_model.Ridge(alpha=h_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_XGBRegressor(trial):
    model = xgboost.XGBRegressor(objective='reg:squarederror')
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_extra_trees(trial):
    h_n_estimators = trial.suggest_int("n_estimators", 10, 100)  # descrição
    h_random_state = trial.suggest_int("random_state", 1, 50)
    h_min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
    h_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    model = ensemble.ExtraTreesRegressor(min_samples_leaf=h_min_samples_leaf,
                                         min_samples_split=h_min_samples_split, n_estimators=h_n_estimators, random_state=h_random_state)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_KNeighborsRegressor(trial):
    h_n_neighbors = trial.suggest_int("n_neighbors", 2, 7)
    model = neighbors.KNeighborsRegressor(n_neighbors=h_n_neighbors)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_SGDRegressor(trial):
    sdg_loss = trial.suggest_categorical(
        "sdg_loss", ['squared_loss', 'huber', 'epsilon_insensitive'])
    sdg_penalty = trial.suggest_categorical(
        "penalty", ['l1', 'l2', 'elasticnet'])
    model = linear_model.SGDRegressor(
        verbose=1, penalty=sdg_penalty, loss=sdg_loss)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_catBoostRegressor(trial):
    cat_n_estimators = trial.suggest_int("n_estimators", 800, 1000)
    cat_loss = trial.suggest_categorical(
        "MAE", ['squared_loss', 'MAPE', 'Poisson'])
    model = CatBoostRegressor(
        n_estimators=cat_n_estimators,
        loss_function='MAE',
        eval_metric='RMSE',
        verbose=False
    )
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_LightBoost(trial):
    light_min_samples_split = trial.suggest_int("min_samples_split", 200, 1001)
    light_max_depth = trial.suggest_int("max_depth", 7, 14)
    model = lgbm.LGBMRegressor(max_depth=light_max_depth,
                               min_samples_split=light_min_samples_split, subsample=0.8, random_state=10)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(
            key="best_model", value=trial.user_attrs["best_model"])
