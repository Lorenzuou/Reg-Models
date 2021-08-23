from pickle import TRUE
import optuna
from optuna.study import Study
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
from tqdm import tqdm
from sklearn.linear_model import RidgeCV


class RegModels:
    def __init__(self, modo, X_trainS, X_testS, y_trainS, y_testS, path="fitted_models") -> None:
        self.modo = modo
        self.path = path
        self.X_train = X_trainS
        self.y_train = y_trainS
        self.X_test = X_testS
        self.y_test = y_testS
        self.stack_model = RidgeCV()
        global X_train
        X_train = self.X_train
        global y_train
        y_train = self.y_train
        global X_test
        X_test = self.X_test
        global y_test
        y_test = self.y_test
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
                                'KNN_Regressor': obj_KNeighborsRegressor,
                                'Ridge_CV': obj_ridge_CV,
                                'SGD_Reg':obj_SGDRegressor}

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

    def fit_models(self, modelsList=False, path="fitted_models", verbose=True):
        self.X_train = self.X_train
        self.y_train = self.y_train
        self.path = path
        self.stacking_regressor = False

        self.modelsList = modelsList

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

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
                          'Light_Boost':  5,
                          'Ridge_CV': 15}

        k = 0
        if(opt):
            self.models_fit = []  # cria um dict para armazenar os modelos treinados pelo optuna
            self.study_list = {}
            qtd = 0
            for m in modelsList:
                qtd = qtd + modelsList[m]
            iteration = 0
            for m in modelsList:
                model = self.standard_models[m]
                print('Optimizing model', m)
                best_model, study = set_model(model, modelsList[m])
                iteration += modelsList[m]
                self.models_fit.append((m, best_model))
                save_model(self.models_fit[k][1], (m + ".sav"), path=self.path)
                self.study_list[m] = study
                print("\n\nModel concluded:  {} saved as '{}' \n ".format(
                    self.models_fit[k][1], m + ".sav"))
                k += 1
                # tqdm(iteration, total=qtd,

                #      desc="Text You Want")
        else:
            import os
            savedModels = os.listdir(self.path)
            self.models_fit = []  

            for m in modelsList:
                for e in savedModels:
                    if((m + '.sav') == e):
                        self.models_fit.append(
                            (e, load_model(e, self.path).fit(X_train, y_train)))

    def visualize_slice(self, model_name, param_list=False):
        if self.study_list:
            fig = optuna.visualization.plot_slice(self.study_list[model_name])
            fig.show()

        else:
            print("No studies have been done by the class")

    # faz o ensemble de todos os modelos de regressão para resultar em uma só previsao

    def stack_models(self, models_to_stack=False,  verbose=True):
        from sklearn.ensemble import StackingRegressor

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        estimators = []
        self.models_fit = []

        if models_to_stack:
            if(len(self.models_fit) > 0):  # Se ja houveram modelos fitados, o programa os usa
                for e in models_to_stack:
                    model = self.models_fit[e]
                    estimators.append(e, model)
            else:
                import os
                savedModels = os.listdir(self.path)

                for m in models_to_stack:
                    for e in savedModels:
                        print(load_model(e, self.path))

                        if(m + ".sav" == e):
                            estimators.append((e, load_model(e, self.path)))

        elif(len(self.models_fit) > 0):
            estimators = self.models_fit
        else:
            import os

            savedModels = os.listdir(self.path)

            if(len(savedModels) < 1):
                print("There are no saved models on: {}".format(self.path))
                return False

            for e in savedModels:
                estimators.append((e, load_model(e, self.path)))

        print("Estimators are: \n ", estimators)

        self.stacking_regressor = StackingRegressor(
            estimators=estimators, final_estimator= self.stack_model)
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
        if(self.modo == "optimize"):
            for m in self.models_fit:
                test_pred = m[1].predict(X_test)
                scoreList.append(m[1].score(X_test, y_test))
                test_metrics.append(calculate_metrics(y_test, test_pred, m[0]))
        else:
            import os

            savedModels = os.listdir(self.path)

            if(len(savedModels) < 1):
                print("There are no saved models on: {}".format(self.path))
                return False

            for m in savedModels:
                model = load_model(m, self.path)
                test_pred = model.predict(X_test)
                scoreList.append(model.score(X_test, y_test))
                test_metrics.append(calculate_metrics(y_test, test_pred, m))

        data = pd.concat(test_metrics)
        data['R2'] = scoreList
        return data

    def stack_model_tune(self, n_trials, model=False):
        if not model:
            final = set_model(obj_ridge_CV, n_trials)
        else:
            final = set_model(self.standard_models[model], n_trials)
        self.stack_model = final
        return final


def set_model(model, n_trials):
    study = optuna.create_study(direction="maximize")

    study.optimize(model, n_trials=n_trials, callbacks=[callback])

    print(study.best_trial)
    best_model = study.user_attrs["best_model"]

    return best_model, study


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


def obj_ridge_CV(trial):
    # h_alpha = trial.suggest_float("n_alphas", 0.1, 1)
    h_random_state = trial.suggest_int("random_state", 1, 50)
    cv = trial.suggest_categorical("cv", [None,  2, 3, 4, 5])

    if cv == None:
        alpha_per_target = trial.suggest_categorical(
            "alpha_per_target", [True, False])
        model = RidgeCV(
            cv=cv, alpha_per_target=alpha_per_target)
    else:
        alpha_per_target = trial.suggest_categorical(
            "alpha_per_target", [True, False])
        model = RidgeCV(
            cv=cv)

    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_lasso(trial):
    h_alpha = trial.suggest_float("alpha", 0.1, 1.0)

    model = linear_model.Lasso(alpha=h_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)
# min_samples_split=h_min_samples_split,
# max_depth=h_max_depth, n_estimators=h_n_estimators, min_samples_leaf=h_min_samples_leaf


def obj_random_forest(trial):
    f_n_estimators = trial.suggest_int("n_estimators", 100, 500)  # descrição
    f_max_depth = trial.suggest_int("max_depth",10,50)
    f_min_samples_leaf = trial.suggest_categorical("leaf", [1, 2, 4])
    f_min_samples_split = trial.suggest_categorical(
        "samples_split", [2, 5, 10])
    int_max_features = np.linspace(len(X_train.columns),10,5,dtype=int) 
    
    max_features_list = ["auto","sqrt","log2",None] 


    for e in int_max_features: 
        max_features_list.append(int(e))


    f_max_features = trial.suggest_categorical("max_features",max_features_list )
    model = ensemble.RandomForestRegressor(max_depth =f_max_depth, n_estimators = f_n_estimators, min_samples_split = f_min_samples_split, min_samples_leaf = f_min_samples_leaf, max_features = f_max_features, n_jobs = -1)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_elastic_net(trial):
    e_alpha = trial.suggest_float("alpha",0.1,1)
    # e_random_state = trial.suggest_int("random_state", 1, 50)
    # e_max_iter = trial.suggest_int("max_iter",100,10000)
    e_l1_ratio = trial.suggest_float("l1_ratio",0.1,1)
    # e_cv = trial.suggest_int("cv", 3, 5)
    model = linear_model.ElasticNet(
       alpha=e_alpha,l1_ratio = e_l1_ratio )
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_ada_boost(trial):
    h_n_estimators = trial.suggest_int("n_estimators", 10, 1000)  # descrição
    h_random_state = trial.suggest_int("random_state", 1, 50)
    h_learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    h_loss = trial.suggest_categorical(
        "loss", ['linear', 'square', 'exponential'])
    model = ensemble.AdaBoostRegressor(random_state=h_random_state, learning_rate=h_learning_rate, loss=h_loss,
                                       n_estimators=h_n_estimators)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_ridge(trial):
    h_alpha = trial.suggest_float("alpha",0.1, 500)
    model = linear_model.Ridge(alpha=h_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_XGBRegressor(trial):
    x_max_deph = trial.suggest_int("max_depth",3,10)
    x_learning_rate = trial.suggest_float("learning_rate",0.01,0.1)
    x_n_estimators = trial.suggest_int("n_estimators",100,1000)
    x_colsample_bytree = trial.suggest_float("bytree",0.1,0.9)
    x_min_child_weight =  trial.suggest_int("min_child_weight",4,9)
    model = xgboost.XGBRegressor(max_deph = x_max_deph,learning_rate = x_learning_rate, n_estimators = x_n_estimators, colsample_bytree = x_colsample_bytree, min_child_weight  =x_min_child_weight   )
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_extra_trees(trial):
    e_n_estimators = trial.suggest_int("n_estimators", 10, 100)  # descrição
    e_random_state = trial.suggest_int("random_state", 1, 50)
    e_min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
    e_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    e_oob_score = trial.suggest_categorical("oob_score",[True,False])
    e_bootstrap = e_oob_score if False else True # necessary for the model to work 
    
    model = ensemble.ExtraTreesRegressor(oob_score = e_oob_score, min_samples_leaf=e_min_samples_leaf,
                                         min_samples_split=e_min_samples_split, n_estimators=e_n_estimators, random_state=e_random_state, bootstrap = e_bootstrap)
    trial.set_user_attr(key="best_model", value=model )

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
    sdg_alpha = trial.suggest_float('alpha',0.0001,1)
    model = linear_model.SGDRegressor(
        verbose=0, penalty=sdg_penalty, loss=sdg_loss, alpha=sdg_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)


def obj_catBoostRegressor(trial):
    cat_n_estimators = trial.suggest_int("n_estimators", 800, 2000)
    cat_depth = trial.suggest_int("depth",8,12)
    model = CatBoostRegressor(
         n_estimators=cat_n_estimators,
        depth=10,
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
