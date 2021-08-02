# Guia Classe de otimização com Optuna

Este guia foi escrito no intuito auxiliar o usuário na utilização do código que gera uma classe em python para facilitar o uso e otimização de modelos de Machine Learning de Regressão. 
Entre as utilidades estã a otimização de modelos e o armazenamento deles. O ensemble de modelos de regressão e a comparação de resultados entre eles. 
Além disso, adicionar e editar as funções de otimização de cada modelo é simples. 

Por padrão, há alguns parâmetros de otimização no código das funções dos modelos. No entanto, espera-se que o usuário os edite da forma adequada para seus processos específicos. 

### Introdução

O código desenvolvido possui uma classe chamada  **IrregModels.**  
Espera-se que o usuário já tenha preparado os dados de treino e teste para utilizar a classe.

Sua instanciação é feita da seguinte forma 

 **

```python
obj = RegModels(X_train = X_train, 
	X_test = X_test, y_train = y_train, y_test = y_test)
```

Sendo os parâmetros os dados de treino e teste a serem utilizados no modelo

### Uso

Baixe esse repositório e importe: "from diretorio.Reg_Models import RegModels

### Atributos

modo → "optimize" para utilizacao das funcoes 

X_train

X_test

y_train

y_test

models_fit → Lista de modelos a serem treinados 

standard_models → Lista de modelos padrão com as respectivas funções de otimização no optuna. 

## Funções:

get_Standard_Models(self, verbose = True)

Retorna a lista com todos os modelos disponiveis para consulta do usuário no formato: {'nome do modelo',modelo} 

```python
a = RegModels(modo="optimize", X_trainS=X_train,
                X_testS=X_test, y_trainS=y_train, y_testS=y_test, path="fitted_models")

a.getStandardModels(verbose=True)
```

```python
output: 
Modelos disponíveis: 
Lasso
Random_Forest
Elastic_Net
Ada_Boost
Ridge
XGBR
Extra_Trees
Cat_Boost
Light_Boost
KNN_Regressor
................
```

**get_Fitted_Models**(self,verbose = True)

* retorna uma lista com os modelos já treinados da instância

**fit_models**(self, modelsList=False, path="modelosTreinados", verbose = True)

* funcao para treinar os modelos

**modelList**

* padrão = False, o programa uma lista arbitrária com todos os modelos do código 

* É um dicionário no formato {'Nome do modelo 1' : 15, 'Nome do modelo 2' : 15} 

* A key é o nome do modelo (tem que ser igual ao que estiver na lista de **getStandardModels** )  e o valor é a quantidade de tentativas (trials) que o modelo vai ser executado no optuna com parâmetros diferentes. (número maior de tentativas = maior tempo de execução) 

**path** 

* caminho de diretório aonde os modelos serão salvos em arquivos .sav 

* pré-condição: caminho deve existir 

**verbose**
* padrão = True, exibe os logs do optuna

**stack_models**(self)

* Quando chamada, a funcao irá utilizar a lista de funções especificadas pelo usuário anteriormente em **fit_models() para aplicar o ensemble (utilizando o StackingRegressor do sklearn)** no objetivo de atingir uma previsão que considere mais de um modelo de regressão.  
**models_to_stack**  
* Uma lista no mesmo formato utilizado para dar fit em models. Se utilizada, o modelo só ira realizar o stack dos modelos especificados. Se "fit_models()" foi utilizado, a funcao ira utilizar os modelos otimizados. Se não, irá utilizar os modelos salvos. 

**verbose**
* padrão = True, exibe os logs do optuna


**get_Stacked_Model(self)**

* Retorna, se, e somente se, **stack_models()** foi utilizado anteriormente, o modelo *sklearn* criado pelo método de Stacking 

** stack_model_tune(self, n_trials, model = False )

* Otimiza o modelo de regressao que sera utilizado para fazer o stcking
* n_trials: Número de trials para otimização
* model: Se não especificado, utiliza o modelo de regressao ridgeCV. 
**models_performace(self)**

* Retorna um *DataFrame* do módulo *pandas* com a avaliação das métricas: R2, MAE e RMSE respectivo dos desempenhos de cada modelo treinado

* Exemplo: 

```python
a = RegModels(modo="load", X_trainS=X_train,
                X_testS=X_test, y_trainS=y_train, y_testS=y_test)
minhaLista = {'Lasso': 15,
              'Ridge': 15, 
              'Elastic_Net': 10 }
a.fit_models(minhaLista)

table = a.models_performace()
print(table)

MAE      MAPE      RMSE        R2
Lasso.sav        1.845159  3.889345  2.563748  0.615375
Ridge.sav        1.822004  4.025883  2.518531  0.628823
Elastic_Net.sav  1.821933  4.033852  2.519516  0.628532
```


**visualize_slice(self, model_name, param_list=False)**

* Função para visualizar o desempenho dos hiperparâmetros do modelo na otimização. 

* model_name: Nome do modelo para visualização. (Mesmo nome utilizado na lista de fit_models()),  




### Modelos

### Definindo hiperparâmetros

Par editar os **hiperparâmetros no código fonte, basta encontrar sua funçao de objetivo no padrão: obj_Modelo().** 

As sugestões e ranges de hiperparâmetros são definidas utilizado funções do [optuna](https://optuna.readthedocs.io/en/stable/): 

```python
parameterFloat = trial.suggest_float("parameter's name", 0.1, 1.0)
parameterInteger = trial.suggest_int("parameter's name", 100, 500)
parameterCategoricalNumeric = trial.suggest_categorical("parameter's name", [1, 2, 4])
parameterCategoricalString = trial.suggest_categorical("nparameter's name", ["a", "b", "c"])
```

Assim, o próximo passo é aplicar esses hiperparâmetros na chamada do modelo, como no exemplo em que o *alpha* de modelo de Regressão Lasso é sugerido pelo *optuna* na instanciação do modelo: 

```python
def obj_lasso(trial):
    suggest_alpha = trial.suggest_float("alpha", 0.1, 1.0, log=True)
    model = linear_model.Lasso(alpha=suggest_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)
```

### Adicionando modelos

Para adicionar novos modelos de regressão (padrão sklearn), deve-se, primeiramente, adicionar uma nova função objetivo assim como as que já existem no código. O padrão a ser utilizado pode ser copiado a partir do seguinte exemplo: 

```python
def obj_model_name(trial):
    suggest_alpha = trial.suggest_float("alpha", 0.1, 1.0)
    model = modelObject(alpha=h_alpha)
    trial.set_user_attr(key="best_model", value=model)

    return score_method(model, trial)
```

O próximo passo é mudar a lista de **standardModels**, que contem todos os modelos que estão disponíveis no código. 

No arquivo de Reg_Models.py, altere a lista atribuida à variável **self.standardModels** com o nome do modelo e a sua respectiva função que foi adicionada**:** 

```python
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
                                'KNN_Regressor': obj_KNeighborsRegressor,
																'New model name': obj_model_name,}

```
