from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

x = housing.data
y = housing.target

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

!pip install autokeras

import autokeras as ak
print(ak.__version__)

!pip install autokeras --upgrade

import autokeras as ak

# Definir o nó de entrada genérico
input_node = ak.Input()

# Definir o nó de saída para regressão
output_node = ak.RegressionHead(
    loss='mean_squared_error',
    metrics=['mean_squared_error']
)

# Criar o modelo automático usando AutoModel
automl = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    max_trials=4,
    objective='val_mean_squared_error',
    overwrite=True,
    seed=42
)

automl.fit(xtrain, ytrain, epochs=2)

import pandas as pd

# Acessar o tuner
tuner = automl.tuner

# Acessar os trials
trials = tuner.oracle.trials

# Criar uma lista para armazenar os dados dos trials
trial_data = []

for trial_id, trial in trials.items():
    # Obter a métrica de interesse do trial
    score = trial.score  # Pode ser None em alguns casos
    # Se o score for None, você pode acessar o valor da melhor métrica
    if score is None:
        # Substitua 'val_mean_squared_error' pela sua métrica de interesse
        score = trial.metrics.get_best_value('val_mean_squared_error')

    # Obter os hiperparâmetros usados no trial
    hparams = trial.hyperparameters.values
    # Armazenar os dados
    trial_data.append({
        'Trial ID': trial_id,
        'Score': score,
        'Hyperparameters': hparams
    })

# Criar um DataFrame com os resultados
leaderboard = pd.DataFrame(trial_data)

# Ordenar a tabela pelos scores (ajuste ascending conforme necessário)
leaderboard = leaderboard.sort_values(by='Score', ascending=True).reset_index(drop=True)

# Exibir a leaderboard
print(leaderboard)

import numpy as np

ypred = automl.predict(xtest)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)


print("MSE:", mse)
print("RMSE:", rmse)

