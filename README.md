# AutoMLPratica
O seguinte código foi com o intuito de aprender e explorar um pouco sobre técnicas de AutoML (Automated Machine Learning)

# Auto ML com AutoKeras e Scikit-Learn

Este repositório contém um script em Python que executa uma análise automatizada de aprendizado de máquina (AutoML) para prever preços de habitação na Califórnia, utilizando o conjunto de dados "California Housing" do `scikit-learn`. O script implementa uma solução de regressão com o `AutoKeras`, uma biblioteca de AutoML, facilitando a criação e a otimização de modelos de machine learning sem a necessidade de ajuste manual dos hiperparâmetros.

## Requisitos

Para rodar o script, certifique-se de que as bibliotecas abaixo estejam instaladas:

- Python 3.x
- scikit-learn
- pandas
- numpy
- AutoKeras (versão mais recente)

Instale as dependências com:

```bash
pip install scikit-learn pandas numpy autokeras
```

## Descrição do Código

1. **Importação dos Dados**:
   O conjunto de dados `California Housing` é importado usando `scikit-learn`. Ele inclui informações de várias regiões da Califórnia e é comumente usado para modelos de regressão.

   ```python
   from sklearn.datasets import fetch_california_housing
   housing = fetch_california_housing()
   x = housing.data
   y = housing.target
   ```

2. **Divisão dos Dados**:
   O conjunto de dados é dividido em `train` e `test`, com 20% dos dados reservados para testes.

   ```python
   from sklearn.model_selection import train_test_split
   xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
   ```

3. **Configuração do AutoKeras**:
   O `AutoKeras` é configurado para resolver uma tarefa de regressão. O nó de entrada é definido genericamente, e o nó de saída é configurado para uma métrica de erro quadrático médio (MSE) para otimização.

   ```python
   import autokeras as ak
   input_node = ak.Input()
   output_node = ak.RegressionHead(loss='mean_squared_error', metrics=['mean_squared_error'])
   ```

4. **Criação do Modelo Automático**:
   O modelo `AutoModel` do AutoKeras é criado com as entradas e saídas configuradas, com um limite de `max_trials=4` (número de tentativas para diferentes modelos) e com a métrica `val_mean_squared_error` como objetivo de otimização.

   ```python
   automl = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=4, objective='val_mean_squared_error', overwrite=True, seed=42)
   automl.fit(xtrain, ytrain, epochs=2)
   ```

5. **Acessando Resultados do Tuner**:
   Após o treinamento, os detalhes de cada `trial` (tentativa de modelo) são extraídos e organizados em um `DataFrame` para análise.

   ```python
   tuner = automl.tuner
   trials = tuner.oracle.trials
   ```

   Para cada `trial`, as métricas de desempenho e hiperparâmetros são extraídos e armazenados em uma tabela, permitindo ordenar os resultados conforme o valor da métrica de erro.

   ```python
   trial_data.append({
       'Trial ID': trial_id,
       'Score': score,
       'Hyperparameters': hparams
   })
   leaderboard = pd.DataFrame(trial_data).sort_values(by='Score', ascending=True).reset_index(drop=True)
   ```

6. **Avaliação do Modelo**:
   O modelo treinado é testado no conjunto de teste, e o erro quadrático médio (MSE) e a raiz do erro quadrático médio (RMSE) são calculados para avaliar a precisão.

   ```python
   ypred = automl.predict(xtest)
   mse = mean_squared_error(ytest, ypred)
   rmse = np.sqrt(mse)
   ```

## Resultados

A tabela `leaderboard` exibe os modelos testados, juntamente com suas respectivas métricas e hiperparâmetros, permitindo uma análise comparativa dos melhores modelos. Além disso, os valores de MSE e RMSE calculados fornecem uma métrica de desempenho sobre o conjunto de dados de teste.

## Exemplos de Uso

Para rodar o script, basta executar:

```bash
python auto_ml.py
```

Este script exibirá no console o `leaderboard` dos modelos, juntamente com as métricas MSE e RMSE do melhor modelo treinado, indicando o desempenho sobre o conjunto de teste.

## Observações

- A quantidade de `max_trials` e o número de `epochs` podem ser ajustados conforme a capacidade computacional disponível para obter modelos com maior precisão.
- O AutoKeras facilita a criação de modelos complexos com um esforço mínimo, tornando-o ideal para projetos que demandam otimização rápida sem conhecimento aprofundado em ajuste de hiperparâmetros.
