# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import streamlit as slt

slt.title("Atividade com ChatGPT")

slt.header("Utilizando o CrossValidation")
slt.write("O objetivo da `CrossValidation` é avaliar o desempenho do modelo através de uma divisão do conjunto de dados em vários subconjuntos para que o modelo possa ser treinado e avaliado várias vezes em diferentes partições do conjunto de dados. Sendo o objetivo principal obter uma estimativa mais precisa do desempenho do modelo em dados não vistos.")

url = 'https://gist.githubusercontent.com/tonicprism/95bc1a6de11c9ede0530d250828d24b5/raw/8ae4d8cae2b6a957933956b1e17c9424f641e771/mobile_price_classification.csv'
data = pd.read_csv(url)

slt.subheader("As colunas da base de dados:")
slt.write(data.head())

slt.subheader("Plotando um histograma para todos os atributos numérico presente na base de dados.")
numeric_cols = data.select_dtypes(include='number').columns.tolist()

data[numeric_cols].hist(bins=20, figsize=(10,10))
st.pyplot()


slt.subheader("Definindo as variáveis independentes (X) e a variável dependente (y):")
X = data.drop('price_range', axis=1)
slt.write("variáveis independentes (X): ")
slt.write(X)

y = data['price_range']
slt.write("variável dependente (y): ")
slt.write(y)

"""Criando o modelo de classificação (por exemplo, DecisionTreeClassifier) e defina seus hiperparâmetros, se necessário:"""

clf = DecisionTreeClassifier(max_depth=3)

"""Executando a validação cruzada usando o cross_val_score:"""

scores = cross_val_score(clf, X, y, cv=10)

slt.subheader("Exibindo o resultado da melhoria feita no modelo utilizando o CrossValidation")
slt.write("A média e o desvio padrão das métricas de avaliação:")
slt.write("Acurácia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



slt.header("Utilizando o GridSearch")
slt.write("O objetivo da técnica `GridSearch` é otimizar os **hiperparâmetros** (parâmetros do modelo utilizado) através de uma busca exaustiva em uma grade de possíveis valores de hiperparâmetros (no exemplo a baixo será utilizado os hiperparâmetros do modelo `DecisionTreeClassifier`) para encontrar a combinação que produz o melhor desempenho do modelo.")

from sklearn.model_selection import train_test_split, GridSearchCV

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo a grade de parâmetros a ser testada
param_grid = {'criterion': ['gini', 'entropy'], 
              'max_depth': [2, 4, 6, 8, 10], 
              'min_samples_split': [2, 5, 10, 15, 20],
              'min_samples_leaf': [1, 2, 4, 6, 8]}

# Instanciando o GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=5)

# Treinando o modelo
grid_search.fit(X_train, y_train)

slt.subheader("Imprimindo os melhores parâmetros encontrados utilizando o GridSearch: ")
slt.write("Melhores parâmetros: ", grid_search.best_params_)

# Fazendo previsões no conjunto de teste
y_pred = grid_search.predict(X_test)

slt.subheader("Imprimindo a acurácia do modelo")
print("Acurácia: ", grid_search.score(X_test, y_test))


slt.header("Os resultados do uso das técnicas Cross `Validation` e `GridSearch` (Em gráfico)")

# Criando uma tabela com os resultados
results = pd.DataFrame(grid_search.cv_results_)

slt.subheader("Tabela com os resultados: ")
slt.write(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

correlation = data.corr()

slt.subheader("Gráfico com a correlação dos atributos da minha base de dado: ")
slt.write("Esse gráfico indica o grau de relação entre elas (quanto menor a porcentagem, menor o grau de relação entre o elemento da linha vertical e linha horizontal).")
slt.write(correlation)

slt.subheader("Historiograma com o resultado do uso das técnicas: ")
fig, ax = plt.subplots()
sns.heatmap(correlation, ax=ax, annot = True, fmt=".1%")
slt.write(fig)


slt.subheader("Gráfico de distribuição dos resultados: ")
fig, ax = plt.subplots()
sns.histplot(data=scores, ax=ax, kde=True)
slt.write(fig)