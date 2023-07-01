import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados em um DataFrame
data = pd.read_csv('C:/Users/User/Desktop/dataset.csv')

# Remover a coluna "SystemCodeNumber" e "LastUpdated"
data = data.drop(['SystemCodeNumber', 'LastUpdated'], axis=1)

# Dividir o conjunto de dados em atributos de entrada (X) e atributo classe (y)
X = data.drop('Occupancy', axis=1)
y = data['Occupancy']

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Executar pré-processamento (normalização)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Executar algoritmo de aprendizado de máquina
model = LogisticRegression()
model.fit(X_train, y_train)

# Avaliar o desempenho do modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)

# Calcular a contagem de ocorrências de cada rótulo na classe "Occupancy" do conjunto de teste
class_counts = y_test.value_counts()

# Obter o rótulo majoritário
majority_class = class_counts.idxmax()

# Calcular a proporção do erro majoritário
majority_error = 1 - class_counts[majority_class] / len(y_test)

# Exibir a classe majoritária e o erro majoritário
print("Classe Majoritária:", majority_class)
print("Erro Majoritário:", majority_error)

class_distribution = data['Occupancy'].value_counts()

# Exibir a distribuição dos dados no atributo classe
print("Distribuição dos dados no atributo classe:")
print(class_distribution)