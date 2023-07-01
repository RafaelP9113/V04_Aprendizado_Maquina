import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Carregar o conjunto de dados em um DataFrame
data = pd.read_csv('C:/Users/User/Desktop/dataset.csv')

# Remover a coluna "SystemCodeNumber" e "LastUpdated"
data = data.drop(['SystemCodeNumber', 'LastUpdated'], axis=1)

# Dividir o conjunto de dados em atributos de entrada (X) e atributo classe (y)
X = data.drop('Occupancy', axis=1)
y = data['Occupancy']

# Executar pré-processamento (normalização)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir os modelos de aprendizado de máquina
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'k-NN': KNeighborsClassifier()
}

# Avaliar os modelos utilizando validação cruzada e métricas de avaliação
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1-Score': f1_score,
    'AUC-ROC': roc_auc_score
}

results = {}
for model_name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results[model_name] = scores.mean()

# Selecionar o melhor modelo com base na acurácia média
best_model = max(results, key=results.get)

# Treinar o melhor modelo com o conjunto de dados completo
model = models[best_model]
model.fit(X, y)

# Imprimir os resultados
print("Resultados dos modelos:")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")

print("\nMelhor modelo:", best_model)
