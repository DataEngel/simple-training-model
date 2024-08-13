import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Generar datos dummy
np.random.seed(42)
X_dummy = np.random.rand(150, 4)  # 150 ejemplos, 4 características (igual que Iris)
y_dummy = np.random.randint(0, 3, 150)  # 150 etiquetas de clase (3 clases, igual que Iris)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

# Entrenar un modelo de XGBoost con datos dummy
dtrain = xgb.DMatrix(X_train, label=y_train)
param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'num_class': 3
}
model = xgb.train(param, dtrain, num_boost_round=10)

# Guardar el modelo
model.save_model('model.bst')
