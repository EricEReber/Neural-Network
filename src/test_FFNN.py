from FFNN import FFNN
from Schedulers import *
from activationFunctions import *
from costFunctions import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

cancer_dataset = load_breast_cancer()
cancer_X = cancer_dataset.data
cancer_t = cancer_dataset.target
cancer_t = cancer_t.reshape(cancer_t.shape[0], 1)

epochs = 200
rho = 0.9
rho2 = 0.999
adam_eta = 1e-3
adam_lambda = 1e-4
batches = 10

np.random.seed(1337)
X_train, X_val, t_train, t_val = train_test_split(cancer_X, cancer_t)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

dims = (X_train.shape[1], 20, 20, 1)
neural = FFNN(
    dims, hidden_func=LRELU, output_func=sigmoid, cost_func=CostLogReg, seed=1337
)

scores = neural.fit(
    X_train,
    t_train,
    Adam(adam_eta, rho, rho2),
    lam=adam_lambda,
    epochs=epochs,
    batches=batches,
    X_val=X_val,
    t_val=t_val,
)
