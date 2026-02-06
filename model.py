import torch
from getting_Model_Ready import getting_Model_Ready
from cleaning import clean_Climate_Change_Dataset
from sklearn.model_selection import train_test_split

df = clean_Climate_Change_Dataset()

X, y = getting_Model_Ready(df=df)


def model(X: torch.tensor = X, y: torch.tensor = y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)


