import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("healthstatus6.csv")

def filter(x):
    if x < 6.5:
        return 'good'
    if x >= 6.5:
        return 'poor'
df['glu']=df['hba1c'].apply(filter)
    
X = df[["sbp", "dbp", "smoking"]]
X = X.replace(["Yes", "No"], [1, 0])
y = df["glu"]

clf = LogisticRegression() 
clf.fit(X, y)


import joblib

joblib.dump(clf, "hstat.pkl")
