import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
import warnings


warnings.filterwarnings(action='ignore')

df = pd.read_csv("C:\\Users\\41muh\\OneDrive\\Masaüstü\\inovasyon-aws-bigdata-task\\data\\diabetes.csv")
print(df.head())

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf_model= RandomForestClassifier(random_state=17)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: % {:10.2f}".format(accuracy * 100))


joblib.dump(rf_model, "saved_models/01.rf_model.pkl")


rf_model_loaded = joblib.load("saved_models//01.rf_model.pkl")



X_manual_test = [[6,148,72,35,0,33.6,0.627,50]]
print("X_manual_test", X_manual_test)

prediction_raw = rf_model_loaded.predict(X_manual_test)
print("prediction_raw", prediction_raw)

