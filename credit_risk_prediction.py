import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score





np.random.seed(42)


data={
    "Age": np.random.randint(20,65,1000),
    "Sex": np.random.choice(["male","female"],1000),
    "Job":np.random.randint(0,4,1000),
    "Housing":np.random.choice(["own", "rent","free"],1000),
    "Saving accounts":np.random.choice(["Little", "Moderate", "Quite Rich", "Rich"],1000),
    "Cheking account":np.random.randint(0,20000,1000),
    "Credit amount":np.random.randint(500,20000,1000),
    "Duraiton":np.random.randint(6,72,1000),
    "Purpose":np.random.choice(["Car", "Education", "Buisness", "Vacation" , "Repairs"],1000),

}


df = pd.DataFrame(data)


print(df.head())

df.to_csv("generated_credit_risk_data.csv", index=False)

df=pd.read_csv("generated_credit_risk_data.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

df = pd.get_dummies(df, columns=["Sex","Housing","Saving accounts","Purpose"],drop_first=True)

print(df.head())


X=df.drop(columns=["Credit amount"])
y=df["Credit amount"]

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu",X_test.shape)

scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled , y_train)

y_pred = model.predict(X_test_scaled)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Absolute Error (MAE) :",mae)
print("Mean Squared Error (MSE) :",mae)
print("R2 Score:",mae)


new_customer= np.array([[50,3,1,20000,2,0,0,1,0,1,0,1, 0, 0]])
new_customer_scaled =scaler.transform(new_customer)

predicted_credit =model.predict(new_customer_scaled)

print("Yeni müşteri için tahmin edilen kredi miktarı :", predicted_credit[0])



