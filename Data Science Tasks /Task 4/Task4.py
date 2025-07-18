import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Task 4/Advertising.csv")
df = df.drop(columns=["Unnamed: 0"])

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nAdvertising Impact on Sales:")
print(coeff_df)

plt.figure(figsize=(10, 5))
sns.regplot(x=y_test, y=y_pred, scatter_kws={
            "color": "blue"}, line_kws={"color": "red"})
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
