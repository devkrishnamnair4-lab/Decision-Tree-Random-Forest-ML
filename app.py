import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
print("Decision Tree Accuracy:", dt.score(X_test, y_test))

plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=X.columns, filled=True, fontsize=8)
plt.savefig("decision_tree.png")
plt.close()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", rf.score(X_test, y_test))

importances = rf.feature_importances_
print("Feature Importances:", importances)

cv = cross_val_score(rf, X, y, cv=5)
print("Cross Validation Score:", cv.mean())
