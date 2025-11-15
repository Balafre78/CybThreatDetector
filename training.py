from preprocessing import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Split features (X) and label (y)
X = df_cleaned.iloc[:, :-1]
y = df_cleaned.iloc[:, -1]

# 80% train / 20% test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=39, shuffle=True, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# First model : decision tree classifier
dt_model = DecisionTreeClassifier(
    max_depth=None,
    random_state=39
)

dt_model.fit(X_train, y_train)

# Second model : random forest classifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=39
)

rf_model.fit(X_train, y_train)

### Visualization

plt.figure(figsize=(20, 12))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=y.unique(),
    filled=True,
    max_depth=3
)
plt.title("Decision Tree (depth=3)")
plt.show()

def plot_feature_importance(model, X, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns

    plt.figure(figsize=(12, 10))
    sns.barplot(x=importances[indices][:20], y=features[indices][:20])
    plt.title(title + " - Top 20 Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

plot_feature_importance(dt_model, X_train, "Decision Tree")
plot_feature_importance(rf_model, X_train, "Random Forest")