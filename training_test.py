from preprocessing import *
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

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

def evaluate_model(model, X_test, y_test, model_name):
    print(f" MODEL EVALUATION : {model_name}")
    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {accuracy:.4f}")

    # F1 scores
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f" F1 Macro: {f1_macro:.4f}")
    print(f" F1 Weighted: {f1_weighted:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return y_pred

dt_pred = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
rf_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")

### Visualization
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
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

def plot_feature_importance(model, X, title, top_n=20):
    importances = model.feature_importances_
    index = np.argsort(importances)[::-1]
    df_plot = pd.DataFrame({"Feature": X.columns[index][:top_n], "Importance": importances[index][:top_n]})
    plt.figure(figsize=(12, 9))
    sns.barplot(data=df_plot, x="Importance", y="Feature", hue="Feature", palette="viridis", dodge=False, legend=False)
    plt.title(f"{title} - Top {top_n} Features", fontsize=18)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()
    plt.show()


plot_feature_importance(dt_model, X_train, "Decision Tree")
plot_feature_importance(rf_model, X_train, "Random Forest")