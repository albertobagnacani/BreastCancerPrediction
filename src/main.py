# %%

###
# Same code of the notebook
###

import numpy as np  # Array computing
import pandas as pd  # Data structure
import matplotlib.pyplot as plt  # Plots
import seaborn as sns  # Data visualization
from sklearn.metrics import recall_score, precision_score

# Read the data
df = pd.read_csv("../res/data/data.csv")

# %%

# 1) Quick look at the data structure

# 1.1) Get insights

# First 5 rows of data
df.head()

# %%

# Columns of the dataset
df.columns

# %%

# Statistic about the data
df.describe()

# %%

# Dimensionality of the dataset
df.shape

# %%

# Summary of the dataset
df.info()

# %%

# Histogram of all features
# %matplotlib inline
# data.hist(bins=50, figsize=(20, 15))
# plt.show()

# %%

# 1.2) Discover and visualize the data

# %%

# 1.3) Look for correlations between features (not target) for feature selection/extraction

# %%

# 1.4) Attribute combinations

# %%

# 2) Prepare the data for ML algorithms

# 2.1) Categorical variables conversion and data cleaning
# Convert diagnosis (B(enign)=0, M(alignant)=1) (before dividing features from labels)
df["diagnosis"] = df["diagnosis"].astype('category').cat.codes

# Remove useless columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# %%

# 2.2) Feature scaling
# TODO

# %%

# 2.3) Train/test set creation

# Divide features from labels
X = df.drop('diagnosis', axis=1)
y = df.diagnosis

# Create a stratified train and test set
from sklearn.model_selection import train_test_split  # sklearn: ML library

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                    stratify=y)  # 80% train set, 20% test set

# %%

# 2.4) Look for correlations with the target
corr_matrix = df.corr()
corr_matrix["diagnosis"].sort_values(ascending=False)  # Correlation with the target

# %%

# 2.5) Custom trasformers and trasformation pipelines

# %%

# 3) Select and train a model
predictions = {}
metrics = {}
# specificity
# f_score
# cross_val_Score

# %%

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# ROC_AUC score
def roc_auc_for_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]  # probs of positive class
    auc_s = roc_auc_score(y_test, probs)
    return auc_s, probs


# Plot the roc curve
def plot_roc_curve(name, fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("../res/plots/roc_" + name + ".png")
    plt.show()


# %%

def get_metrics(model_name, model, X_test, y_test):
    auc_s, probs = roc_auc_for_model(model, X_test, y_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plot_roc_curve(model_name, fpr, tpr)

    metrics[model_name] = {}
    metrics[model_name]["accuracy"] = model.score(X_test, y_test)
    metrics[model_name]["recall"] = recall_score(y_test, y_pred)
    metrics[model_name]["precision"] = precision_score(y_test, y_pred)
    metrics[model_name]["roc_auc"] = auc_s


# %%

# def plotDecisionBoundaries():

# 3.1) Logistic Regression
from sklearn.linear_model import LogisticRegression

model_name = "Logistic_Regression"
model = LogisticRegression(solver="liblinear")  # liblinear ideal for small dataset
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions[model_name] = y_pred

get_metrics(model_name, model, X_test, y_test)

print(metrics[model_name])

# %%

# cross_val_score
from sklearn.model_selection import cross_val_score

print("Cross validation score with several folds: ")
print(cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy"))

# %%

# 3.3) KNN Classification
from sklearn.neighbors import KNeighborsClassifier


def best_n_of_neighbors(neighbors, score_list, X_train, y_train, X_test, y_test):
    for i in range(1, neighbors):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        score_list.append(score)

    return score_list.index(max(score_list)) + 1


def knn_score_plot(neighbors, score_list):
    plt.plot(range(1, neighbors), score_list)
    plt.xticks(np.arange(1, neighbors, 1))
    plt.title('KNN scoring (accuracy) vs number of neighbors')
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.savefig("../res/plots/knn_neighbors.png")
    plt.show()


neighbors = 20
score_list = []
model_name = "K-Nearest_Neighbors"
model = KNeighborsClassifier(n_neighbors=best_n_of_neighbors(neighbors, score_list, X_train, y_train, X_test, y_test))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions[model_name] = y_pred

get_metrics(model_name, model, X_test, y_test)

print(metrics[model_name])

knn_score_plot(neighbors, score_list)

# %%

# Grid search
from sklearn.model_selection import GridSearchCV

cv = 5


def grid_search(model, param_grid, cv):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train, y_train)
    # grid_search.best_params_
    print(grid_search.best_estimator_)
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #    print(np.sqrt(-mean_score), params)


# %%

# 3.4.2) SVM with feature scaling and grid search
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model_name = "Support-Vector_Machines_scaling_grid"
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=0.1, probability=True))
    # Alternatives: LinearSVC(C=0.1, loss="hinge", max_iter=5000) [need of the CalibratedClassifier for proba, but faster than SVC with linear kernel], SGDClassifier(loss="hinge", alpha=1/(m*C))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions[model_name] = y_pred

get_metrics(model_name, model, X_test, y_test)

print(metrics[model_name])

# %%

# Grid search for SVM
param_grid = [
    {'svc__C': [0.1, 0.2, 0.4, 0.6, 0.8, 1]}
    # lsvc name in the pipeline, double underscore means to selet that parameter annotated after it
]

grid_search(model, param_grid, cv)

# %%

# 3.5) Naive Bayes
from sklearn.naive_bayes import GaussianNB

model_name = "Naive_Bayes"
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions[model_name] = y_pred

get_metrics(model_name, model, X_test, y_test)

print(metrics[model_name])

# %%

# 3.6) Decision Tree
from sklearn.tree import DecisionTreeClassifier

model_name = "Decision_Tree"
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions[model_name] = y_pred

get_metrics(model_name, model, X_test, y_test)

print(metrics[model_name])

# %%

# 3.7.2) Random Forest with grid search
from sklearn.ensemble import RandomForestClassifier

model_name = "Random_Forest_grid"
model = RandomForestClassifier(bootstrap=False, max_features=6, n_estimators=100, warm_start=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions[model_name] = y_pred

get_metrics(model_name, model, X_test, y_test)

print(metrics[model_name])

# %%

# Grid search for Random Forest
param_grid = [
    # {'n_estimators': [3, 10, 50, 100, 200, 500], 'max_features': [2, 6, 10]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 50, 100, 200, 500], 'max_features': [2, 6, 10]},
    {'warm_start': [True], 'n_estimators': [3, 10, 50, 100, 200, 500], 'max_features': [2, 6, 10]}
]

grid_search(model, param_grid, cv)


# %%

# 3.8) Fine-tuning

# %%

# 3.9) Comparison
def metric_names(metric):
    metric_names = []
    for k, v in metrics.items():
        for k1, v1 in v.items():
            metric_names.append(k1)

    metric_names = list(dict.fromkeys(metric_names))
    return metric_names


def metric_values_from_name(metrics, models, name):
    res = []
    for model in models:
        res.append(metrics[model][name])

    return res


def plot_metric(metric_name, models, values):
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 5))
    plt.title(metric_name.capitalize() + " plot of different models")
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel(metric_name.capitalize())
    plt.xlabel("Models")
    # sns.barplot(x=list(metric.keys()), y=list(metric.values()))
    sns.barplot(x=list(models), y=values)
    plt.savefig("../res/plots/" + metric_name + ".png")
    plt.show()


# do in a more functional way
print(metrics)
models = metrics.keys()
metric_names = metric_names(metrics)

for name in metric_names:
    plot_metric(name, models, metric_values_from_name(metrics, models, name))


# %%

# 3.10) Confusion matrix
def plot_confusion_matrices(cms):
    plt.figure(figsize=(24, 12))

    plt.suptitle("Confusion matrices of different models", fontsize=24)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    i = 1
    for model_name in cms.keys():
        plt.subplot(2, 3, i)
        plt.title(model_name + " confusion matrix")
        sns.heatmap(cms[model_name])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        i += 1

    plt.savefig("../res/plots/confusion_matrix.png")
    plt.show()


cms = {}
from sklearn.metrics import confusion_matrix

for model in models:
    cms[model] = confusion_matrix(y_test, predictions[model])

plot_confusion_matrices(cms)
