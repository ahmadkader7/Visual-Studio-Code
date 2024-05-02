import Features as feat
import matplotlib.pyplot as plt
import numpy as np
import Utilities as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd
import xgboost as xgb
from sklearn.tree import plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score


def RandomForestTree(localFeatures, graphFeatures, heuristicsFeature, krng, sumHeuristicsFeature, mstFeature, targets):
    nearn, farins, randins, nearins, cheapins, mst, christo = heuristicsFeature[:7]
    (
        localfeature1,
        localfeature2,
        localfeature3,
        localfeature4,
        localfeature5,
        localfeature6,
    ) = localFeatures[:6]
    graphFeature1, graphFeature2, graphFeature3, graphFeature4 = graphFeatures[:4]

    X = pd.DataFrame(
        {
            "localfeature1": localfeature1,
            "localfeature2": localfeature2,
            "localfeature3": localfeature3,
            "localfeature4": localfeature4,
            "localfeature5": localfeature5,
            "localfeature6": localfeature6,
            "graphFeature1": graphFeature1,
            "graphFeature2": graphFeature2,
            "graphFeature3": graphFeature3,
            "graphFeature4": graphFeature4,
            "nearn": nearn,
            "farins": farins,
            "randins": randins,
            "nearins": nearins,
            "cheapins": cheapins,
            "mst": mst,
            "christo": christo,
            "krng": krng,
            "sumHeuristicsFeature": sumHeuristicsFeature,
            "mstFeature": mstFeature,
        }
    )
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        criterion="gini",
    )
    model.fit(X_train, y_train)
    
    y_predicted = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)

    print("Test accuracy is: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1)
    
    cm = confusion_matrix(y_test, y_predicted)

    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()
    return model, X, cm


def xGBoostClassifier(localFeatures,graphFeatures,heuristicFeatures,krng,sumHeuristicsFeature,mstFeature,targets,feature_names):
    nearn, farins, randins, nearins, cheapins, mst, christo = heuristicFeatures[:7]
    (
        localfeature1,
        localfeature2,
        localfeature3,
        localfeature4,
        localfeature5,
        localfeature6,
    ) = localFeatures[:6]
    graphFeature1, graphFeature2, graphFeature3, graphFeature4 = graphFeatures[:4]
    X = pd.DataFrame(
        {
            "localfeature1": localfeature1,
            "localfeature2": localfeature2,
            "localfeature3": localfeature3,
            "localfeature4": localfeature4,
            "localfeature5": localfeature5,
            "localfeature6": localfeature6,
            "graphFeature1": graphFeature1,
            "graphFeature2": graphFeature2,
            "graphFeature3": graphFeature3,
            "graphFeature4": graphFeature4,
            "nearn": nearn,
            "farins": farins,
            "randins": randins,
            "nearins": nearins,
            "cheapins": cheapins,
            "mst": mst,
            "christo": christo,
            "krng": krng,
            "sumHeuristicsFeature": sumHeuristicsFeature,
            "mstFeature": mstFeature,
        }
    )
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2)
    dM_X_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dM_X_test = xgb.DMatrix(X_test, feature_names=feature_names)
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "scale_pos_weight": 1,
        "objective": "binary:logistic",
    }
    model = xgb.train(params, dM_X_train, num_boost_round=1000)
    y_predicted = model.predict(dM_X_test)
    accuracy = accuracy_score(y_test, (y_predicted > 0.5).astype(int))
    print(f"Accuracy: {accuracy:.2f}")
    print(
        "Classification Report:\n",
        classification_report(y_test, (y_predicted > 0.5).astype(int)),
    )
    cm = confusion_matrix(y_test, (y_predicted > 0.5).astype(int))
    print("Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()
    return model


def rft_feature_importance(X, model, feature_names):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance in Random Forest")
    plt.show()


def xgb_feature_importance(model):
    feature_importances = model.get_score(importance_type="weight")
    sorted_feature_importances = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )
    features, importance_values = zip(*sorted_feature_importances)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importance_values, align="center")
    plt.xticks(range(len(features)), features, rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance in XGBoost")
    plt.show()

def visualize_decision_tree(model, feature_names, max_depth=None):
    tree_to_visualize = model.estimators_[0]

    plt.figure(figsize=(20, 10))
    class_names = list(map(str, np.unique(targets)))
    plot_tree(
        tree_to_visualize,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
    )
    plt.show()

mstFeature = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\mstFeature.npy"
)
localfeatures = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\localfeatures.npy"
)
graphFeatures = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\graphFeatures.npy"
)
heuristicsFeature = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\heuristicsFeature.npy"
)
krngFeature = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\krngFeature.npy"
)
sumHeuristicsFeature = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\sumHeuristicsFeature.npy"
)
targets = np.load(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\targets.npy"
)
feature_names = [
    "localfeature1",
    "localfeature2",
    "localfeature3",
    "localfeature4",
    "localfeature5",
    "localfeature6",
    "graphFeature1",
    "graphFeature2",
    "graphFeature3",
    "graphFeature4",
    "nearn",
    "farins",
    "randins",
    "nearins",
    "cheapins",
    "mst",
    "christo",
    "krng",
    "sumHeuristicsFeature",
    "mstFeature",
]
"""
vertices = ut.listNodes()
edges_optimal = ut.listToTuple(ut.listOptimalTour())
edges_greedy = ut.listToTuple(ut.listGreedyTour())

edges_nearn = ut.listToTuple(ut.listNearnTour())
edges_farins = ut.listToTuple(ut.listFarinsTour())
edges_randins = ut.listToTuple(ut.listRandinsTour())
edges_nearins = ut.listToTuple(ut.listNearinsTour())
edges_cheapins = ut.listToTuple(ut.listCheapinsTour())
edges_mst = ut.listToTuple(ut.listMSTTour())
edges_christo = ut.listToTuple(ut.listChristoTour())

localfeatures = feat.localFeatures(vertices, edges_greedy)
graphFeatures = feat.graphFeatures(vertices, edges_greedy)
heuristicsFeature = feat.heuristicsFeature(
    edges_greedy,
    edges_nearn,
    edges_farins,
    edges_randins,
    edges_nearins,
    edges_cheapins,
    edges_mst,
    edges_christo,
)
sumHeuristicsFeature = feat.sumHeuristicsFeature(
    heuristicsFeature[0],
    heuristicsFeature[1],
    heuristicsFeature[2],
    heuristicsFeature[3],
    heuristicsFeature[4],
    heuristicsFeature[5],
    heuristicsFeature[6],
)
krngFeature = feat.kRNGFeature(vertices, edges_greedy)
targets = feat.calculateTargets(edges_optimal, edges_greedy)
values = feat.mstFeature(vertices, edges_greedy)


np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\localfeatures.npy",
    localfeatures,
)
np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\graphFeatures.npy",
    graphFeatures,
)
np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\heuristicsFeature.npy",
    heuristicsFeature,
)
np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\sumHeuristicsFeature.npy",
    sumHeuristicsFeature,
)
np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\krngFeature.npy",
    krngFeature,
)
np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\targets.npy",
    targets,
)
np.save(
    "C:\\Users\\ahmad\\Documents\\Visual Studio Code\\Workspace\\TSP\\mstFeature.npy",
    values,
)
"""
model, X, cm = RandomForestTree(
    localfeatures,
    graphFeatures,
    heuristicsFeature,
    krngFeature,
    sumHeuristicsFeature,
    mstFeature,
    targets,
)

visualize_decision_tree(model, feature_names, max_depth=3)
rft_feature_importance(X, model, feature_names)


xgboost = xGBoostClassifier(
    localfeatures,
    graphFeatures,
    heuristicsFeature,
    krngFeature,
    sumHeuristicsFeature,
    mstFeature,
    targets,
    feature_names,
)
xgb_feature_importance(xgboost)