from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dvclive import Live

print("getting params")
params = yaml.safe_load(open("params.yaml"))["train"]

print("reading data")
# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

print("starting experiment")
with Live(save_dvc_exp=False) as live:
    print("logging params")
    live.log_params(params)
    # Fit a model
    depth = params.get("depth")
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    with open("metrics.txt", "w") as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")

    # Plot it
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
    )
    plt.savefig("plot.png")
    print("Logging confusion matrix")
    live.log_sklearn_plot("confusion_matrix", X_test, y_test)
    print("Finished logging")

print("Done. Exiting.")
    
