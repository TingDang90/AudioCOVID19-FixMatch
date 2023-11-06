from sklearn import metrics
import joblib
import numpy as np
import pandas as pd


def get_metrics(probs, labels):
    """calculate metrics.
    :param probs: list
    :type probs: float
    :param labels: list
    :type labels: int
    :return: metrics
    """
    probs = np.array(probs)
    probs = np.squeeze(probs)

    probs_of_true = []
    for i in range(len(probs)):
        probs_of_true.append(probs[i][np.argmax(labels[i])])

    predicted = []
    for i in range(len(probs)):
        if probs[i][0] <= 0.5:
            predicted.append(0)
        else:
            predicted.append(1)

    label = np.array(labels)
    label = np.squeeze(label)

    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)

    pre = metrics.precision_score(label, predicted)
    acc = metrics.accuracy_score(label, predicted)
    auc = metrics.roc_auc_score(label, probs_of_true)
    rec = metrics.recall_score(label, predicted)

    #TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
    TP, FN, FP, TN = metrics.confusion_matrix(label, predicted).ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # PPV = TP/(TP + FP)
    # NPV = TN/(TN + FN)
    
    # auc, sens, spec, acc, bal_acc, pre, F1
    return auc, TPR, TNR, acc, (TPR + TNR) / 2, pre, 2 * (pre * rec) / (pre + rec)


def get_CI(data):
    AUCs = []
    SENSs = []
    SPECs = []
    ACCs = []
    BALACCs = []
    PREs = []
    F1s = []
    for s in range(1000):
        np.random.seed(s)  # Para2
        sample = np.random.choice(range(len(data)), len(data), replace=True)
        samples = [data[i] for i in sample]
        sample_pro = [x[0] for x in samples]
        sample_label = [x[1] for x in samples]
        try:
            get_metrics(sample_pro, sample_label)
        except ValueError:
            np.random.seed(1001)  # Para2
            sample = np.random.choice(range(len(data)), len(data), replace=True)
            samples = [data[i] for i in sample]
            sample_pro = [x[0] for x in samples]
            sample_label = [x[1] for x in samples]
        else:
            auc, sens, spec, acc, bal_acc, pre, F1 = get_metrics(sample_pro, sample_label)
        AUCs.append(auc)
        SENSs.append(sens)
        SPECs.append(spec)
        ACCs.append(acc)
        BALACCs.append(bal_acc)
        PREs.append(pre)
        F1s.append(F1)

    auc_0 = pd.DataFrame(np.array(AUCs)).quantile(0.025)[0]  # 2.5% percentile
    auc_1 = pd.DataFrame(np.array(AUCs)).quantile(0.975)[0]  # 97.5% percentile

    sens_0 = pd.DataFrame(np.array(SENSs)).quantile(0.025)[0]  # 2.5% percentile
    sens_1 = pd.DataFrame(np.array(SENSs)).quantile(0.975)[0]  # 97.5% percentile

    spec_0 = pd.DataFrame(np.array(SPECs)).quantile(0.025)[0]  # 2.5% percentile
    spec_1 = pd.DataFrame(np.array(SPECs)).quantile(0.975)[0]  # 97.5% percentile

    acc_0 = pd.DataFrame(np.array(ACCs)).quantile(0.025)[0]  # 2.5% percentile
    acc_1 = pd.DataFrame(np.array(ACCs)).quantile(0.975)[0]  # 97.5% percentile

    bal_acc_0 = pd.DataFrame(np.array(BALACCs)).quantile(0.025)[0]  # 2.5% percentile
    bal_acc_1 = pd.DataFrame(np.array(BALACCs)).quantile(0.975)[0]  # 97.5% percentile

    pre_0 = pd.DataFrame(np.array(PREs)).quantile(0.025)[0]  # 2.5% percentile
    pre_1 = pd.DataFrame(np.array(PREs)).quantile(0.975)[0]  # 97.5% percentile

    F1_0 = pd.DataFrame(np.array(F1s)).quantile(0.025)[0]  # 2.5% percentile
    F1_1 = pd.DataFrame(np.array(F1s)).quantile(0.975)[0]  # 97.5% percentile

    return [auc_0, auc_1], [sens_0, sens_1], [spec_0, spec_1], [acc_0, acc_1], [bal_acc_0, bal_acc_1], [pre_0, pre_1], [F1_0, F1_1]

def plot_roc_curve(labs, probs):
    """ Plot ROC curve """
    ac_labs, ac_probs = labs, []
    for i in range(len(labs)):
        ac_probs.append(probs[i][0])
    fpr, tpr, _ = metrics.roc_curve(ac_labs, ac_probs)
    print("AUC IS", metrics.roc_auc_score(ac_labs, ac_probs))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("./roc_curve.png")

# Get predictions
with open("pred.pk", "rb") as f:
    preds = joblib.load(f)

# Get all metrics from predictions, reporting CI
test_probs, test_labels = [], []
for sample in preds["test"]:
    test_probs.append(sample["logits"][0])
    test_labels.append(sample["labels"][0])
met_names = ["auc", "sens", "spec", "acc", "bal_acc", "pre", "F1"]
mets = get_metrics(test_probs, test_labels)
metrics_CIs = get_CI(list(zip(test_probs, test_labels)))
for i in range(len(mets)):
    print(met_names[i] + ": " + str(round(mets[i], 2)) + "(" + str(round(metrics_CIs[i][0], 2)) + "-" + str(round(metrics_CIs[i][1], 2)) + ")")

# Plot ROC curve
import matplotlib.pyplot as plt

plt.clf()
plot_roc_curve(test_labels, test_probs)
plt.savefig("./roc_curve.png")

# Print number of unlabelled samples included, pos+neg
conf_pos_count, conf_neg_count = 0, 0
pred_pos_count, pred_neg_count = 0, 0
true_pos_count, true_neg_count = 0, 0
for sample in preds["unlabelled"]:
    if sample["logits"][0][0] <= 0.5:  # label [0,1] is pos
        pred_pos_count += 1
        if sample["logits"][0][1] >= 0.95:
            conf_pos_count += 1
    else:
        pred_neg_count += 1
        if sample["logits"][0][0] >= 0.95:
            conf_neg_count += 1
    
    if sample["labels"][0] <= 0.5:
        true_pos_count += 1
    else:
        true_neg_count += 1

print("Confident pos num:", conf_pos_count, "Confident neg num:", conf_neg_count)
print("Predicted pos num:", pred_pos_count, "Predicted neg num:", pred_neg_count)
print("True pos num:", true_pos_count, "True neg num:", true_neg_count)

