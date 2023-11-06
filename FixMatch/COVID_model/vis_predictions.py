import joblib
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import os

"""
Labelled pos vs neg
Labelled sympos, asympos, symneg, asymneg
Test pos vs neg
Test sympos, asympos, symneg, asymneg
Unlabelled pos vs neg
All data
"""
WITH_SYM = False

def flatten_data(pred_data):
    all_latents = []
    all_labels = []
    for data_type in ["labelled", "unlabelled", "test"]:
        for sample in pred_data[data_type]:
            all_latents.append(sample["latent"][0])
            all_labels.append(data_type + ("_pos" if sample["labels"][0] == 0 else "_neg") + (("_" + sample["sym"]) if WITH_SYM and data_type != "unlabelled" else ""))

    return all_latents, all_labels


def get_tsne(latents, labels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(latents)
    plt.figure(figsize=(16,10))
    data = pd.DataFrame({"tsne1": tsne_result[:,0], "tsne2": tsne_result[:,1], "label": labels})
    #data = data.loc[data.pca1 < 25]
    return data

def vis_tsne(data):
    # Labelled pos vs neg
    df = data.copy()
    df = df[df["label"].str.startswith("labelled")]
    df["label"] = np.where(df["label"].str.startswith("labelled_pos"), "pos", "neg")
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/tsne_lab.png")
    plt.clf()

    if WITH_SYM:
        # Labelled sympos, asympos, symneg, asymneg
        df = data.copy()
        df = df[df["label"].str.startswith("labelled")]   
        df["label"] = df["label"].str.replace("labelled_","")
        plt.figure(figsize=(16,10))
        plt.axis('off')
        sns.scatterplot(
            x="tsne1", y="tsne2",
            hue="label",
            data=df,
            legend="full",
            s=200,
            palette=["Red", "Orange", "Green", "Blue"]
        )
        plt.savefig("./vis/tsne_lab_sym.png")
        plt.clf()

    # Test pos vs neg
    df = data.copy()
    df = df[df["label"].str.startswith("test")]
    df["label"] = np.where(df["label"].str.startswith("test_pos"), "pos", "neg")
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/tsne_test.png")
    plt.clf()

    if WITH_SYM:
        # Test sympos, asympos, symneg, asymneg
        df = data.copy()
        df = df[df["label"].str.startswith("test")]   
        df["label"] = df["label"].str.replace("test_","")
        plt.figure(figsize=(16,10))
        plt.axis('off')
        sns.scatterplot(
            x="tsne1", y="tsne2",
            hue="label",
            data=df,
            legend="full",
            s=200,
            palette=["Red", "Orange", "Green", "Blue"]
        )
        plt.savefig("./vis/tsne_test_sym.png")
        plt.clf()

    # Unlabelled pos vs neg
    df = data.copy()
    df = df[df["label"].str.startswith("unlabelled")]
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/tsne_unlab.png")
    plt.clf()

    # All
    df = data.copy()
    df["label"] = np.where(df["label"].str.contains("pos"), "pos", "neg")
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/tsne_all.png")
    plt.clf()


def get_pca(latents, labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latents)
    print("Explained variance for principle components", pca.explained_variance_ratio_)
    plt.figure(figsize=(16,10))
    data = pd.DataFrame({"pca1": pca_result[:,0], "pca2": pca_result[:,1], "label": labels})
    return data.loc[data.pca1 < 25]

def vis_pca(data):
    # Labelled pos vs neg
    df = data.copy()
    df = df[df["label"].str.startswith("labelled")]
    df["label"] = np.where(df["label"].str.startswith("labelled_pos"), "pos", "neg")
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/pca_lab.png")
    plt.clf()

    if WITH_SYM:
        # Labelled sympos, asympos, symneg, asymneg
        df = data.copy()
        df = df[df["label"].str.startswith("labelled")]   
        df["label"] = df["label"].str.replace("labelled_","")
        plt.figure(figsize=(16,10))
        plt.axis('off')
        sns.scatterplot(
            x="pca1", y="pca2",
            hue="label",
            data=df,
            legend="full",
            s=200,
            palette=["Red", "Orange", "Green", "Blue"]
        )
        plt.savefig("./vis/pca_lab_sym.png")
        plt.clf()

    # Test pos vs neg
    df = data.copy()
    df = df[df["label"].str.startswith("test")]
    df["label"] = np.where(df["label"].str.startswith("test_pos"), "pos", "neg")
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/pca_test.png")
    plt.clf()

    if WITH_SYM:
        # Test sympos, asympos, symneg, asymneg
        df = data.copy()
        df = df[df["label"].str.startswith("test")]   
        df["label"] = df["label"].str.replace("test_","")
        plt.figure(figsize=(16,10))
        plt.axis('off')
        sns.scatterplot(
            x="pca1", y="pca2",
            hue="label",
            data=df,
            legend="full",
            s=200,
            palette=["Red", "Orange", "Green", "Blue"]
        )
        plt.savefig("./vis/pca_test_sym.png")
        plt.clf()

    # Unlabelled pos vs neg
    df = data.copy()
    df = df[df["label"].str.startswith("unlabelled")]
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/pca_unlab.png")
    plt.clf()

    # All
    df = data.copy()
    df["label"] = np.where(df["label"].str.contains("pos"), "pos", "neg")
    plt.figure(figsize=(16,10))
    plt.axis('off')
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="label",
        data=df,
        legend="full",
        s=200,
        palette=["Orange", "Blue"]
    )
    plt.savefig("./vis/pca_all.png")
    plt.clf()
    

def main():
    # create directories and load predictions
    if not os.path.exists("./vis"):
        os.mkdir("./vis")
    with open("./pred.pk", "rb") as f:
        pred_data = joblib.load(f)

    # get PCA and t-SNE vector spaces
    latents, labels = flatten_data(pred_data)
    pca = get_pca(latents, labels)
    tsne = get_tsne(latents, labels)

    # visualise
    vis_pca(pca)
    vis_tsne(tsne)


if __name__ == "__main__":
    main()

