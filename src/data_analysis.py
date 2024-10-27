import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def my_pca(df, plot=False):
    print("Principal Component Analysis...")

    # Transform data to normalized normal distribution (standardization)
    sc = StandardScaler()
    data = sc.fit_transform(df)

    # PCA
    pca = PCA(n_components=data.shape[1])
    y = pca.fit_transform(data)

    # Eigenvalues, i.e. variances
    eigenvalues = pca.explained_variance_
    print(f"Eigenvalues (explained variance): {eigenvalues}")

    # Plot PCA
    if plot:
        plot_pareto(pca)
        plot_loadings(pca, df)

    # I want to preserve 90 % of the information
    preserve_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9) + 1
    print(f"Number of components to preserve 90 % of the information: {preserve_90}")

    return y[:, :preserve_90], pca


def plot_pareto(pca):
    """
    Pareto graph
    :param pca: PCA object from sklearn
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot explained variance ratio
    ax.bar(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color="green")

    ax.set_xticks(np.arange(len(pca.explained_variance_ratio_)))
    ax.set_xticklabels(np.arange(1, len(pca.explained_variance_ratio_) + 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Proportion of variance explained")
    ax.set_title("Pareto graph")

    # Plot cumulative sum
    ax.plot(np.cumsum(pca.explained_variance_ratio_), color="red")

    # Create legend
    bar_patch = mpatches.Patch(color='green', label='Explained Variance')
    line_patch = mlines.Line2D([], [], color='red', label='Cumulative Sum')
    ax.legend(handles=[bar_patch, line_patch], loc='best')

    plt.grid()
    plt.savefig("img/pca_pareto.svg")
    plt.show()


def plot_loadings(pca, df):
    """
    Biplot of Principal Component 1 and Principal Component 2 (scatter) with loadings (arrows)
    :param pca: PCA object from sklearn
    :param df: Data (original)
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Loadings
    loadings = pca.components_
    features = df.columns
    colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))

    for i, (feature, color) in enumerate(zip(features, colors)):
        # Plot arrows with different colors
        ax.arrow(0, 0, loadings[0, i], loadings[1, i], head_width=0.02, head_length=0.05, fc=color, ec=color)
        ax.text(loadings[0, i], loadings[1, i], feature, color="k", ha="center", va="center")

    # Create legend
    patches = [mpatches.Patch(color=color, label=feature) for feature, color in zip(features, colors)]
    ax.legend(handles=patches, loc='best')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Loadings")

    plt.grid()
    plt.savefig("img/pca_loadings.svg")
    plt.show()
