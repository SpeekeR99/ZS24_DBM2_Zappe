import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys

circle_viable = (sys.version_info[0] == 3 and sys.version_info[1] >= 8)  # PyCirclize funguje pouze pro Python 3.8 a vyssi
if circle_viable:
    from pycirclize import Circos


def my_pca(df):
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
    fig = plt.figure(figsize=(10, 6))
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
    bar_patch = mpatches.Patch(color="green", label="Explained Variance")
    line_patch = mlines.Line2D([], [], color="red", label="Cumulative Sum")
    ax.legend(handles=[bar_patch, line_patch], loc="best")

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
    ax.legend(handles=patches, loc="best")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Loadings")

    plt.grid()
    plt.savefig("img/pca_loadings.svg")
    plt.show()


def correlation(df):
    print("Correlation analysis...")

    # Correlation matrix
    corr = df.corr()

    # Plot correlation matrix
    plot_matrix(corr)
    if circle_viable:
        plot_circle(corr)


def plot_matrix(matrix):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Draw matrix
    mat = ax.matshow(matrix, cmap="Greens")
    plt.colorbar(mat)

    # Draw labels
    labels = matrix.index
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", fontsize=10)

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=33.75)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    ax.set_title("Correlation matrix")

    # Adjust subplot parameters to shift the figure to the right
    plt.subplots_adjust(left=0.15)

    plt.savefig("img/corr_matrix.svg")
    plt.show()


def plot_circle(matrix):
    # Preparation of sectors for circle graph
    labels = matrix.index
    sectors = {}
    for label in labels:  # Each label has its own sector
        sectors[label] = 1  # Random size of sector (important that all sectors are the same size)

    # Color palette
    cmap = plt.cm.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Create circle graph
    circos = Circos(sectors, space=3)
    for j, sector in enumerate(circos.sectors):
        track = sector.add_track((93, 100))
        track.axis(fc=colors[j])
        track.text(sector.name, color="white", size=12)

    # The circle graph wants the values to be from 0 to 1, but correlation can be negative
    matrix = matrix.abs()
    # Normalize
    matrix = matrix / matrix.max().max()

    for j, row in enumerate(matrix.values):
        for k, val in enumerate(row):
            # Diagonal is not interesting (always 1)
            if j == k:
                continue

            row_name = labels[j]
            row_size = sectors[row_name]
            row_sector_part = row_size / len(labels)  # N-th part of sector

            col_name = labels[k]
            col_size = sectors[col_name]
            col_sector_part = col_size / len(labels)  # N-th part of sector

            row_width = row_sector_part * val
            row_start = row_sector_part * k + (row_sector_part - row_width) / 2
            row_end = row_start + row_width

            col_width = col_sector_part * val
            col_start = col_sector_part * k + (col_sector_part - col_width) / 2
            col_end = col_start + col_width

            # Add connection between two sectors
            circos.link(
                (row_name, row_start, row_end),
                (col_name, col_end, col_start),  # Start and End are reversed so the connections are visible
                color=colors[j],
                direction=1
            )

    circos.text("Correlation circle graph", color="black", deg=0, r=120, size=12)
    circos.plotfig(ax=ax)
    plt.savefig("img/corr_circle.svg")
    plt.show()
