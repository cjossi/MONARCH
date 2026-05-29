# Standard Imports

# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    DBSCAN
)
import umap

# Local Imports

# ------------------------------------------------------------------------------
# Data Analysis
# ------------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame) -> None:
    """
    Create a correlation matrix to visualize the relationships between 
    different gait parameters
    """

    corr_matrix = df.corr()

    # Generate mask for lower triangle
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(18, 14))

    ax = sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='seismic', 
        mask=mask,
        vmin=-1, 
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    ax.xaxis.tick_top()

    plt.xticks(rotation=45, ha='left', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.title(
        "Correlation Matrix of Gait Parameters",
        fontsize=16,
        pad=20
    )

    plt.tight_layout()
    plt.show()

    return None

def PCA_analysis(
        original_data: pd.DataFrame,
        normalised_data: pd.DataFrame
) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) to reduce dimensionality 
    and identify key patterns in the gait data.

    Parameters
    ----------
    original_data : DataFrame
        The input DataFrame containing the original gait parameters.
    normalised_data : DataFrame
        The input DataFrame containing normalised gait parameters
        and variability indices.
    
    Returns
    -------
    np.ndarray
        The transformed data containing the 8 principal components 
        and explained variance.
    """
   
    my_pca = PCA()

    my_pca.fit_transform(normalised_data)

    # Print explained variance ratio for each principal component
    print(my_pca.explained_variance_ratio_)

    cumulative_variance = np.cumsum(my_pca.explained_variance_ratio_)
    print(cumulative_variance)

    loadings = pd.DataFrame(
        my_pca.components_.T,
        columns=[f'PC{i+1}' for i in range(my_pca.n_components_)],
        index=normalised_data.columns
    )

    loadings.to_csv("pca_loadings.csv")


    # ===== Scree Plot =====

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(my_pca.explained_variance_ratio_) + 1), 
        my_pca.explained_variance_ratio_ * 100
    )
    plt.plot(
        range(1, len(my_pca.explained_variance_ratio_) + 1), 
        my_pca.explained_variance_ratio_ * 100, 
        marker='o', 
        color='black'
    )
    plt.plot(
        range(1, len(cumulative_variance) + 1), 
        cumulative_variance * 100, 
        marker='o', 
        color='red'
    )
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.show()

    # ===== Biplot =====

    principal_components_8 = PCA_biplot(
        original_data=original_data,
        normalised_data=normalised_data,
        features_names=normalised_data.columns.tolist(),
        pc1=0,
        pc2=1
    )

    return principal_components_8


def PCA_biplot(
        original_data: pd.DataFrame,
        normalised_data: pd.DataFrame,
        features_names: list[str],
        pc1: int = 0,
        pc2: int = 1
) -> np.ndarray:
    """
    Create a PCA biplot to visualize the relationships between participants
    and gait parameters in the space defined by the first two principal 
    components.

    Parameters
    ----------
    original_data : DataFrame
        The input DataFrame containing the original gait parameters.
    normalised_data : DataFrame
        The input DataFrame containing normalised gait parameters
        and variability indices.
    features_names : list[str]
        List of feature names corresponding to the columns in normalised_data.
    pc1 : int, optional
        The index of the first principal component to plot (default is 0).
    pc2 : int, optional
        The index of the second principal component to plot (default is 1).

    Returns
    -------
    np.ndarray
        The transformed data containing the first two principal components.
    """
    
    # ===== PCA =====
    pca_8: PCA = PCA(n_components=8)

    principal_components_8: np.ndarray = pca_8.fit_transform(normalised_data)

    # ===== Participant IDs =====
    participant_ids: pd.Series = original_data['snr_id']
    
    plt.figure(figsize=(12, 10))

    # ===== Scatter Plot =====
    for participant_id in participant_ids.unique():
        mask = participant_ids == participant_id

        plt.scatter(
            principal_components_8[mask, pc1],
            principal_components_8[mask, pc2],
            label=f'Participant {participant_id}',
            alpha=0.7
        )

    # ===== PCA Loadings =====
    loadings: np.ndarray = pca_8.components_.T

    loading_score: np.ndarray = (
        np.abs(loadings[:, pc1]) + np.abs(loadings[:, pc2])
    )

    top_features_indices: np.ndarray = np.argsort(loading_score)[-20:]

    # ===== Feature Arrows =====
    for feature_index in top_features_indices:
        x: float = loadings[feature_index, pc1]
        y: float = loadings[feature_index, pc2]

        plt.arrow(
            0, 0,
            x * 5, y * 5,
            color='red',
            alpha=0.5,
            head_width=0.02
        )

        plt.text(
            x * 5.2, y * 5.2,
            features_names[feature_index],
            color='red',
            fontsize=5
        )
    
    # ===== Plot Customization =====
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.xlabel(f'Principal Component {pc1 + 1}')
    plt.ylabel(f'Principal Component {pc2 + 1}')

    expl_var: float = pca_8.explained_variance_ratio_[pc1] * 100
    expl_var_2: float = pca_8.explained_variance_ratio_[pc2] * 100

    plt.title(f'PCA Biplot (PC{pc1 + 1} vs PC{pc2 + 1})\n'
              f'Explained variance: {expl_var:.2f}% / {expl_var_2:.2f}%')
    plt.grid()

    plt.show()

    return principal_components_8

def UMAP_analysis(
        original_data: pd.DataFrame,
        principal_components_8: np.ndarray
) -> np.ndarray:
    """
    Perform Uniform Manifold Approximation and Projection (UMAP) to 
    visualize the high-dimensional gait data in a lower-dimensional space.

    Parameters
    ----------
    original_data : DataFrame
        The input DataFrame containing the original gait parameters.
    principal_components_8 : np.ndarray
        The input array containing the first eight principal components.

    Returns
    -------
    np.ndarray
        The transformed data containing the UMAP embedding.
    """

    reducer = umap.UMAP(
        n_neighbors=5,      # Local for small dataset
        min_dist=0.3,       # Allow some clustering but not too tight
        n_components=2,
        random_state=42
    )

    embedding: np.ndarray = reducer.fit_transform(
        principal_components_8
    ) # type: ignore

    plt.figure(figsize=(12, 10))

    participant_ids: pd.Series = original_data['snr_id']

    for participant_id in participant_ids.unique():
        mask = participant_ids == participant_id

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=f'Participant {participant_id}',
            alpha=0.8
        )
    
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP Projection of Gait Data', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()

    return embedding

def KMeans_clustering(
        principal_components_8: np.ndarray,
        embedding_umap: np.ndarray
) -> None:
    """
    Perform K-Means clustering to identify distinct groups of participants 
    based on their gait parameters.
    """

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    clusters = kmeans.fit_predict(principal_components_8)

    plt.figure(figsize=(12, 10))

    plt.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=clusters,
        cmap='viridis',
        alpha=0.8
    )

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('K-Means Clustering of Gait Data', fontsize=16)

    plt.grid()
    plt.show()

def DBSCAN_clustering(
        principal_components_8: np.ndarray,
        embedding_umap: np.ndarray
) -> None:
    """
    Perform DBSCAN clustering to identify distinct groups of participants 
    based on their gait parameters.
    """

    dbscan = DBSCAN(
        eps=0.1,
        min_samples=2
    )

    cluster_dbscan = dbscan.fit_predict(principal_components_8)

    plt.figure(figsize=(12, 10))

    plt.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=cluster_dbscan,
        cmap='viridis',
        alpha=0.8
    )

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('DBSCAN Clustering of Gait Data', fontsize=16)

    plt.grid()
    plt.show()