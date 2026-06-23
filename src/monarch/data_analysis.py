# Standard Imports

# Third Party Imports
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)
from scipy.spatial import ConvexHull
from sklearn.cluster import (
    KMeans,
    DBSCAN
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import umap

# Local Imports

# ------------------------------------------------------------------------------
# Data Analysis
# ------------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame) -> None:
    """
    Create Pearson and Spearman correlation matrices side by side.
    """

    pearson_corr_matrix = df.corr(method='pearson')
    spearman_corr_matrix = df.corr(method='spearman')

    # Generate mask for lower triangle
    mask = np.tril(np.ones_like(pearson_corr_matrix, dtype=bool))

    corr = spearman_corr_matrix.where(mask)

    mean_abs_corr = corr.abs().stack().mean()

    print(f'Mean absolute Spearman correlation: {mean_abs_corr:.4f}')

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 8)
    )

    sns.heatmap(
        pearson_corr_matrix, 
        ax=axes[0],
        annot=True,
        cmap='seismic', 
        mask=mask,
        vmin=-1, 
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    axes[0].set_title('Pearson Correlation Matrix')
    axes[0].xaxis.tick_top()
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)

    sns.heatmap(
        spearman_corr_matrix, 
        ax=axes[1],
        annot=True, 
        cmap='seismic',
        mask=mask,
        vmin=-1, 
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    axes[1].set_title('Spearman Correlation Matrix')
    axes[1].xaxis.tick_top()
    axes[1].tick_params(axis='x', rotation=90, labelsize=8)
    axes[1].tick_params(axis='y', rotation=0, labelsize=8)

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
    Scree_plot(
        my_pca=my_pca,
        cumulative_variance=cumulative_variance
    )

    # ===== Biplot =====
    principal_components = PCA_biplot(
        original_data=original_data,
        normalised_data=normalised_data,
        features_names=normalised_data.columns.tolist(),
        components=4,
        pc1=0,
        pc2=1
    )

    return principal_components

def Scree_plot(
        my_pca: PCA,
        cumulative_variance: np.ndarray
) -> None:
    """
    Create a scree plot to visualize the explained variance ratio of each 
    principal component.
    """

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
    
    return None

def PCA_biplot(
        original_data: pd.DataFrame,
        normalised_data: pd.DataFrame,
        features_names: list[str],
        components: int = 8,
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
    components : int, optional
        The number of principal components to consider (default is 8).
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
    rank = np.linalg.matrix_rank(normalised_data)
    n_components: int = min(
        components,
        rank
    )
    pca_8: PCA = PCA(n_components=n_components)

    principal_components: np.ndarray = pca_8.fit_transform(normalised_data)

    # ===== Participant IDs =====
    participant_ids: pd.Series = original_data['snr_id']
    
    plt.figure(figsize=(12, 10))

    # ===== Scatter Plot =====
    for participant_id in participant_ids.unique():
        mask = participant_ids == participant_id

        plt.scatter(
            principal_components[mask, pc1],
            principal_components[mask, pc2],
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

    return principal_components

def UMAP_analysis(
        original_data: pd.DataFrame,
        principal_components: np.ndarray
) -> np.ndarray:
    """
    Perform Uniform Manifold Approximation and Projection (UMAP) to 
    visualize the high-dimensional gait data in a lower-dimensional space.

    Parameters
    ----------
    original_data : DataFrame
        The input DataFrame containing the original gait parameters.
    principal_components : np.ndarray
        The input array containing the first eight principal components.

    Returns
    -------
    np.ndarray
        The transformed data containing the UMAP embedding.
    """

    # tuning UMAP parameters for better separation of participants 
    # in the embedding space
    reducer = umap.UMAP(
        n_neighbors=15,     # Local for small dataset
        min_dist=0.3,       # Allow some clustering but not too tight
        n_components=2,
        random_state=42
    )

    embedding: np.ndarray = reducer.fit_transform(
        principal_components
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
        principal_components: np.ndarray,
        embedding_umap: np.ndarray
) -> np.ndarray:
    """
    Perform K-Means clustering to identify distinct groups of participants 
    based on their gait parameters.

    Parameters
    ----------
    principal_components : np.ndarray
        The input array containing the first principal components.
    embedding_umap : np.ndarray
        The UMAP embedding of the data.
    """

    kmeans = KMeans(
        n_clusters=4,   # Tuned using silhouette and inertia analysis = 8
        random_state=42
    )

    clusters = kmeans.fit_predict(principal_components)

    plt.figure(figsize=(12, 10))

    plt.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=clusters,
        cmap='viridis',
        alpha=0.8
    )

    for cluster_id in np.unique(clusters):
        cluster_points = embedding_umap[clusters == cluster_id]

        # =====Centroid plotting=====
        centroid = cluster_points.mean(axis=0)

        plt.scatter(
            centroid[0],
            centroid[1],
            marker='X',
            s=200,
            c='red',
            edgecolor='black',
            label=f'Centroid {cluster_id}'
        )

        # ====Convex Hull plotting=====
        if len(cluster_points) < 3:
            continue

        hull = ConvexHull(cluster_points)

        for simplex in hull.simplices:
            plt.plot(
                cluster_points[simplex, 0],
                cluster_points[simplex, 1],
                'k-',
                alpha=0.5,
                linewidth=1
            )

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('K-Means Clustering of Gait Data', fontsize=16)

    plt.grid()
    plt.show()

    return clusters

def KMeans_tuning(
        principal_components: np.ndarray
):
    """
    Tune KMeans parameters for better clustering of participants.
    Silhouette & Inertia analysis can be used to determine the optimal number 
    of clusters. Use of the scikit-learn example code.
    https://scikit-learn.org/stable/auto_examples/cluster/
    plot_kmeans_silhouette_analysis.html

    Parameters
    ----------
    principal_components : np.ndarray
        The input array containing the first principal components.
    """

    # =====Silhouette analysis=====
    range_clusters = range(2, 50)

    silhouette_avgs = []

    for n_clusters in range_clusters:
        clusters = KMeans(
            n_clusters=n_clusters, 
            random_state=42
        )

        cluster_labels = clusters.fit_predict(principal_components)

        silhouette_avg = silhouette_score(
            principal_components,
            cluster_labels
        )

        silhouette_avgs.append(silhouette_avg)

        print(
            "For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg
        )
    
    plt.figure(figsize=(10, 6))
    plt.plot(
        range_clusters,
        silhouette_avgs,
        marker='o'
    )
    plt.title('Silhouette Analysis for KMeans Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.grid()
    plt.show()


    # =====Inertia analysis=====
    inertia_values = []

    for n_clusters in range_clusters:
        clusters = KMeans(
            n_clusters=n_clusters, 
            random_state=42
        )

        clusters.fit(principal_components)

        inertia_values.append(clusters.inertia_)

        print(
            "For n_clusters =", n_clusters,
            "The inertia is :", clusters.inertia_
        )

    plt.figure(figsize=(10, 6))
    plt.plot(
        range_clusters,
        inertia_values,
        marker='o'
    )
    plt.title('Inertia Analysis for KMeans Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()

    return None

def DBSCAN_clustering(
        principal_components: np.ndarray,
        embedding_umap: np.ndarray
) -> np.ndarray:
    """
    Perform DBSCAN clustering to identify distinct groups of participants 
    based on their gait parameters.
    """

    dbscan = DBSCAN(
        eps=0.09,       # Tuned using k-distance graph
        min_samples=5   # k = nb_dimension + 1 = 4 + 1
    )

    cluster_dbscan = dbscan.fit_predict(principal_components)

    plt.figure(figsize=(12, 10))

    plt.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=cluster_dbscan,
        cmap='viridis',
        alpha=0.8
    )

    for cluster_id in np.unique(cluster_dbscan):
        if cluster_id == -1:
            continue

        cluster_points = embedding_umap[cluster_dbscan == cluster_id]

        if len(cluster_points) < 3:
            continue

        hull = ConvexHull(cluster_points)

        for simplex in hull.simplices:
            plt.plot(
                cluster_points[simplex, 0],
                cluster_points[simplex, 1],
                'k-',
                alpha=0.5,
                linewidth=1
            )

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('DBSCAN Clustering of Gait Data', fontsize=16)

    plt.grid()
    plt.show()

    return cluster_dbscan

def DBSCAN_tuning(
        principal_components: np.ndarray
) -> None:
    """
    Tune DBSCAN parameters for better clustering of participants.
    Use of k-distance graph to determine the optimal eps value.
    """

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(principal_components)

    distances, indices = neighbors_fit.kneighbors(principal_components)

    distances = np.sort(distances[:, 4])

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance to 5th Nearest Neighbor')
    plt.title('K-Distance Graph for DBSCAN Tuning')
    plt.grid()
    plt.show()

    return None

def hierarchical_clustering(
        principal_components: np.ndarray,
        original_data: pd.DataFrame
) -> None:
    """
    Perform hierarchical clustering to identify distinct groups of participants 
    based on their gait parameters.
    """

    linked = linkage(
        principal_components, 
        method='ward'
    )

    plt.figure(figsize=(12, 10))

    labels = [
        f'{participant}_{timeline}'
        for participant, timeline in zip(
            original_data['snr_id'], 
            original_data['timeline_stage']
        )
    ]

    dendrogram(
        linked,
        labels=labels
    )

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    plt.xticks(rotation=45, ha='right', fontsize=8)

    plt.show()

def cluster_profiles(
        full_data: pd.DataFrame,
        clusters: np.ndarray,
        embedding_umap: np.ndarray,
        activity_profiles: pd.DataFrame
) -> None:
    """
    Set the activity levels profiles for each cluster based on the DLSM dataset.
    Example:
        Cluster 1:
            Sedentary: 20%
            Low: 30%
            Moderate: 30%
            Vigorous: 20%
    """

    full_data['cluster'] = clusters

    # Get the participats & timeline stages profile for each cluster.
    merged_data = full_data.merge(
        activity_profiles,
        on=['snr_id', 'timeline_stage'],
        how='left'
    )

    print(merged_data.head())

    aggregated_row = []

    for cluster_id in np.unique(clusters):
        sedentary_total: float = merged_data.loc[
            merged_data['cluster'] == cluster_id,
            'sedentary_profile'
        ].sum()

        low_total: float = merged_data.loc[
            merged_data['cluster'] == cluster_id,
            'low_profile'
        ].sum()

        moderate_total: float = merged_data.loc[
            merged_data['cluster'] == cluster_id,
            'moderate_profile'
        ].sum()

        vigorous_total: float = merged_data.loc[
            merged_data['cluster'] == cluster_id,
            'vigorous_profile'
        ].sum()

        total: float = (sedentary_total 
                        + low_total 
                        + moderate_total 
                        + vigorous_total)

        participant_count = merged_data.loc[
            merged_data['cluster'] == cluster_id, 
            'snr_id'
        ].nunique()

        print(merged_data.groupby('cluster')['snr_id'].nunique())

        windows_count = merged_data.groupby(
            ['snr_id', 'timeline_stage']
        )['cluster'].value_counts()
        print(windows_count)

        aggregated_row.append({
            'cluster': cluster_id,
            'sedentary_total': sedentary_total / total if total != 0 else 0,
            'low_total': low_total / total if total != 0 else 0,
            'moderate_total': moderate_total / total if total != 0 else 0,
            'vigorous_total': vigorous_total / total if total != 0 else 0,
            'participant_count': participant_count
        })
    
    aggregated_data = pd.DataFrame(aggregated_row)

    print(aggregated_data)

