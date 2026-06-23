# MONARCH
# -------
# Motion Observation and Neuromotor Analysis for Rehabilitation
# and Clinical Health
#
# Author: Corentin Jossi
# Date: 20.05.2026
# Description: This script serves as the main entry point 
#              for the gait analysis project.
# ------------------------------------------------------------------------------

# Standard Imports

# Third Party Imports
import pandas as pd

# Local Imports
import monarch.config as config
from monarch.data_analysis import (
    DBSCAN_tuning,
    KMeans_tuning,
    cluster_profiles,
    correlation_matrix,
    DBSCAN_clustering,
    KMeans_clustering,
    hierarchical_clustering,
    PCA_analysis,
    UMAP_analysis,
)
from monarch.data_creation import (
    aggregated_dataset,
    data_cleaning,
    data_extraction,
    calculate_variability_indices,
    dlsm_activity_profiles,
    variation_dataset,
    z_score_normalisation
)

def dataset_analysis(
        dataset_name: str,
) -> None:
        # ===== Step 1: Data Extraction =====
    data: pd.DataFrame = data_extraction(dataset_name)

    cleaned_data: pd.DataFrame = data_cleaning(data)

    if dataset_name != 'dataset_A':
        #aggregated_data: pd.DataFrame = aggregated_dataset(
        #    cleaned_gait_data
        #)

        # If sliding window is enabled, we create a new dataset 
        # with the variation of the gait parameters over time giving us more 
        # data points to work with.
        print("Creating variation dataset...")
        aggregated_data: pd.DataFrame = variation_dataset(
            cleaned_data,
            sliding_window = True
        )
    else:
        aggregated_data: pd.DataFrame = cleaned_data

    dlsm_data: pd.DataFrame = data_cleaning(
        data_extraction('dataset_dlsm')
    )

    print(aggregated_data.head())

    # ===== Step 2: Variability Indices Calculation =====
    #variability_indices: pd.DataFrame = calculate_variability_indices(
    #    aggregated_data, 
    #    dataset_name
    #)

    #full_data: pd.DataFrame = pd.concat(
    #    [aggregated_data, variability_indices], 
    #    axis=1
    #)

    full_data: pd.DataFrame = aggregated_data.dropna()

    # ===== Step 3: Data Normalisation =====
    normalised_data: pd.DataFrame = z_score_normalisation(
        full_data, 
        'global'
    )

    print(normalised_data.head())

    # ===== Step 4: Correlation Matrix =====
    #correlation_matrix(normalised_data)

    # ===== Step 5: PCA Analysis ===== 
    principal_components = PCA_analysis(full_data, normalised_data)

    # ===== Step 6: UMAP Analysis =====
    embedding_umap = UMAP_analysis(full_data, principal_components)

    # ===== Step 7: K-Means Clustering =====
    #KMeans_tuning(principal_components)
    clusters_KMeans = KMeans_clustering(principal_components, embedding_umap)

    # ===== Step 8: DBSCAN Clustering =====
    #DBSCAN_tuning(principal_components)
    clusters_DBSCAN = DBSCAN_clustering(principal_components, embedding_umap)

    # ===== Step 9: Hierarchical Clustering =====
    #hierarchical_clustering(principal_components, full_data)

    # ==== Step 10: DLSM Linkage =====
    activity_profiles = dlsm_activity_profiles(dlsm_data)

    cluster_profiles(
        full_data=full_data, 
        clusters=clusters_KMeans, 
        embedding_umap=embedding_umap, 
        activity_profiles=activity_profiles
    )


def main() -> None:
    dataset_analysis('dataset_B')

    return None

if __name__ == "__main__":
    main()