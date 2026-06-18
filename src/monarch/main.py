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
    variation_dataset,
    z_score_normalisation
)

def dataset_analysis(
        dataset_name: str
) -> None:
        # ===== Step 1: Data Extraction =====
    gait_data: pd.DataFrame = data_extraction(dataset_name)

    cleaned_gait_data: pd.DataFrame = data_cleaning(gait_data)

    if dataset_name != 'dataset_A':
        #aggregated_data: pd.DataFrame = aggregated_dataset(
        #    cleaned_gait_data
        #)

        aggregated_data: pd.DataFrame = variation_dataset(
            cleaned_gait_data
        )
    else:
        aggregated_data: pd.DataFrame = cleaned_gait_data

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

    full_data: pd.DataFrame = aggregated_data

    # ===== Step 3: Data Normalisation =====
    normalised_data: pd.DataFrame = z_score_normalisation(
        full_data, 
        'global'
    )

    print(normalised_data.head())

    # ===== Step 4: Correlation Matrix =====
    correlation_matrix(normalised_data)

    # ===== Step 5: PCA Analysis =====
    principal_components_8 = PCA_analysis(full_data, normalised_data)

    # ===== Step 6: UMAP Analysis =====
    embedding_umap = UMAP_analysis(full_data, principal_components_8)

    # ===== Step 7: K-Means Clustering =====
    KMeans_clustering(principal_components_8, embedding_umap)

    # ===== Step 8: DBSCAN Clustering =====
    DBSCAN_clustering(principal_components_8, embedding_umap)

    # ===== Step 9: Hierarchical Clustering =====
    hierarchical_clustering(principal_components_8, full_data)


def main() -> None:
    dataset_analysis('dataset_B')

    return None

if __name__ == "__main__":
    main()