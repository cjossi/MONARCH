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
    create_cov_long_dataframe,
    DBSCAN_clustering,
    KMeans_clustering,
    get_spatio_temporal_distribution,
    hierarchical_clustering,
    PCA_analysis,
    UMAP_analysis,
    paired_scatter_pre_post,
    shapiro_test,
    violin_boxplot_CoV,
    wilcoxon_pre_post
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
    healthy_data: pd.DataFrame = data_extraction('healthy_dataset')

    cleaned_data: pd.DataFrame = data_cleaning(data)
    healthy_cleaned_data: pd.DataFrame = data_cleaning(healthy_data)

    if dataset_name != 'dataset_A':
        # If sliding window is enabled, we create a new dataset 
        # with the variation of the gait parameters over time giving us more 
        # data points to work with.
        cov_window_data: pd.DataFrame = variation_dataset(
            cleaned_data,
            sliding_window = True
        )

        cov_session_data: pd.DataFrame = variation_dataset(
            cleaned_data,
            sliding_window = False
        )
    else:
        cov_session_data: pd.DataFrame = cleaned_data
        cov_window_data: pd.DataFrame = cleaned_data

    dlsm_data: pd.DataFrame = data_cleaning(
        data_extraction('dataset_dlsm')
    )

    # ===== Understanding the data =====
    get_spatio_temporal_distribution(cleaned_data)

    long_CoV = create_cov_long_dataframe(cleaned_data)
    healthy_long_CoV = create_cov_long_dataframe(healthy_cleaned_data)

    violin_boxplot_CoV(long_CoV, healthy_long_CoV)

    #shapiro_results = shapiro_test(long_CoV)
    # We will take Wilcoxon test for the paired comparaison to be robust.

    #wilcoxon_results = wilcoxon_pre_post(long_CoV)

    #paired_scatter_pre_post(long_CoV)

    #full_session_data: pd.DataFrame = cov_window_data.dropna()

    #print(
    #    f"Samples for lustering: {len(full_session_data)}"
    #)

    # Data Normalisation
    #normalised_session_data: pd.DataFrame = z_score_normalisation(
    #    full_session_data, 
    #    'global'
    #)

    # Correlation matrices
    #correlation_matrix(
    #    normalised_session_data,
    #    title="Spearman Correlation Matrix - All sessions"
    #)

    #correlation_matrix(
    #    normalised_session_data[
    #        full_session_data['timeline_stage'] == 'admission'
    #    ],
    #    title="Spearman Correlation Matrix - Pre sessions"
    #)

    #correlation_matrix(
    #    normalised_session_data[
    #        full_session_data['timeline_stage'] == 'discharge'
    #    ],
    #    title="Spearman Correlation Matrix - Post sessions"
    #)

    # ===== Analysis PCA/UMAP/Clustering =====
    # Sliding window representation is used for unseupervised learning methods
    # to capture local vvariability patterns and increase the numper of samples.
    #full_window_data: pd.DataFrame = cov_window_data.dropna()

    #print(
    #    f"Samples for lustering: {len(full_window_data)}"
    #)

    # Data Normalisation
    #normalised_data: pd.DataFrame = z_score_normalisation(
    #    full_window_data, 
    #    'global'
    #)

    # ===== Step 5: PCA Analysis ===== 
    #principal_components = PCA_analysis(full_window_data, normalised_data)

    # ===== Step 6: UMAP Analysis =====
    #embedding_umap = UMAP_analysis(full_window_data, principal_components)

    # ===== Step 7: K-Means Clustering =====
    #KMeans_tuning(principal_components)
    #clusters_KMeans = KMeans_clustering(principal_components, embedding_umap)

    # ===== Step 8: DBSCAN Clustering =====
    #DBSCAN_tuning(principal_components)
    #clusters_DBSCAN = DBSCAN_clustering(principal_components, embedding_umap)

    # ===== Step 9: Hierarchical Clustering =====
    #hierarchical_clustering(principal_components, full_window_data)

    # ==== Step 10: DLSM Linkage =====
    #activity_profiles = dlsm_activity_profiles(dlsm_data)

    #cluster_profiles(
    #    full_data=full_window_data, 
    #    clusters=clusters_KMeans, 
    #    embedding_umap=embedding_umap, 
    #    activity_profiles=activity_profiles
    #)


def main() -> None:
    dataset_analysis('dataset_B')

    return None

if __name__ == "__main__":
    main()