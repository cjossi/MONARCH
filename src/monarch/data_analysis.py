# Standard Imports

# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

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
) -> None:
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


    # Scree Plot
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

    # Biplot

    principal_components_8: np.ndarray = PCA(
        n_components=8
        ).fit_transform(normalised_data)
    
    pca_8: PCA = PCA(n_components=8).fit(normalised_data)

    participant_ids: pd.Series = original_data['snr_id']
    participant_ids_unique: np.ndarray = participant_ids.unique()

    PCA_biplot(
        participant_ids,
        participant_ids_unique,
        principal_components_8,
        pca_8,
        features_names=normalised_data.columns.tolist(),
        pc1=0,
        pc2=1
    )


def PCA_biplot(
        participant_ids: pd.Series,
        participant_ids_unique: np.ndarray,
        principal_components_8: np.ndarray,
        pca_8: PCA,
        features_names: list[str],
        pc1: int = 0,
        pc2: int = 1
) -> None:
    """
    Create a PCA biplot to visualize the relationships between 
    the original features and the principal components.

    Parameters
    ----------
    participant_ids : pd.Series
        A Series containing the participant identifiers corresponding to each 
        row in the normalised_data DataFrame.

    participant_ids_unique : np.ndarray
        An array of unique participant identifiers.

    normalised_data : DataFrame
        The input DataFrame containing normalised gait parameters
        and variability indices.

    pca : PCA
        The fitted PCA object containing the principal components 
        and explained variance.

    features_names : list[str]
        List of feature names corresponding to the columns in the 
        normalised_data DataFrame.

    pc1 : int, optional
        The index of the first principal component to plot (default is 0).

    pc2 : int, optional
        The index of the second principal component to plot (default is 1).
    """
    plt.figure(figsize=(12, 10))

    for participant_id in participant_ids_unique:
        mask = participant_ids == participant_id

        plt.scatter(
            principal_components_8[mask, pc1],
            principal_components_8[mask, pc2],
            label=f'Participant {participant_id}',
            alpha=0.7
        )

    loadings = pca_8.components_.T # Shape: (n_features, n_components)

    loading_score = (
        np.abs(loadings[:, pc1]) + np.abs(loadings[:, pc2])
    )

    top_features_indices = np.argsort(loading_score)[-20:]

    for feature_index in top_features_indices:
        x = loadings[feature_index, pc1]
        y = loadings[feature_index, pc2]

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
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.xlabel(f'Principal Component {pc1 + 1}')
    plt.ylabel(f'Principal Component {pc2 + 1}')

    expl_var = pca_8.explained_variance_ratio_[pc1] * 100
    expl_var_2 = pca_8.explained_variance_ratio_[pc2] * 100

    plt.title(f'PCA Biplot (PC{pc1 + 1} vs PC{pc2 + 1})\n'
              f'Explained varaince: {expl_var:.2f}% / {expl_var_2:.2f}%')
    plt.grid()

    plt.show()
