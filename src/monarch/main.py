# MONARCH
# Motion Observation and Neuromotor Analysis for Rehabilitation and Clinical Health
# Author: Corentin Jossi
# Date: 20.05.2026
# Description: This script serves as the main entry point 
#              for the gait analysis project.
# ------------------------------------------------------------------------------

# Standard Imports

# Third Party Imports
import pandas as pd

# Local Imports
from monarch.data_analysis import correlation_matrix
from monarch.data_creation import (
    data_extraction,
    calculate_variability_indices,
    z_score_normalisation
)
from monarch.data_analysis import PCA_analysis


def main() -> None:
    # Step 1: Data Extraction
    gait_data = data_extraction()

    # Step 2: Variability Indices Calculation
    variability_indices = calculate_variability_indices(gait_data)

    full_data = pd.concat([gait_data, variability_indices], axis=1)

    # Step 3: Data Normalisation
    normalised_data = z_score_normalisation(full_data, 'global')

    # Step 4: Correlation Matrix
    correlation_matrix(normalised_data)
    print("Correlation matrix generated successfully.")

    # Step 5: PCA Analysis
    PCA_analysis(full_data, normalised_data)
    print("PCA analysis completed successfully.")
    

    return None

if __name__ == "__main__":
    main()