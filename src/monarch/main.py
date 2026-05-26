# MONARCH
# Motion Observation and Neuromotor Analysis for Rehabilitation and Clinical Health
# Author: Corentin Jossi
# Date: 20.05.2026
# Description: This script serves as the main entry point 
#              for the gait analysis project.
# ------------------------------------------------------------------------------

# Standard Imports

# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Local Imports
from monarch.data_creation import data_extraction


def correlation_matrix(df: pd.DataFrame) -> None:
    """
    Create a correlation matrix to visualize the relationships between 
    different gait parameters
    """

    corr_matrix = df.corr()

    print("Correlation Matrix:")
    print(corr_matrix)

    # Generate mask for lower triangle
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    ax = sns.heatmap(corr_matrix, annot=True, cmap='seismic', mask=mask)
    ax.set(xlabel='', ylabel='')
    ax.xaxis.tick_top()

    plt.title('Correlation Matrix of Gait Parameters')
    plt.show()

    return None

def main() -> None:
    # Step 1: Data Extraction
    gait_data = data_extraction()

    # Step 2: Correlation Matrix
    #correlation_matrix(gait_data)

    return None

if __name__ == "__main__":
    main()