# MONARCH
# Author: Corentin Jossi
# Date: 20.05.2026
# Description: This script is responsible to merge all data extraction.
# ------------------------------------------------------------------------------

# Third Party Imports
from numpy import log
from pathlib import Path
import pandas as pd

# Local Imports
import monarch.config as config
from monarch.constant import EPSILON


def data_extraction() -> pd.DataFrame:
    """
    Read the data from a csv file and return it as a Pandas DataFrame.
    The csv file is already preprocessed and contains the gait parameters 
    extracted from the raw data. This is done by WideLog.

    Returns
    -------
    DataFrame
        A Pandas DataFrame containing the extracted gait parameters
    """

    cfg = config.Config.load_from_yaml()

    gait_parameters_folder: Path = cfg.gait_parameters_folder

    # Read csv file:
    gait_parameters = pd.read_csv(
        gait_parameters_folder / 'gait_parameters.csv'
    )

    return gait_parameters

# ------------------------------------------------------------------------------
# Variability Indices Calculation
# ------------------------------------------------------------------------------

def calculate_asymmetry_variability_indice(
        left: pd.Series,
        right: pd.Series
) -> pd.Series:
    """
    Calculate the asymmetry index for a given pair of left and right 
    gait parameters.

    Parameters
    ----------
    left : Series
        Left gait parameter (e.g., stride length, velocity, CoV)
    right : Series
        Right gait parameter (e.g., stride length, velocity, CoV)

    Returns
    -------
    Series
        The calculated asymmetry index
    """

    asymmetry_index: pd.Series = (left - right) / (left + right + EPSILON)

    return asymmetry_index

def calculate_spatio_temporal_variability_indice(
        CoV_stride_length: pd.Series, 
        CoV_velocity: pd.Series
) -> pd.Series:
    """
    Calculate the spatio-temporal variability indice based on the coefficient of 
    variation (CoV) for stride length and stride time.

    Parameters
    ----------
    CoV_stride_length : Series
        Coefficient of variation for stride length
    CoV_velocity : Series
        Coefficient of variation for velocity

    Returns
    -------
    Series
        The calculated spatio-temporal variability indice
    """

    spatio_temporal_indice: pd.Series = pd.Series(
        log(CoV_stride_length / CoV_velocity + EPSILON)
    )

    return spatio_temporal_indice

def calculate_global_variability_indice(
        Cov_stride_length_left: pd.Series,
        Cov_stride_length_right: pd.Series, 
        CoV_velocity_left: pd.Series,
        CoV_velocity_right: pd.Series
) -> pd.Series:
    """
    Calculate the global variability indice based on the coefficient
    of variation (CoV) for stride length and velocity.

    Parameters
    ----------
    Cov_stride_length_left : Series
        Coefficient of variation for stride length (left side)
    Cov_stride_length_right : Series
        Coefficient of variation for stride length (right side)
    CoV_velocity_left : Series
        Coefficient of variation for velocity (left side)
    CoV_velocity_right : Series
        Coefficient of variation for velocity (right side)

    Returns
    -------
    Series
        The calculated global variability indice
    """

    global_variability_indice: pd.Series = (
        Cov_stride_length_left +
        Cov_stride_length_right + 
        CoV_velocity_left + 
        CoV_velocity_right
    ) / 4

    return global_variability_indice

def calculate_variability_indices(
        extracted_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate all variability indices.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the values of the calculated variability indices
    """

    CoV_stride_length_left: pd.Series = extracted_data[
        'CoV_stride_length_left (%)'
    ]
    CoV_stride_length_right: pd.Series = extracted_data[
        'CoV_stride_length_right (%)'
    ]
    CoV_velocity_left: pd.Series = extracted_data[
        'CoV_velocity_left (%)'
    ]
    CoV_velocity_right: pd.Series = extracted_data[
        'CoV_velocity_right (%)'
    ]
    single_support_time_left: pd.Series = extracted_data[
        'mean_single_support_time_left (s)'
    ]
    single_support_time_right: pd.Series = extracted_data[
        'mean_single_support_time_right (s)'
    ]

    CoV_velocity_asymmetry_index = calculate_asymmetry_variability_indice(
        CoV_velocity_left, CoV_velocity_right
    )
    CoV_stride_length_asymmetry_index = calculate_asymmetry_variability_indice(
        CoV_stride_length_left, CoV_stride_length_right
    )
    spatio_temporal_indice = calculate_spatio_temporal_variability_indice(
        CoV_stride_length_left, CoV_velocity_left
    )
    global_variability_indice = calculate_global_variability_indice(
        CoV_stride_length_left, CoV_stride_length_right, 
        CoV_velocity_left, CoV_velocity_right
    )
    support_asymmetry_indice = calculate_asymmetry_variability_indice(
        single_support_time_left, single_support_time_right
    )

    return pd.DataFrame({
        'CoV_stride_length_asymmetry_index': CoV_stride_length_asymmetry_index,
        'CoV_velocity_asymmetry_index': CoV_velocity_asymmetry_index,
        'spatio_temporal_indice': spatio_temporal_indice,
        'global_variability_indice': global_variability_indice,
        'support_asymmetry_indice': support_asymmetry_indice
    })

# ------------------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------------------

def z_score_normalisation(
        extracted_data: pd.DataFrame,
        normalisation_type: str
) -> pd.DataFrame:
    """
    Compute the z-score normalization for all variability indices.
    
    Parameters
    ----------
    extracted_data : pd.DataFrame
        The DataFrame containing the extracted gait parameters

    normalisation_type : str
        The type of normalization to be applied 
        (e.g., 'global', 'per participant')

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the normalized values of the variability indices
    """

    # =====
    # For the test data, no need to drop any column.
    # =====
    #data: pd.DataFrame = extracted_data.drop(columns=[
    #    'timeline_stage',
    #    'test_type',
    #    'snr_id'
    #])

    #normalised_data: pd.DataFrame = data.copy()

    normalised_data: pd.DataFrame = extracted_data.copy()

    data: pd.DataFrame = extracted_data.copy()

    if normalisation_type == 'global':
        for column in data.columns:
            mean: float = data[column].mean()
            std: float = data[column].std()

            normalised_data[column] = (
                (data[column] - mean) / (std + EPSILON)
            )

        return normalised_data

    else:
        raise ValueError(f"Invalid normalisation type: {normalisation_type}")

if __name__ == "__main__":
    extracted_data: pd.DataFrame = data_extraction()

    engineered_features: pd.DataFrame = calculate_variability_indices(
        extracted_data
    )

    full_data: pd.DataFrame = pd.concat(
        [extracted_data, engineered_features],
        axis=1
    )

    normalised_data: pd.DataFrame = z_score_normalisation(full_data, 'global')
