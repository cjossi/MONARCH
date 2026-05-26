# MONARCH
# Author: Corentin Jossi
# Date: 20.05.2026
# Description: This script is responsible to merge all data extraction.
# ------------------------------------------------------------------------------

# Third Party Imports
import pandas as pd

# Local Imports
import monarch.config as config

def data_extraction() -> pd.DataFrame:
    """
    Future implementation will involve reading raw data from a CSV file

    Returns
    -------
    DataFrame
        A Pandas DataFrame containing the extracted gait parameters
    """

    cfg = config.Config.load_from_yaml()

    gait_parameters_folder = cfg.gait_parameters_folder

    # Read csv file:
    gait_parameters = pd.read_csv(
        gait_parameters_folder / 'gait_parameters.csv'
    )


    return gait_parameters

# ------------------------------------------------------------------------------
# Variability Indices Calculation
# ------------------------------------------------------------------------------

def calculate_asymmetry_variability_indice(
        CoV_left: float,
        CoV_right: float
) -> float:
    """
    Calculate the asymmetry index based on the coefficient of variation (CoV)
    for the left and right sides.

    Parameters
    ----------
    CoV_left : float
        Coefficient of variation for the left side
    CoV_right : float
        Coefficient of variation for the right side

    Returns
    -------
    float
        The calculated asymmetry index
    """

    asymmetry_index = (CoV_left - CoV_right) / (CoV_left + CoV_right)

    return asymmetry_index

def calculate_spatio_temporal_variability_indice(
        CoV_stride_length: float, 
        CoV_velocity: float
) -> float:
    """
    Calculate the spatio-temporal variability indice based on the coefficient of 
    variation (CoV) for stride length and stride time.

    Parameters
    ----------
    CoV_stride_length : float
        Coefficient of variation for stride length
    CoV_velocity : float
        Coefficient of variation for velocity

    Returns
    -------
    float
        The calculated spatio-temporal variability indice
    """

    spatio_temporal_indice = CoV_stride_length / CoV_velocity

    return spatio_temporal_indice

def calculate_global_variability_indice(
        Cov_stride_length_left: float,
        Cov_stride_length_right: float, 
        CoV_velocity_left: float,
        CoV_velocity_right: float
) -> float:
    """
    Calculate the global variability indice based on the coefficient
    of variation (CoV) for stride length and velocity.

    Parameters
    ----------
    Cov_stride_length_left : float
        Coefficient of variation for stride length (left side)
    Cov_stride_length_right : float
        Coefficient of variation for stride length (right side)
    CoV_velocity_left : float
        Coefficient of variation for velocity (left side)
    CoV_velocity_right : float
        Coefficient of variation for velocity (right side)

    Returns
    -------
    float
        The calculated global variability indice
    """

    global_variability_indice = (
        Cov_stride_length_left +
        Cov_stride_length_right + 
        CoV_velocity_left + 
        CoV_velocity_right
    ) / 4

    return global_variability_indice

def calculate_support_asymmetry_indice(
        single_support_time_left: float,
        single_support_time_right: float
) -> float:
    """
    Calculate the support asymmetry index based on the single and double
    support times.

    Parameters
    ----------
    single_support_time_left : float
        Single support time for the left side
    single_support_time_right : float
        Single support time for the right side

    Returns
    -------
    float
        The calculated support asymmetry index
    """

    support_asymmetry_indice = (
        (single_support_time_left - single_support_time_right) / 
        (single_support_time_left + single_support_time_right)
    )

    return support_asymmetry_indice

def calculate_variability_indices(
        extracted_data: pd.DataFrame
) -> tuple[float, float, float, float]:
    """
    Calculate all variability indices.

    Returns
    -------
    tuple[float, float, float, float]
        A tuple containing the values of the calculated variability indices
    """

    CoV_stride_length_left = extracted_data[
        'CoV_stride_length_left (%)'
    ].iloc[0]
    CoV_stride_length_right = extracted_data[
        'CoV_stride_length_right (%)'
    ].iloc[0]
    CoV_velocity_left = extracted_data[
        'CoV_velocity_left (%)'
    ].iloc[0]
    CoV_velocity_right = extracted_data[
        'CoV_velocity_right (%)'
    ].iloc[0]
    single_support_time_left = extracted_data[
        'mean_single_support_time_left (s)'
    ].iloc[0]
    single_support_time_right = extracted_data[
        'mean_single_support_time_right (s)'
    ].iloc[0]

    asymmetry_index = calculate_asymmetry_variability_indice(
        CoV_stride_length_left, CoV_stride_length_right
    )
    spatio_temporal_indice = calculate_spatio_temporal_variability_indice(
        CoV_stride_length_left, CoV_velocity_left
    )
    global_variability_indice = calculate_global_variability_indice(
        CoV_stride_length_left, CoV_stride_length_right, 
        CoV_velocity_left, CoV_velocity_right
    )
    support_asymmetry_indice = calculate_support_asymmetry_indice(
        single_support_time_left, single_support_time_right
    )

    return (
        asymmetry_index,
        spatio_temporal_indice,
        global_variability_indice,
        support_asymmetry_indice
    )

def z_score_normalisation(value: float, mean: float, std: float) -> float:
    """
    Perform z-score normalization on a given value.

    Parameters
    ----------
    value : float
        The value to be normalized
    mean : float
        The mean of the dataset
    std : float
        The standard deviation of the dataset

    Returns
    -------
    float
        The z-score normalized value
    """

    if std == 0:
        raise ValueError(
            "Standard deviation cannot be zero for z-score normalization."
        )

    z_score = (value - mean) / std

    return z_score

def variability_indices_normalisation(
        extracted_data: pd.DataFrame,
        normalisation_type: str
) -> tuple[float, float, float, float]:
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
    tuple[float, float, float, float]
        A tuple containing the normalized values of the variability indices
    """

    # The normalisation is false here, to correct.
    if normalisation_type == 'global':
        # take only the columns corresponding to numeral values:
        data = extracted_data.drop(columns=[
            'timeline_stage', 'test_type', 'snr_id'
        ])

        normalised_data = pd.DataFrame(index = range(1), columns=data.columns)

        for column in data.columns:
            mean = data[column].mean()
            std = data[column].std()

            normalised_data[column] = data[column].apply(
                lambda x: z_score_normalisation(x, mean, std)
            )
        
        print("Global Normalization:")
        print(normalised_data)

            



    return None


if __name__ == "__main__":
    extracted_data = data_extraction()

    variability_indices_normalisation(extracted_data, 'global')