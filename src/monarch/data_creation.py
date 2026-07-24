# MONARCH
# Author: Corentin Jossi
# Date: 20.05.2026
# Description: This script is responsible to merge all data extraction.
# ------------------------------------------------------------------------------

# Third Party Imports
import numpy as np
from pathlib import Path
import pandas as pd

# Local Imports
import monarch.config as config
from monarch.constant import (
    EPSILON,
    DLSM_NO_PCA
)

def data_extraction(
        dataset: str
) -> pd.DataFrame:
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

    if dataset == 'dataset_A':
        gait_parameters = pd.read_csv(
            gait_parameters_folder / 'gait_parameters.csv'
        )
    elif dataset == 'dataset_B':
        gait_parameters = pd.read_csv(
            gait_parameters_folder / 'gait_parameters_outcome.csv'
        )
    elif dataset == 'dataset_dlsm':
        gait_parameters = pd.read_csv(
            gait_parameters_folder / 'dlsm_parameters.csv'
        )
    elif dataset == 'dataset_clinical':
        gait_parameters = pd.read_csv(
            gait_parameters_folder / 'gait_parameters_clinical.csv'
        )
    elif dataset == 'healthy_dataset':
        gait_parameters = pd.read_csv(
            gait_parameters_folder / 'gait_parameters_healthy.csv'
        )
    else:
        raise ValueError(f"Invalid dataset for data extraction: {dataset}")

    return gait_parameters

def data_cleaning(extracted_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the extracted data by handling missing values, outliers and impossible
    values.

    Parameters
    ----------
    extracted_data : pd.DataFrame
        The DataFrame containing the extracted gait parameters
    """

    if "Timestamp" in extracted_data.columns:
        extracted_data.drop("Timestamp", axis=1, inplace=True)
        print("Dropped 'Timestamp' column from the dataset.")

    # NaN check
    print("Missing values (NaN) count per column:")
    print(extracted_data.isna().sum())

    print("Percentage of missing values per column:")
    print(extracted_data.isna().mean() * 100)

    # Infinite values check
    print("Infinite values count per column:")
    print(np.isinf(extracted_data.select_dtypes(include=np.number)).sum())

    # Check types
    print("Data types of each column:")
    print(extracted_data.dtypes)

    # Impossible values check
    # to implement
    return extracted_data

# ------------------------------------------------------------------------------
# Mean, Standard Deviation and Covariance Calculation
# ------------------------------------------------------------------------------

def calculate_variance(
        data: pd.Series
) -> float:
    """
    Calculate the variance of a pd.Series. This is done by the formula:
    Var = (std / (mean + EPSILON)) * 100
    """

    mean: float = data.mean()
    std: float = data.std(ddof=0)

    covariance: float = (std / (mean + EPSILON)) * 100

    return covariance

def variation_dataset(
        extracted_data: pd.DataFrame,
        sliding_window: bool = False
) -> pd.DataFrame:
    """
    Create a dataset containing the variability indices for all participants,
    test types and timeline stages.
    """
    grouped = extracted_data.groupby(['snr_id', 'test_type', 'timeline_stage'])

    aggregated_rows: list = []

    window_size: int = 300

    # Sliding window. Size: 300, overlap: 299.
    if sliding_window:
        for group_keys, group_data in grouped:
            for start in range(0, len(group_data) - window_size + 1):
                window = group_data.iloc[start:start + window_size]
                
                row ={
                    'snr_id': group_keys[0],
                    'test_type': group_keys[1],
                    'timeline_stage': group_keys[2]
                }

                for column in group_data.columns:
                    if column not in (
                        ['snr_id', 'test_type', 'timeline_stage'] + 
                        DLSM_NO_PCA
                        ):
                        mean = window[column].mean()
                        std = window[column].std(ddof=0)

                        row[f'CoV_{column} (%)'] = (
                            std / (mean + EPSILON) * 100
                        )
                    
                aggregated_rows.append(row)

    else:
        for group_keys, group_data in grouped:
            row ={
                'snr_id': group_keys[0],
                'test_type': group_keys[1],
                'timeline_stage': group_keys[2]
            }

            for column in group_data.columns:
                if column not in (
                    ['snr_id', 'test_type', 'timeline_stage'] + 
                    DLSM_NO_PCA
                    ):
                    row[f'CoV_{column} (%)'] = calculate_variance(
                        group_data[column]
                    )
                
            aggregated_rows.append(row)

    cfg = config.Config.load_from_yaml()

    aggregated_dataset_path: Path = cfg.aggregated_dataset_path

    aggregated_df = pd.DataFrame(aggregated_rows)

    aggregated_df.to_csv(
        aggregated_dataset_path,
        index=False
    )

    return aggregated_df

def aggregated_dataset(
        extracted_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create an aggregated dataset containing the means, standard deviations and
    variability indices for all participants, test types and timeline stages.
    It will be like the dataset A but with more features.
    """
    grouped = extracted_data.groupby(['snr_id', 'test_type', 'timeline_stage'])

    aggregated_rows: list = []

    for group_keys, group_data in grouped:
        row ={
            'snr_id': group_keys[0],
            'test_type': group_keys[1],
            'timeline_stage': group_keys[2]
        }

        for column in group_data.columns:
            if column not in (
                ['snr_id', 'test_type', 'timeline_stage'] + 
                DLSM_NO_PCA
                ):
                row[f'mean_{column}'] = group_data[column].mean()
                row[f'std_{column}'] = group_data[column].std(ddof=0)
                row[f'CoV_{column} (%)'] = calculate_variance(
                    group_data[column]
                )
            
        aggregated_rows.append(row)

    cfg = config.Config.load_from_yaml()

    aggregated_dataset_path: Path = cfg.aggregated_dataset_path

    aggregated_df = pd.DataFrame(aggregated_rows)

    aggregated_df.to_csv(
        aggregated_dataset_path,
        index=False
    )

    return aggregated_df

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
        CoV_spatial: pd.Series, 
        CoV_temporal: pd.Series
) -> pd.Series:
    """
    Calculate the spatio-temporal variability indice based on the coefficient of 
    variation (CoV) for stride length and stride time.

    Parameters
    ----------
    CoV_spatial : Series
        Coefficient of variation for spatial parameters
    CoV_temporal : Series
        Coefficient of variation for temporal parameters

    Returns
    -------
    Series
        The calculated spatio-temporal variability indice
    """

    spatio_temporal_indice: pd.Series = pd.Series(
        np.log((CoV_spatial + EPSILON) / (CoV_temporal + EPSILON))
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
        extracted_data: pd.DataFrame,
        dataset: str
) -> pd.DataFrame:
    """
    Calculate all variability indices.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the values of the calculated variability indices
    """

    if dataset == 'dataset_A':
        ASYMMETRY_COLUMNS = {
            "CoV_velocity_asymmetry_index":(
                'CoV_velocity_left (%)',
                'CoV_velocity_right (%)'
            ),
            "CoV_stride_length_asymmetry_index":(
                'CoV_stride_length_left (%)',
                'CoV_stride_length_right (%)'
            ),
            "single_support_time_asymmetry_index":(
                'mean_single_support_time_left (s)',
                'mean_single_support_time_right (s)'
            )
        }

        features = {}

        for feature, (left_col, right_col) in ASYMMETRY_COLUMNS.items():
            features[feature] = (
                calculate_asymmetry_variability_indice(
                    extracted_data[left_col],
                    extracted_data[right_col]
                )
            )

        spatio_temporal_indice = calculate_spatio_temporal_variability_indice(
            extracted_data['CoV_stride_length_left (%)'],
            extracted_data['CoV_velocity_left (%)']
        )
        global_variability_indice = calculate_global_variability_indice(
            extracted_data['CoV_stride_length_left (%)'], 
            extracted_data['CoV_stride_length_right (%)'], 
            extracted_data['CoV_velocity_left (%)'], 
            extracted_data['CoV_velocity_right (%)']
        )

        features['spatio_temporal_indice'] = spatio_temporal_indice
        features['global_variability_indice'] = global_variability_indice
    
    elif dataset == 'dataset_B':
        # ===== Extract the CoV for all parameters =====
        ASYMMETRY_COLUMNS = {
            "CoV_velocity_asymmetry_index":(
                'CoV_velocity_left (m/s) (%)',
                'CoV_velocity_right (m/s) (%)'
            ),
            "CoV_cadence_asymmetry_index":(
                'CoV_cadence_left (Hz) (%)',
                'CoV_cadence_right (Hz) (%)'
            ),
            "CoV_stride_length_asymmetry_index":(
                'CoV_stride_length_left (m) (%)',
                'CoV_stride_length_right (m) (%)'
            ),
            "CoV_stride_time_asymmetry_index":(
                'CoV_stride_time_left (s) (%)',
                'CoV_stride_time_right (s) (%)'
            ),
            "CoV_swing_time_asymmetry_index":(
                'CoV_swing_time_left (s) (%)',
                'CoV_swing_time_right (s) (%)'
            ),
            "CoV_stance_time_asymmetry_index":(
                'CoV_stance_time_left (s) (%)',
                'CoV_stance_time_right (s) (%)'
            ),
            "single_support_time_asymmetry_index":(
                'CoV_single_support_time_left (s) (%)',
                'CoV_single_support_time_right (s) (%)'
            )
        }

        features = {}

        for feature, (left_col, right_col) in ASYMMETRY_COLUMNS.items():
            features[feature] = (
                calculate_asymmetry_variability_indice(
                    extracted_data[left_col],
                    extracted_data[right_col]
                )
            )

        # ===== Calculate the spatio-temporal variability index =====
        mean_CoV_stride_length: pd.Series = (
            (extracted_data['CoV_stride_length_left (m) (%)'] + 
             extracted_data['CoV_stride_length_right (m) (%)']) / 2
        )
        mean_CoV_stride_time: pd.Series = (
            (extracted_data['CoV_stride_time_left (s) (%)'] + 
             extracted_data['CoV_stride_time_right (s) (%)']) / 2
        )
        spatio_temporal_indice = calculate_spatio_temporal_variability_indice(
            mean_CoV_stride_length, mean_CoV_stride_time
        )

        # ===== Calculate the global variability index =====
        CoVs: list = []

        for feature in ASYMMETRY_COLUMNS.items():
            left_col, right_col = feature[1]
            CoVs.append(extracted_data[left_col])
            CoVs.append(extracted_data[right_col])

        CoVs.append(extracted_data['CoV_double_support_time (s) (%)'])

        global_variability_indice = pd.concat(
            CoVs, 
            axis=1
        ).mean(axis=1)

        features['spatio_temporal_indice'] = spatio_temporal_indice
        features['global_variability_indice'] = global_variability_indice

    elif dataset == 'dataset_dlsm':
        # ===== Calculate the asymmetry indices =====
        ASYMMETRY_COLUMNS = {
            "Mean_AC_wrist_asymmetry_index":(
                'mean_AC_wrist_l',
                'mean_AC_wrist_r'
            ),
            "Mean_AC_ankle_asymmetry_index":(
                'mean_AC_ankle_l',
                'mean_AC_ankle_r'
            )
        }

        features = {}

        for feature, (left_col, right_col) in ASYMMETRY_COLUMNS.items():
            features[feature] = (
                calculate_asymmetry_variability_indice(
                    extracted_data[left_col],
                    extracted_data[right_col]
                )
            )

        # ===== Calculate the bilateral ratio =====
        bilateral_ratio = (
            extracted_data['mean_act_bilateral'] / 
            (
                extracted_data['mean_act_unilateral_wrist_l'] +
                extracted_data['mean_act_unilateral_wrist_r']
            )
        )
        features['bilateral_wrist_ratio_index'] = bilateral_ratio

    else:
        raise ValueError(f"Invalid dataset (variability indices): {dataset}")

    return pd.DataFrame(features)

def dlsm_activity_profiles(
        extracted_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Set the activity levels profiles for participant and timeline stages based
    on the DLSM dataset.

    Parameters
    ----------
    extracted_data : pd.DataFrame
        The DataFrame containing the extracted dlsm parameters
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the activity levels profiles for each participant
        and timeline stage
    """

    grouped = extracted_data.groupby(['snr_id', 'timeline_stage'])

    aggregated_rows: list = []

    for group_keys, group_data in grouped:
        row = {
            'snr_id': group_keys[0],
            'timeline_stage': group_keys[1]
        }

        sedentary_tot = group_data['Sedentary'].sum()
        low_tot = group_data['Low'].sum()
        moderate_tot = group_data['Moderate'].sum()
        vigorous_tot = group_data['Vigorous'].sum()
        total = sedentary_tot + low_tot + moderate_tot + vigorous_tot

        row['sedentary_profile'] = sedentary_tot / total
        row['low_profile'] = low_tot / total
        row['moderate_profile'] = moderate_tot / total
        row['vigorous_profile'] = vigorous_tot / total

        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)

# ------------------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------------------

def z_score_normalisation(
        extracted_data: pd.DataFrame,
        normalisation_type: str
) -> pd.DataFrame:
    """
    Compute the z-score normalisation for all variability indices.
    
    Parameters
    ----------
    extracted_data : pd.DataFrame
        The DataFrame containing the extracted gait parameters

    normalisation_type : str
        The type of normalisation to be applied 
        (e.g., 'global', 'per participant')

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the normalised values of the variability indices
    """

    data: pd.DataFrame = extracted_data.drop(columns=[
        'timeline_stage',
        'test_type',
        'snr_id'
    ])

    normalised_data: pd.DataFrame = data.copy()

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

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    extracted_data: pd.DataFrame = data_extraction('dataset_B')

    aggregated_data: pd.DataFrame = aggregated_dataset(extracted_data)

    calculated_variability_indices: pd.DataFrame = (
        calculate_variability_indices(
            aggregated_data, 'dataset_B'
        )
    )

    full_data: pd.DataFrame = pd.concat(
        [aggregated_data, calculated_variability_indices],
        axis=1
    )

    print(full_data)
