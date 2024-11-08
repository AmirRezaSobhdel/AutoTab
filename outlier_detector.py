from enum import Enum

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from Dataset import Dataset, ColumnType
from utils.utils import get_validated_input


class DatasetType(Enum):
    UNIVARIATE = "univariate"
    MULTIVARIATE_LOW_DIMENSIONAL = "multivariate_low_dimensional"
    MULTIVARIATE_HIGH_DIMENSIONAL = "multivariate_high_dimensional"


def detect_dataset_numerical_columns_type(dataset: Dataset):

    num_columns = dataset.get_numeric_columns_count()

    if num_columns == 1:
        return DatasetType.UNIVARIATE
    elif 2 <= num_columns < 10:
        return DatasetType.MULTIVARIATE_LOW_DIMENSIONAL
    else:
        return DatasetType.MULTIVARIATE_HIGH_DIMENSIONAL


def isolation_forest_outlier_detector(dataset: Dataset):

    # todo: show proper logs and ask proper questions from the user

    df = dataset.train_df

    # Identify the numerical columns based on c_type
    numerical_columns = [col for col, series in dataset.train_df.items() if series.c_type == ColumnType.NUMERICAL]
    numerical_data = df[numerical_columns]

    # Outlier Detection using Isolation Forest
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)

    # todo hyperparameter: what about different contamination and random_state?
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_predictions = iso_forest.fit_predict(scaled_data)

    # Remove the outliers from the original DataFrame
    df_cleaned = df[outlier_predictions != -1]

    # Update the dataset's train_df with the cleaned DataFrame
    dataset.update_train_df(df_cleaned)


def lof_local_outlier_factor_outlier_detector(dataset: Dataset):

    # todo: show proper logs and ask proper questions from the user

    df = dataset.train_df

    numerical_columns = [col for col, series in dataset.train_df.items() if series.c_type == ColumnType.NUMERICAL]
    numerical_data = df[numerical_columns]

    # Outlier Detection using Local Outlier Factor (LOF)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)

    options = {"n_neighbors": int}
    defaults = {"n_neighbors": 20}
    user_inputs = get_validated_input(
        "Local Outlier Factor algorithm will be applied to detect and remove the outliers."
        " Please specify the hyperparameters.", options, defaults)

    n_neighbors = user_inputs.get("n_neighbors", 20)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    outlier_predictions = lof.fit_predict(scaled_data)

    # Remove the outliers from the original DataFrame
    df_cleaned = df[outlier_predictions != -1]

    dataset.update_train_df(df_cleaned)


def iqr_interquartile_range_outlier_detector(dataset: Dataset):

    df = dataset.train_df
    col = None
    for _, series in dataset.train_df:
        if series.c_type == ColumnType.NUMERICAL:
            col = series
            break

    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    lower_outliers = df[df[col] < lower_bound]
    upper_outliers = df[df[col] > upper_bound]

    if not upper_outliers.empty:
        mssg = f"Using Interquartile Range (IQR) method, In column {col.name}, Values larger than {upper_bound}" \
               f" are recognized as outliers." \
               f" Do you approve? If yes, type yes, If not, type in the upper bound you think is better."

        inp = get_validated_input(mssg, {"yes": str, "upper_bound": float})

        if inp != "yes":
            upper_bound = float(inp)
        df = df[df[col] <= upper_bound]

    if not lower_outliers.empty:
        mssg = f"Using Interquartile Range (IQR) method, In column {col.name}, Values smaller than {lower_bound}" \
               f" are recognized as outliers." \
               f" Do you approve? If yes, type yes, If not, type in the lower bound you think is better."

        inp = get_validated_input(mssg, {"yes": str, "lower_bound": float})

        if inp != "yes":
            lower_bound = float(inp)
        df = df[df[col] <= lower_bound]

    dataset.update_train_df(df)


def find_and_remove_outliers(dataset: Dataset):
    type = detect_dataset_numerical_columns_type(dataset)

    if type == DatasetType.UNIVARIATE:
        iqr_interquartile_range_outlier_detector(dataset)
    elif type == DatasetType.MULTIVARIATE_LOW_DIMENSIONAL:
        lof_local_outlier_factor_outlier_detector(dataset)
    elif type == DatasetType.MULTIVARIATE_HIGH_DIMENSIONAL:
        isolation_forest_outlier_detector(dataset)
