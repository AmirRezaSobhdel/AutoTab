import os
from enum import Enum
import pandas as pd


class ColumnType(Enum):
    NUMERICAL = "Numerical"
    BOOLEAN = "Boolean"
    CATEGORICAL_LOW_CARDINALITY = "Low Cardinality Categorical"
    CATEGORICAL_HIGH_CARDINALITY = "High Cardinality Categorical"
    TEXTUAL_ENTITY = "Textual Entity"
    UNIQUE_IDENTIFIER = "Unique Identifier"
    DATETIME = "Datetime"
    BINARY = "Binary"
    UNKNOWN = "Unknown"


class Dataset:
    def __init__(self, train: str, test: str):

        self.train_df: pd.DataFrame = self._load_data(train)

        for _, series in self.train_df.items():
            series.c_type = self._get_column_type(series)

    def update_train_df(self, df: pd.DataFrame):
        self.train_df = df
        for _, series in self.train_df.items():
            series.c_type = self._get_column_type(series)

    def _load_data(self, path: str) -> pd.DataFrame:
        file_extension = os.path.splitext(path)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(path)
        elif file_extension == '.tsv':
            df = pd.read_csv(path, sep='\t')
        elif file_extension == '.xls' or file_extension == '.xlsx':
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return df

    def _get_column_type(self, column):
        cleaned_column = column.dropna()

        unique_values = cleaned_column.nunique()

        if unique_values == 2:
            return ColumnType.BINARY

        if pd.api.types.is_bool_dtype(cleaned_column):
            return ColumnType.BOOLEAN

        elif pd.api.types.is_datetime64_any_dtype(cleaned_column):
            return ColumnType.DATETIME

        elif pd.api.types.is_numeric_dtype(cleaned_column):
            # Check if the column is likely an ID or unique identifier
            total_values = len(cleaned_column)
            if unique_values == total_values:
                return ColumnType.UNIQUE_IDENTIFIER

            # Check if the numerical column could be categorical
            # Assuming low unique values compared to total count could indicate categorical nature
            elif unique_values / total_values < 0.05:
                return ColumnType.CATEGORICAL_LOW_CARDINALITY

            # Treat as regular numerical data if not categorical or identifier
            return ColumnType.NUMERICAL

        elif pd.api.types.is_categorical_dtype(cleaned_column) or cleaned_column.dtype == object:
            total_values = len(cleaned_column)

            # Check if it's a unique identifier
            if unique_values == total_values:
                return ColumnType.UNIQUE_IDENTIFIER

            # Low cardinality categorical
            elif unique_values / total_values < 0.05:
                return ColumnType.CATEGORICAL_LOW_CARDINALITY

            # High cardinality categorical
            elif 0.05 <= unique_values / total_values < 0.3:
                return ColumnType.CATEGORICAL_HIGH_CARDINALITY

            # Likely a textual entity if it's a string/object column with many unique values
            return ColumnType.TEXTUAL_ENTITY

        # Default to unknown if none of the above conditions match
        return ColumnType.UNKNOWN

    def get_numeric_columns_count(self):
        return len([s for _, s in self.train_df if s.c_type == ColumnType.NUMERICAL])

    def get_columns_count(self):
        return len(self.train_df.columns)
