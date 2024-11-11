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


class DatasetSize(Enum):
    TINY = "Tiny (under 1,000 rows)"
    SMALL = "Small (1,000 - 10,000 rows)"
    MEDIUM = "Medium (10,000 - 100,000 rows)"
    LARGE = "Large (100,000 - 1,000,000 rows)"
    VERY_LARGE = "Very Large (1,000,000 - 10,000,000 rows)"
    HUGE = "Huge (over 10,000,000 rows)"


class Dataset:
    def __init__(self, train: str, test: str, drop: [str], show_info: bool = False):

        self.train_df: pd.DataFrame = self._load_data(train, drop)
        self.dataset_size: DatasetSize = self.get_dataset_size(self.train_df)

        for _, series in self.train_df.items():
            series.c_type = self._get_column_type(series)

        if show_info:
            self._log_info()

    def update_train_df(self, df: pd.DataFrame):
        self.train_df = df
        for _, series in self.train_df.items():
            series.c_type = self._get_column_type(series)

    def _load_data(self, path: str, drop: [str] = None) -> pd.DataFrame:
        file_extension = os.path.splitext(path)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(path)
        elif file_extension == '.tsv':
            df = pd.read_csv(path, sep='\t')
        elif file_extension == '.xls' or file_extension == '.xlsx':
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        if drop:
            df.drop(drop, axis=1, inplace=True)

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

    def get_dataset_size(self, df: pd.DataFrame) -> DatasetSize:
        total_rows = df.shape[0]

        if total_rows < 1000:
            return DatasetSize.TINY
        elif 1000 <= total_rows < 10000:
            return DatasetSize.SMALL
        elif 10000 <= total_rows < 100000:
            return DatasetSize.MEDIUM
        elif 100000 <= total_rows < 1000000:
            return DatasetSize.LARGE
        elif 1000000 <= total_rows < 10000000:
            return DatasetSize.VERY_LARGE
        else:
            return DatasetSize.HUGE

    def get_missing_values_ratio(self):
        total_cells = self.train_df.size
        missing_cells = self.train_df.isnull().sum().sum()
        return missing_cells / total_cells

    def get_numeric_columns_count(self):
        return len([s for _, s in self.train_df.items() if s.c_type == ColumnType.NUMERICAL])

    def get_columns_count(self):
        return len(self.train_df.columns)

    def _log_info(self):
        print("Dataset size: ", self.dataset_size.value)
        print("=====================================")

        for _, series in self.train_df.items():
            print(f"Column: {series.name}, Type: {series.c_type}")
