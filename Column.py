from enum import Enum
import pandas as pd
from pandas import Series


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


class Column:
    def __init__(self, column: Series):
        self.name = column.name
        self.c_type: ColumnType = self._column_type(column)

    def _column_type(self, column):
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

    def __str__(self):
        return f"name: {self.name}\n type: {self.c_type}"
