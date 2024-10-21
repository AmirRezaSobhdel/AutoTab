import os
import pandas as pd

from Column import Column


class AutoTab:
    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.columns = []

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

    def fit(self, train: str, test: str):
        train_df: pd.DataFrame = self._load_data(train)

        for _, series in train_df.items():
            c = Column(series)
            self.columns.append(c)
            print(c)





