from Dataset import Dataset
from missing_values import visualize_missing_values


class AutoTab:
    def __init__(self, train: str, test: str, verbosity: int = 2):
        self.verbosity = verbosity
        self.dataset = Dataset(train, test)

    def analyze(self):
        visualize_missing_values(self.dataset)






