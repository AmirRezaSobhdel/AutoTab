import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno


from Dataset import Dataset


def visualize_missing_values(dataset: Dataset):
    msno.bar(dataset.train_df)
    plt.show()
