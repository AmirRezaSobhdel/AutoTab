from AutoTab import AutoTab

if __name__ == '__main__':
    autotab = AutoTab("./sample_data/titanic/train.csv", "./sample_data/titanic/test.csv")
    autotab.analyze()
