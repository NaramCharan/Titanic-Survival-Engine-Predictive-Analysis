import pandas as pd

class preprocessing:
    def __init__(self):
        self.df = pd.read_csv('../dataset/Titanic-Dataset.csv')

    def feature_engineering(self):
        self.df['Family_size'] = self.df['Parch'] + self.df['SibSp'] + 1
        self.df = self.df.drop(columns=['Parch', 'SibSp'])
        self.df['Name_len'] = self.df['Name'].str.len()
        self.df['Word_len'] = self.df['Name'].str.split().str.len()
        self.df['Title'] = self.df['Name'].str.split().str[1].str.split(".").str[0]

        X = self.df.drop(columns=['PassengerId', 'Survived', 'Ticket', 'Name'])
        y = self.df['Survived']
        return X, y






