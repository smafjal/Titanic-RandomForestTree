import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# thanks to https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish/notebook
# Data Normalization
def normalize_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def normalize_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def normalize_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def normalize_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0][:-1])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1][:-1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def encode_features(data):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data

def plot_data(data):
    sns.barplot(x="Age", y="Survived", hue="Sex", data=data);
    plt.show()

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    # print "Data Size: ",data.shape
    data=normalize_ages(data)
    data=normalize_fares(data)
    data=normalize_cabins(data)
    data=normalize_name(data)
    data=drop_features(data)
    data=encode_features(data)
    return data


def main():
    train_path='data/train.csv'
    test_path = 'data/test.csv'
    data=preprocess_data(train_path)
    print "-- Model Data --"
    print data.head()
    print "*"*70
    plot_data(data=data)

if __name__ == "__main__":
    main()
