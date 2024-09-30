
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing 
import pandas as pd 

regression_mode = 'regression'
classification_mode = 'classification'
label_encoder = preprocessing.LabelEncoder()

def california_housing():
    data = fetch_california_housing()
    X = data['data']
    y = data['target']
    
    return X, y, "california housing", regression_mode


def diabetes():
    data = load_diabetes()
    X = data['data']
    y = data['target']
    
    return X, y, "diabetes", regression_mode


# MPG Dataset (Auto MPG)
# Description: Predict miles per gallon used in cars.
def miles_per_gallon():
    data = sns.load_dataset('mpg')
    data.dropna(inplace=True)  # Removing NA values
    X = data.drop('mpg', axis=1).values
    y = data['mpg'].values

    return X, y, "miles/gallon", regression_mode

#Description: Predict tips given in restaurants.
def tips():
    data = sns.load_dataset('tips')
    X = data.drop('tip', axis=1)
    y = data['tip']

    return X, y, 'tips'

# Longley Dataset (Multicollinearity Test)
# Description: A highly multicollinear dataset to test regression algorithms
def totem():
    data = sm.datasets.longley.load_pandas().data
    X = data.drop('TOTEMP', axis=1)
    y = data['TOTEMP']

    return X, y, 'totem'

#Stackloss Dataset
#Description: Predict the operation efficiency of a plant
def stackloos():
    data = sm.datasets.stackloss.load_pandas().data
    X = data.drop("STACKLOSS", axis=1)
    y = data["STACKLOSS"]

    return X, y, 'stackloss'

# Fair's Extramarital Affairs Data
# Description: Predict the rate of extramarital affairs.
def extramarital_affairs():
    data = sm.datasets.fair.load_pandas().data
    X = data.drop('affairs', axis=1).values
    y = data['affairs'].values

    return X, y, 'extramarital_affairs', regression_mode

# Guerry Dataset
# Description: Data used for an early example of data-driven socio-economic analysis.
def guerry():
    data = sm.datasets.get_rdataset("Guerry", "HistData").data
    X = data.drop(['Lottery', 'Region', 'Department'], axis=1)  # Choose 'Lottery' or another as the target
    y = data['Lottery'].values

    return X.values[:,1:], y, 'querry', regression_mode

# ModeChoice Dataset
# Description: Model the choice of transportation modes.
def mode_choice():
    data = sm.datasets.modechoice.load_pandas().data
    X = data.drop('choice', axis=1).values
    y = data['choice'].values

    return X, y, 'mode choice', regression_mode

# Heart Dataset
# Description: Predict mortality from heart failure using clinical records.
def heart_mortality():
    heart = fetch_openml(name="heart-statlog", version=1)
    X = heart.data.values
    y = heart.target
    y = label_encoder.fit_transform(y)


    return X, y, 'heart mortality', classification_mode

# Wisconsin Breast cancer dataset
def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target

    return X, y, 'breast cancer', classification_mode


#Statlog (Heart) Dataset
#Description: This heart disease database contains 13 attributes used to determine the presence of heart disease in the patient. 
#It is another staple in binary classification tasks.
def statlog_heart():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
    df = pd.read_csv(url, header=None, sep=' ')
    
    X = df.values[:,:-1]
    y = df.values[:,-1]
    y[y == 2] = 0

    return X, y, 'Statlog Heart', classification_mode


# Australian Credit Approval Dataset
# Description: This dataset concerns credit card applications. 
# All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.
def credit_approval():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    df = pd.read_csv(url, header=None, sep=' ')
    
    X = df.values[:,:-1]
    y = df.values[:,-1]

    return X, y, 'credit approval', classification_mode


# Sonar, Mines vs. Rocks Dataset
# Description: This dataset involves the prediction of whether an object is a mine or a rock based on sonar returns. 
# It's an excellent example of binary classification with a balanced feature set.
def sonar():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df = pd.read_csv(url, header=None, sep=' ')
    
    X = df.values[:,:-1]
    y = df.values[:,-1]

    return X, y, 'Sonar', classification_mode
