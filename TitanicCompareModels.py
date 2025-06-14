import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.linear_model import  LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


titanic_df = pd.read_csv('datasets/titanic_processed_train.csv')
FEATURES = list(titanic_df.columns[1:]) #Extract the features(Columns name) from the data, except the Survived
print(FEATURES)
result_dict = {}
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return{'accuracy': acc,
           'precision':prec,
           'recall':recall,
           'accuracy_count':num_acc}
def build_model(classifier_fn, name_of_y_col, names_of_x_cols, dataset, test_frac=0.2):
    x = dataset[names_of_x_cols]
    y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_frac)

    model = classifier_fn(x_train, y_train)

    y_pred_test = model.predict(x_test)

    y_pred_train = model.predict(x_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred_test)

    pred_results = pd.DataFrame({'y_test': y_test,
                                 'y_pred': y_pred_test})
    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)

    return{
        'training':train_summary,
        'test':test_summary,
        'confusion_matrix': model_crosstab
    }
def compare_results():
    for key in result_dict:
        print('Classification: ', key)
        print()

        print('Training data')
        for score in result_dict[key]['training']:
            print(score, result_dict[key]['training'][score])
        print()
        print('Test data')
        for score in result_dict[key]['test']:
            print(score, result_dict[key]['test'][score])

        print()
def logistic_fn(x_train, y_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    return model
def linearDiscrAna_fn(x_train, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(x_train, y_train)
    return model

result_dict['survived ~ logistic'] = build_model(logistic_fn, 'Survived', FEATURES, titanic_df)
result_dict['survived ~ linear discriminant analysis'] = build_model(linearDiscrAna_fn, 'Survived', FEATURES, titanic_df)
compare_results()