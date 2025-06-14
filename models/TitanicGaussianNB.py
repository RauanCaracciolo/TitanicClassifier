import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.naive_bayes import GaussianNB


#read data
titanic_df = pd.read_csv('../datasets/titanic_processed_train.csv')
#Removes the label 'Survived' to make the model predict it, saves it in an y to compare later
y = titanic_df['Survived']
x = titanic_df.drop('Survived', axis=1)
#Separate the data in train and test, 80% to the train and 20% to the test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
#Create the model
titanic_dt = GaussianNB()
#Put the model to train
titanic_dt.fit(x_train, y_train)

y_test_pred = titanic_dt.predict(x_test)

#Evaluate the model
acc = accuracy_score(y_test, y_test_pred) #How many guesses the model make it right
prec = precision_score(y_test, y_test_pred) #How many the passengers that the model thought survived actually survived
recall = recall_score(y_test, y_test_pred)#How of the actual survivors did the model correctly predict

#Print the plot of the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
#Add the metrics to the plot
metrics_text = f'Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {recall:.2f}'
plt.gcf().text(0.7, 0.01, metrics_text, fontsize=12, ha='left')  # gcf = get current figure
plt.show()

