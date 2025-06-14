import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#read data
titanic_df = pd.read_csv('../datasets/titanic_processed_train.csv')
#Removes the label 'Survived' to make the model predict it, saves it in an y to compare later
y = titanic_df['Survived']
x = titanic_df.drop('Survived', axis=1)
#Separate the data in train and test, 80% to the train and 20% to the test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
#Create the model
logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=100)
#Put the model to train
logistic_model.fit(x_train, y_train)
#The model makes predict
y_test_pred = logistic_model.predict(x_test)

#Evaluate the model
acc = accuracy_score(y_test, y_test_pred) #How many guesses the model make it right
prec = precision_score(y_test, y_test_pred) #How many the passengers that the model thought survived actually survived
recall = recall_score(y_test, y_test_pred)#How of the actual survivors did the model correctly predict

#Print the plot of the confustion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
#Add the metrics to the plot
metrics_text = f'Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {recall:.2f}'
plt.gcf().text(0.7, 0.01, metrics_text, fontsize=12, ha='left')  # gcf = get current figure
plt.show()



