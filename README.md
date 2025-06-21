<h1>Titanic Machine Learning Projetct</h1>
<h2>Developed using diverses python libraries, such as Pandas, MatPlotLib and mainly scikit-learn. <br>
<img src="https://pandas.pydata.org/static/img/pandas_mark.svg" alt="Pandas" height="40"/>
<img src="https://matplotlib.org/_static/images/logo2.svg" alt="Matplotlib" height="40"/>
<img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="scikit-learn" height="40"/></h2>

<h5>This project aim to process the classic Titanic passangers to make it usable to Binary Classification models to evaluate and compare them.</h5>
<h2>Processing the data</h2>
<h4>Bases datasets.</h4>

![Image Train](https://i.imgur.com/So591eT.png)

<h5>  We see that this dataset have many Data that its uselesss to our model, like:'PassengerId', 'Name', 'Ticket' and 'Cabin'. We need to remove them from the dataset before starting to train the model.</h5>

<h4>After useless columns removal.</h4>

![image](https://github.com/user-attachments/assets/bb057307-991d-48a3-aaa2-667e4c55a66c)

<h5>Almost there! in this example, we cannot see much, but some of the lines is incomplete, to be more specific, is missing in our near 900 rolls the exact amount in the below. </h5>

| Column    | Null values |
|------------|-------------------|
| Survived   | 179               |
| Pclass     | 179               |
| Sex        | 179               |
| Age        | 2                 |
| SibSp      | 179               |
| Parch      | 179               |
| Fare       | 179               |
| Embarked   | 177               |

<h5>We need to remove this to finally make the dataset correct.</h5>

<h4>After null lines removal.</h4>

![image](https://github.com/user-attachments/assets/a1679566-c881-4437-88c8-a61f95675730)

<h5>Ok! The useless data is sucefully full removed, but, the model can only read numeric values, so we need to transform the categorical values to numeric signficant values.</h5>

<h4>Transform female and male in 0 and 1, respectively.</h4>

![image](https://github.com/user-attachments/assets/ad363cd4-3b77-45ec-a0eb-b1cc600bc644)

<h5>We still need to transform the 'Embarked' columns in usefull data, but, unlike the binary from male and female. 'Embark' has 3 different possibilites, for this, we will create 3 new columns and made them true or false for the respective 'Embark'</h5>

<h4>Embark treated.</h4>

![image](https://github.com/user-attachments/assets/ab01d338-285b-4cd8-86b6-e6243d101568)

<h5>Finally, for good measures, we will suflle the data and the dataset is ready to be used.</h5>

<h4>Shuffled set.</h4>

![image](https://github.com/user-attachments/assets/34395fdd-2020-42ba-8d02-ca5f6e2fdb2f)

<h2>Evaluate and compare the models.</h2>

![Result](https://github.com/user-attachments/assets/29e8faf9-b7cc-4733-9a64-e0c9a69d6c72)

<h5><b>Accuracy:</b> How much is the model correct overall.</h5>
<h5><b>Precision:</b> How often the model predicting the target class.</h5>
<h5><b>Recall:</b> How % the model predict correct of the class.</h5>

