import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
csv = pd.read_csv('dataset-firda.csv')
dataframe = pd.DataFrame(csv)
x=dataframe.copy()
y=x.pop('class')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
clf = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Use score method to get accuracy of the model
score_te = model.score(X_test, y_test)
print('Accuracy Score: ', score_te