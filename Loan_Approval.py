import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

LoanData = pd.read_csv('Loan.csv')
LoanPrep = LoanData.copy()

LoanPrep.isnull().sum(axis=0)
LoanPrep = LoanPrep.dropna()
LoanPrep = LoanPrep.drop(['gender'], axis=1)

LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)

scalar_ = StandardScaler()
LoanPrep['income'] = scalar_.fit_transform(LoanPrep[['income']])
LoanPrep['loanamt'] = scalar_.fit_transform(LoanPrep[['loanamt']])

X = LoanPrep.drop(['status_Y'], axis=1)
Y = LoanPrep['status_Y']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
lr.score(X_test, y_test)

 
