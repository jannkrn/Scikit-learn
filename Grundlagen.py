from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [10, 20, 30, 40, 50, 60, 70, 80]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42 #random_state sorgt dafür, dass die Aufteilung reproduzierbar ist
)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) #R^2 score genauigkeit des Modells auf den Testdaten

from sklearn.tree import DecisionTreeClassifier

# Features: [Lieferzeit in Tagen, Artikelmenge]
X = [
    [1, 10],
    [2, 20],
    [3, 30],
    [7, 100],
    [8, 120],
    [10, 200]
]

# Ziel: 0 = nicht verspätet, 1 = verspätet
y = [0, 0, 0, 1, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)

prediction = model.predict([[6, 90]])

print(prediction)