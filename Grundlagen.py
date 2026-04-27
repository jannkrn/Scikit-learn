from sklearn.linear_model import LinearRegression

# Eingabedaten X müssen zweidimensional sein
X = [[1], [2], [3], [4], [5]]

# Zielwerte y
y = [10, 20, 30, 40, 50]

# Modell erstellen
model = LinearRegression()

# Modell trainieren
model.fit(X, y)

# Vorhersage für neuen Wert
prediction = model.predict([[6]])

print(prediction)