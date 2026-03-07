import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

global_df = pd.read_csv("NonNoicedDF.csv")

train, test = train_test_split(global_df, test_size = 0.3)

classes = train.pop('Result').values
features = train

testClasses = test.pop('Result').values
testFeatures = test

#
# Тренировка модели
#
model = RandomForestClassifier(n_estimators=100, max_depth=30, n_jobs=15).fit(features, classes)

#
# Отчёты о точности работы
#
print(classification_report(testClasses, model.predict(testFeatures)))
print(accuracy_score(testClasses, model.predict(testFeatures)))
import joblib

# save
joblib.dump(model, "RandomForest(100)_NoiceRed.pkl")

