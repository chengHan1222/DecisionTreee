import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# with open('./decisionTree/data/play-utf8.csv', encoding='utf-8') as csvfile:

#     rows = csv.reader(csvfile)
#     for row in rows:
#         print(row)

df = pd.read_csv('./decisionTree/data/play-utf8.csv')
le = LabelEncoder()
for col in df[['戶外', '溫度', '濕度', '風力']]:
    df[col] = le.fit_transform(df[col])
# df['戶外'] = le.fit_transform(df['戶外'])
# df['溫度'] = le.fit_transform(df['溫度'])
# df['濕度'] = le.fit_transform(df['濕度'])
# df['風力'] = le.fit_transform(df['風力'])
# df['郊遊'] = le.fit_transform(df['郊遊'])
# df.info()
# print('---------------------------')

# sns.pairplot(df, hue="郊遊")
print(df.head(15))

from sklearn.model_selection import train_test_split

X = df.drop('郊遊', axis=1)
Y = df['郊遊']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.166)
# x_train = pd.get_dummies(x_train)
# x_test = pd.get_dummies(x_train)
# y_train = pd.get_dummies(x_train)
# y_test = pd.get_dummies(x_train)


from  sklearn.tree import DecisionTreeClassifier, plot_tree

dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

print('score train test ---------------------------------')
print(dtree.score(x_train, y_train))
print(dtree.score(x_test, y_test))

print(dtree.classes_)
# plt.rcParams['font.family'] = 'jf-openhuninn-1.0'
fig = plt.figure(figsize=(10,8), dpi=150)
# plt.plot(range(1, 10), fig, label="randomforest")
plt.legend()
plt.show()
# plot_tree(dtree)
# plot_tree(dtree, feature_names=X.columns, class_names=dtree.classes_)
# plt.savefig('tree.jpg')

# predictions = dtree.predict(x_test)
# print('predictions -----------------------------')
# print(predictions)
# from sklearn.metrics import classification_report, confusion_matrix
# print('---------------------------')
# print(classification_report(y_test, predictions))
# print('---------------------------')
# print(confusion_matrix(y_test, predictions))
# print('---------------------------')