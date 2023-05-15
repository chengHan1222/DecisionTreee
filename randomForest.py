import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

# 交叉驗證計算得分
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def decisionTree():
    with open('data/iris-utf8.csv', encoding='utf-8') as csv_data:
        csv = pd.read_csv(csv_data)
        # csv.head()
        # csv.info()
        # sns.pairplot(csv, hue='戶外')

        x = csv.drop('分類', axis=1)
        y = csv['分類']
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.30, random_state=101)

        # print("x_train", x_train)
        # print("x_test", x_test)
        # print("y_train", y_train)
        # print("y_test", y_test)

        dtree = DecisionTreeClassifier()
        dtree.fit(x_train, y_train)
        # print(dtree.score(x_test, y_test))

        from sklearn.tree import export_graphviz
        export_graphviz(dtree)

        predictions = dtree.predict(x_test)
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))


def random_forest_origin():
    wine = load_wine()
    print(wine.target)

    x_train, x_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3)
    rf = RandomForestClassifier(n_estimators=40).fit(x_train, y_train)
    print("randomforest train_score: "+str(rf.score(x_train, y_train)))
    print("randomforest test_score: "+str(rf.score(x_test, y_test)))

    rf = RandomForestClassifier(
        n_estimators=40, oob_score=True).fit(wine.data, wine.target)
    print("randomforest oob_score: "+str(rf.oob_score_))

    # rf_s = []
    # for i in range(200):
    #     randomforest = RandomForestClassifier(n_estimators=i+1)
    #     rf_s.append(cross_val_score(randomforest, wine.data, wine.target, cv=10, n_jobs=-1).mean())
    #     print(i)

    # plt.plot(range(1, 201), rf_s, label="randomforest")
    # plt.legend()
    # plt.show()


def random_forest():
    df = pd.read_csv('./data/data.csv')
    df_test = pd.read_csv('./data/data_test.csv')

    X_train_data = df.drop(['程式碼數', '分數'], axis=1)
    Y_train_data = df['分數']
    X_testdata = df_test.drop(['程式碼數', '分數'], axis=1)
    Y_testdata = df_test['分數']

    # 數據標準化
    scale = StandardScaler().fit(X_train_data)
    X_train_stand = scale.transform(X_train_data)
    X_test_stand = scale.transform(X_testdata)

    print(X_train_data)
    # print('----------------------------------------------')

    # x_train, x_test, y_train, y_test = X_train_data, X_testdata, Y_train_data, Y_testdata
    x_train, x_test, y_train, y_test = X_train_stand, X_test_stand, Y_train_data, Y_testdata
    # -- 分割訓練集 --------
    # x_train, x_test, y_train, y_test = train_test_split(X_testdata, Y_testdata, test_size=0.3)

    # -- 窮舉 n_estimators（子树的数量）影響參數 --------
    # final_score = []
    # final_oob_score = []
    # for i in range(10, 40):
    #     # rf = RandomForestClassifier(n_estimators=i + 1, random_state=42, oob_score=True)
    #     # score = cross_val_score(rf, x_train, y_train, cv=2).mean()
    #     # final_score.append(score)

    #     rf = RandomForestClassifier(n_estimators=i + 1, random_state=42, oob_score=True).fit(x_train, y_train)

    #     final_score.append(rf.score(x_test, y_test))
    #     final_oob_score.append(rf.oob_score_)

    # score_max = max(final_score)
    # print('最大得分：{}'.format(score_max),
    #       '子树数量为：{}'.format(final_score.index(score_max)+10 + 1))

    # oob_score_max = max(final_oob_score)
    # print('最大OOB得分：{}'.format(oob_score_max),
    #       '子树数量为：{}'.format(final_oob_score.index(oob_score_max)+10 + 1))

    # # # 绘制学习曲线
    # import numpy as np
    # x = np.arange(11, 41)
    # plt.subplot(111)
    # plt.plot(x, final_score, 'r-')
    # plt.show()

    # -- 窮舉 調整 max_depth 影響參數 --------
    # rf = RandomForestClassifier(n_estimators=14, random_state=42, oob_score=True).fit(x_train, y_train)

    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'max_depth':np.arange(1,20)}
    # GS = GridSearchCV(rf, param_grid, cv=2)
    # GS.fit(x_test, y_test)

    # best_param = GS.best_params_
    # best_score = GS.best_score_
    # print(best_param, best_score)

    # -- 隨機森林建立 --------
    # rf = RandomForestClassifier(n_estimators=18, random_state=42, max_depth=2, oob_score=True).fit(x_train, y_train)
    rf = RandomForestClassifier(n_estimators=18, random_state=42, oob_score=True).fit(x_train, y_train)
    print("randomforest train_score: " + str(rf.score(x_train, y_train)))
    print("randomforest test_score: " + str(rf.score(x_test, y_test)))
    print("randomforest oob_score: " + str(rf.oob_score_))

    # -- 繪製圖片 --------
    # from sklearn.tree import export_graphviz
    # tree = rf.estimators_[0]
    # dot_data = export_graphviz(tree, out_file='tree.dot', rounded=True, precision=1)

    # 使用 pydot 库生成图形
    # import pydot
    # graph = pydot.graph_from_dot_data(dot_data)

    # 将图形保存为 PDF 文件
    # graph[0].write_pdf("iris.pdf")


def compare():
    # 基分類器的準確率在50%以上集成分類才有意義
    import numpy as np
    import math
    x = np.linspace(0, 1, 20)
    y = []
    for epsilon in np.linspace(0, 1, 20):
        E = np.array([math.comb(25, i)*(epsilon**i)*((1-epsilon)**(25-i))
                      for i in range(13, 26)]).sum()
        y.append(E)
    plt.plot(x, y, "o-", label="when estimators are different")
    plt.plot(x, x, "--", color="red", label="if all estimators are same")
    plt.xlabel("individual estimator's error")
    plt.ylabel("RandomForest's error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # from sklearn import datasets
    # iris = datasets.load_iris()
    # x = iris.data
    # y = iris.target
    # print(x)
    # print('-----------------------')
    # print(y)

    # decisionTree()
    # random_forest_origin()
    random_forest()
    # compare()
