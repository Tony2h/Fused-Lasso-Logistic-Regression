from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from read_data import get_data
from sklearn.metrics import f1_score
from timeit import default_timer


f1score = []
ti = []
accuracy = []

classifiers = [
    ('Nearest Neighbors', KNeighborsClassifier(3)),  # K最近邻
    ('Linear SVM', SVC(kernel='linear', C=0.025)),  # 线性的支持向量机
    ('RBF SVM', SVC(gamma=2, C=1)),  # 径向基函数的支持向量机
    ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),  # 随机森林
    ('Naive Bayes', GaussianNB()),  # 朴素贝叶斯
    ('MLP', MLPClassifier(alpha=1)),  # 多层感知机
]

X_train, y_train, X_test, y_test = get_data(standard=1, normal=1)

for _, clf in classifiers:
    start = default_timer()
    clf.fit(X_train, y_train)
    end = default_timer()
    ti.append(end-start)

    accuracy.append(clf.score(X_test, y_test))

    y_hat = clf.predict(X_test)
    f1score.append(f1_score(y_test, y_hat))
