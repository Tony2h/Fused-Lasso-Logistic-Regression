import matplotlib.pyplot as plt

a = [88.2, 87.5, 90.1, 78.3, 83.6, 80.3, 83.6, 75.0]
x = ['FLLR', 'l1-LR', 'KNN', 'Naive Bayes', 'MLPerceptron', 'Random Forest', 'SVM(Linear)', 'SVM(RBF)']
colors = ['r', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy', 'navy']
f = [0.9244,0.9185,0.9051,0.8520,0.8954,0.8900,0.8971,0.8971]

plt.figure(1)
plt.bar(x, a, color=colors)
plt.xticks(x, rotation=37)
for a, b in zip(x, a):
    plt.text(a, b+0.5, '%.1f'%b+'%', ha='center', va='bottom')
plt.ylim((0,109))
plt.xlabel('Classification models')
plt.ylabel('Accuracy(%)')
plt.show()



plt.figure(2)
plt.bar(x, f, color=colors)
plt.xticks(x, rotation=37)
for a, b in zip(x, f):
   plt.text(a, b+0.03,'%.4f'%b,ha='center',va='bottom')
plt.ylim((0,1.2))
plt.yticks([0,0.6,0.7,0.8,0.9,1.0])
plt.ylabel('F1  score')
plt.xlabel('Classification models')
plt.show()
