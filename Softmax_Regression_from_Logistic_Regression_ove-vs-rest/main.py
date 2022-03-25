import numpy as np

np.random.seed(1)                          # for fixing random values

'''
Với Softmax Regression, tôi tạo simulated data như sau:
Trong ví dụ đơn giản này, số điểm dữ liệu chỉ là N = 2, số chiều dữ liệu d = 2, và số classes C = 3. 
Những giá trị đủ nhỏ này giúp cho việc kiểm tra có thể được thực hiện một cách tức thì. 
Sau khi thuật toán chạy đúng với những giá trị nhỏ này,
 ta có thể thay N, d, C bằng vài giá trị khác trước khi sử dụng dữ liệu thật.
'''
# randomly generate data
N = 2 # number of training sample
d = 2 # data dimension
C = 3 # number of classes

X = np.random.randn(d, N)
y = np.random.randint(0, 3, (N,))

'''
Ma trận one-hot codingCó một bước quan trọng nữa trong Softmax Regression là phải chuyển đổi 
mỗi label yi thành một vector yi dưới dạng one-hot coding. Trong đó, chỉ có đúng một phần tử của yi bằng 1, 
các phần tử còn lại bằng 0. Như vậy, với N điểm dữ liệu và C classes, chúng ta sẽ có một ma trận có kích thước C×N 
trong đó mỗi cột chỉ có đúng 1 phần tử bằng 1, còn lại bằng 0. Nếu chúng ta lưu toàn bộ dữ liệu này thì sẽ bị lãng phí bộ nhớ.
Một cách thường được sử dụng là lưu ma trận output Y dưới dạng sparse matrix. 
Về cơ bản, cách làm này chỉ lưu các vị trí khác 0 của ma trận và giá trị khác 0 đó.
Python có hàm scipy.sparse.coo_matrix giúp chúng ta thực hiện việc này. Với one-hot coding, tôi thực hiện như sau:
'''
## One-hot coding
from scipy import sparse
def convert_labels(y, C = C):
    """
    convert 1d label to a matrix label: each column of this
    matrix coresponding to 1 element in y. In i-th column of Y,
    only one non-zeros element located in the y[i]-th position,
    and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

Y = convert_labels(y, C)
print(y)
print(Y)

'''
Dưới đây là một đoạn code viết hàm softmax. Đầu vào là một ma trận với mỗi cột là một vector z,
đầu ra cũng là một ma trận mà mỗi cột có giá trị là a=softmax(z). 
Các giá trị của z còn được gọi là scores.
'''
def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

'''
một phương pháp đơn giản giúp khắc phục hiện tượng overflow là trừ tất cả các zi đi một giá trị đủ lớn.
Trong thực nghiệm, giá trị đủ lớn này thường được chọn là  c = maxizi. 
Vậy chúng ta có thể sửa đoạn code cho hàm softmax phía trên bằng cách trừ mỗi cột của ma trận đầu vào Z
đi giá trị lớn nhất trong cột đó. Ta có phiên bản ổn định hơn là softmax_stable:
'''
def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.
    """
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A


'''
Kiểm tra đạo hàm
Điều cốt lõi trong cách tối ưu hàm mất mát là tính gradient. 
Với biểu thức toán trông khá rối mắt như trên, rất dễ để các bạn nhầm lẫn ở một bước nào đó. 
Softmax Regression vẫn là một thuật toán đơn giản, sau này các bạn sẽ thấy nhưng biểu thức phức tạp hơn nhiều. 
Rất khó để có thể tính toán đúng gradient ở ngay lần thử đầu tiên.
Trong thực nghiệm, một cách thường được làm là so sánh gradient tính được với numeric gradient, 
tức gradient tính theo định nghĩa. Việc kiểm tra đạo hàm được thực hiện như sau:
'''
# cost or loss function
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y * np.log(A))


W_init = np.random.randn(d, C)


def grad(X, Y, W):
    A = softmax((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T)


def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps
            W_n[i, j] -= eps
            g[i, j] = (cost(X, Y, W_p) - cost(X, Y, W_n)) / (2 * eps)
    return g


g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init, cost)

print(np.linalg.norm(g1 - g2))
'''
Như vậy, sau khi in ra kết quả thì thấy sự khác biệt giữa hai đạo hàm là rất nhỏ. 
Nếu các bạn thử vài trường hợp khác nữa của N, C, d, chúng ta sẽ thấy sự sai khác vẫn là nhỏ. 
Điều này chứng tỏ đạo hàm chúng ta tính được coi là chính xác. 
(Vẫn có thể có bug, chỉ khi nào kết quả cuối cùng với dữ liệu thật là chấp nhận được thì ta mới có thể bỏ cụm từ ‘có thể coi’ đi).
Chú ý rằng, nếu N, C, d quá lớn, việc tính toán numerical_grad trở nên cực kỳ tốn thời gian và bộ nhớ. 
Chúng ta chỉ nên kiểm tra với những dữ liệu nhỏ.
'''
'''
Hàm chính cho training Softmax Regression
Sau khi đã có những hàm cần thiết và gradient được tính đúng, 
chúng ta có thể viết hàm chính có training Softmax Regression (theo SGD) như sau:
'''
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W


eta = .05
d = X.shape[0]
W_init = np.random.randn(d, C)

W = softmax_regression(X, y, W_init, eta)
print(W[-1])

'''
Hàm dự đoán class cho dữ liệu mới
Sau khi train Softmax Regression và tính được ma trận hệ số W, class của một dữ liệu mới có thể tìm được bằng cách 
xác định vị trí của giá trị lớn nhất ở đầu ra dự đoán (tương ứng với xác suất điểm dữ liệu rơi vào class đó là lớn nhất). 
Chú ý rằng, các class được đánh số là 0, 1, 2, ..., C.
'''
def pred(W, X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis = 0)

'''
Simulated data
Để minh họa cách áp dụng Softmax Regression, tôi tiếp tục làm trên simulated data.
Tạo ba cụm dữ liệu
'''
#Example
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0).T # each column is a datapoint
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)
C = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

import matplotlib.pyplot as plt


def display(X, label):
    #     K = np.amax(label) + 1
    X0 = X[:, label == 0]
    X1 = X[:, label == 1]
    X2 = X[:, label == 2]

    plt.plot(X0[0, :], X0[1, :], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[0, :], X1[1, :], 'go', markersize=4, alpha=.8)
    plt.plot(X2[0, :], X2[1, :], 'rs', markersize=4, alpha=.8)

    #     plt.axis('equal')
    plt.axis('off')
    plt.plot()
    plt.show()

#trực quan phân bố của các dữ liệu được cho bằng hàm này:
display(X[1:, :], original_label)

#Thực hiện Softmax Regression
W_init = np.random.randn(X.shape[0], C)
W = softmax_regression(X, original_label, W_init, eta)
print(W[-1])
'''
Ta thấy rằng Softmax Regression đã tạo ra các vùng cho mỗi class. 
Kết quả này là chấp nhận được. Từ hình trên ta cũng thấy rằng đường ranh giới giữa các classes là đường thẳng. 
Tôi sẽ chứng minh điều này ở phần sau.
'''

#Visualize
# x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
# x_min, x_max = -4, 14
# y_min, y_max = -4, 14

xm = np.arange(-2, 11, 0.025)
xlen = len(xm)
ym = np.arange(-3, 10, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)


# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# xx.ravel(), yy.ravel()

print(np.ones((1, xx.size)).shape)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

# print(xx.shape, yy.shape)
XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis = 0)


print(XX.shape)

Z = pred(W[-1], XX)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
# plt.figure(1
# plt.pcolormesh(xx, yy, Z, cmap='jet', alpha = .35)

CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

# Plot also the training points
# plt.scatter(X[:, 1], X[:, 2], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())
# plt.axis('equal')
display(X[1:, :], original_label)
plt.savefig('ex1.png', bbox_inches='tight', dpi = 300)
plt.show()


'''
Softmax Regression cho MNIST
Các ví dụ trên đây được trình bày để giúp bạn đọc hiểu rõ Softmax Regression hoạt động như thế nào. 
Khi làm việc với các bài toán thực tế, chúng ta nên sử dụng các thư viện có sẵn, 
trừ khi bạn có thêm bớt vài số hạng nữa trong hàm mất mat.
Softmax Regression cũng được tích hợp trong hàm sklearn.linear_model.LogisticRegression của thư viện sklearn.
'''
'''
MNIST
# %reset
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score

mntrain = MNIST('../MNIST/')
mntrain.load_training()
Xtrain = np.asarray(mntrain.train_images)/255.0
ytrain = np.array(mntrain.train_labels.tolist())

mntest = MNIST('../MNIST/')
mntest.load_testing()
Xtest = np.asarray(mntest.test_images)/255.0
ytest = np.array(mntest.test_labels.tolist())

# train
#Để sử dụng Softmax Regression, ta cần thêm một vài thuộc tính nữa:
logreg = linear_model.LogisticRegression(C=1e5, solver = 'lbfgs', multi_class = 'multinomial')
logreg.fit(Xtrain, ytrain)
'''
#Với Logistic Regression, multi_class = 'ovr' là giá trị mặc định, tương ứng với one-vs-rest.
# solver = 'lbfgs' là một phương pháp tối ưu cũng dựa trên gradient nhưng hiệu quả hơn và phức tạp hơn Gradient Descent
'''

# test
y_pred = logreg.predict(Xtest)
print "Accuracy: %.2f %%" %(100*accuracy_score(ytest, y_pred.tolist()))
Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1))/255.0, Xtrain), axis = 1)
Xtest = np.concatenate((np.ones((Xtest.shape[0], 1))/255.0, Xtest), axis = 1)
print(Xtrain_all.shape)
logreg = linear_model.LogisticRegression(C=1e5, solver = 'lbfgs', multi_class = 'multinomial')
logreg.fit(Xtrain, ytrain)
y_pred = logreg.predict(Xtest)
print "Accuracy: %.2f %%" %(100*accuracy_score(ytest, y_pred.tolist()))
'''
'''
So với kết quả hơn 91% của one-vs-rest Logistic Regression thì Softmax Regression đã cải thiện được một chút. 
Kết quả thấp như thế này là có thể dự đoán được vì thực ra Softmax Regression vẫn chỉ tạo ra 
các đường biên là các đường tuyến tính (phẳng).
'''