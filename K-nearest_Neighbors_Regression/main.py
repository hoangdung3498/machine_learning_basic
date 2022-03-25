'''
Với bài toán Regression, chúng ta cũng hoàn toàn có thể sử dụng phương pháp tương tự:
ước lượng đầu ra dựa trên đầu ra và khoảng cách của các điểm trong K-lân cận.
Việc ước lượng như thế nào các bạn có thể tự định nghĩa tùy vào từng bài toán.
'''
'''
Khi có một thuộc tính trong dữ liệu (hay phần tử trong vector) 
lớn hơn các thuộc tính khác rất nhiều (ví dụ thay vì đo bằng cm 
thì một kết quả lại tính bằng mm), khoảng cách giữa các điểm sẽ phụ thuộc vào 
thuộc tính này rất nhiều. Để có được kết quả chính xác hơn, 
một kỹ thuật thường được dùng là Data Normalization (chuẩn hóa dữ liệu) 
để đưa các thuộc tính có đơn vị đo khác nhau về cùng một khoảng giá trị, 
thường là từ 0 đến 1, trước khi thực hiện KNN. 
'''
'''
Sử dụng các phép đo khoảng cách khác nhau
Ngoài norm 1 và norm 2 tôi giới thiệu trong bài này, 
còn rất nhiều các khoảng cách khác nhau có thể được dùng. 
Một ví dụ đơn giản là đếm số lượng thuộc tính khác nhau giữa hai điểm dữ liệu. 
Số này càng nhỏ thì hai điểm càng gần nhau. 
Đây chính là giả chuẩn 0 mà tôi đã giới thiệu trong Tab Math.
'''
# %reset
import numpy as np
from mnist import MNIST # require `pip install python-mnist`
# https://pypi.python.org/pypi/python-mnist/

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time

# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/
mndata = MNIST('MNIST') # path to your MNIST folder
mndata.load_testing()
mndata.load_training()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)


start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print ("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print ("Running time: %.2f (s)" % (end_time - start_time))