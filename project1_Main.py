import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import gzip


def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1,784)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

X_train = load_mnist_images('train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

match_image=np.reshape(X_test[1113],(28,28))
match_image *= 256
match_image=np.int_(match_image)
def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print row
ascii_show(match_image)


print "train Data" , X_train.shape,y_train.shape
print "test data" , X_test.shape, y_test.shape

#X_train, X_val = X_train[:-10000], X_train[-10000:]
#y_train, y_val = y_train[:-10000], y_train[-10000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)

print '***********1'
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
y_hat=mlp.predict(X_train)
print 'Training accuracy using accuracy_score function',accuracy_score(y_train,y_hat)
y_hat=mlp.predict(X_test)
print 'Training accuracy using accuracy_score function',accuracy_score(y_test,y_hat)

print '*****************2'
k=y_test!=y_hat
print k

print '*****************3'
itemindex = np.where(k==True)
print itemindex[0]
print itemindex[0].shape

print '*****************4'
print len(itemindex[0])

random_index_match=1113
random_index_no_match=1112

print 'matched Xtest Shape ***5'
print X_test[random_index_match].shape


match_image=np.reshape(X_test[random_index_match],(28,28))
plt.imshow(match_image,cmap='gray')
plt.show()
print y_test[random_index_match]
print '*****************6'
print X_test[random_index_no_match].shape
no_match_image=np.reshape(X_test[random_index_no_match],(28,28))
plt.imshow(no_match_image,cmap='gray')
plt.show()

print '*****************7'
print y_test[random_index_no_match],y_hat[random_index_no_match]