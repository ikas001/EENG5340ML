import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import gzip
import sys
from sklearn import svm
from sklearn.linear_model import SGDClassifier

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


def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print row

def split_data(split_percentage, X_train, y_train):
    num_train_rows=split_percentage*60000/100
    num_test_rows=60000-num_train_rows   
    X_train, X_val = X_train[:num_train_rows], X_train[num_train_rows:]
    y_train, y_val = y_train[:num_train_rows], y_train[num_train_rows:]
    return X_train,X_val,y_train,y_val 


def data_collection(X_val,y_val,y_hat):

    k=y_val!=y_hat
    #print k

    #print 'images where there is no match **3'
    No_match_index = np.where(k==True)
    #print No_match_index[0]
    print "# of no Matches",len(No_match_index[0])    

    #print 'images where there is match **4'
    Match_index = np.where(k==False)
    #print Match_index[0]
    print "# of Matches",len(Match_index[0])   
    #print '*****************4'
    #print len(Match_index[0]) 

    random_index_match=Match_index[0][20]
    random_index_no_match=No_match_index[0][20]  

    print 'matched Xtest Shape ***5'
    print X_val[random_index_match].shape  
    

    match_image=np.reshape(X_val[random_index_match],(28,28))
    plt.imshow(match_image,cmap='gray')
    #plt.show()
    plt.savefig('sample.png')
    print y_val[random_index_match]
    #print '*****************6'
    #print X_val[random_index_no_match].shape
    no_match_image=np.reshape(X_val[random_index_no_match],(28,28))
    plt.imshow(no_match_image,cmap='gray')
    #plt.show()  
    plt.savefig('sample_no_match.png')

    print '*****************7'
    print y_val[random_index_no_match],y_hat[random_index_no_match]

    return len(Match_index[0]), len(No_match_index[0]) 

def MLP_Classify(X_train,X_val,y_train,y_val):
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)  

    mlp.fit(X_train, y_train)   

    print '***********1'
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_val, y_val))
    print '***********1.1'
    y_hat=mlp.predict(X_train)
    print 'Training accuracy using accuracy_score function',accuracy_score(y_train,y_hat)
    y_hat=mlp.predict(X_val)
    print 'Validation accuracy using accuracy_score function',accuracy_score(y_val,y_hat) 

    return y_hat   

def SVM_Classify(X_train,X_val,y_train,y_val):
    print "SVM classifier"
    clf = svm.SVC() #define the SVM classifier.
    print "Fitting SVM classifier"
    clf.fit(X_train,y_train) #let's train this sucker!
    print "Predict SVM classifier"
    y_hat=clf.predict(X_train)
    print 'Training accuracy using SVM function',accuracy_score(y_train, y_hat)
    y_hat=clf.predict(X_val)
    print 'Validation accuracy using SVM function',accuracy_score(y_val, y_hat)

    return y_hat

def SGD_Classify(X_train,X_val,y_train,y_val):
    print "SGD classifier"
    clf = SGDClassifier(loss="hinge", penalty="l2") #define the SGD classifier.
    print "Fitting SGD classifier"
    clf.fit(X_train,y_train) #let's train this sucker!
    print "Predict SGD classifier"
    y_hat=clf.predict(X_train)
    print 'Training accuracy using SGD function',accuracy_score(y_train, y_hat)
    y_hat=clf.predict(X_val)
    print 'Validation accuracy using SGD function',accuracy_score(y_val, y_hat)
    
    return y_hat

if __name__=="__main__":
    p=int(sys.argv[1])
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    
    #match_image=np.reshape(X_test[1113],(28,28))
    #match_image *= 256
    #match_image=np.int_(match_image)
    #ascii_show(match_image)

    #Xtr,Xte,ytr,yte=get_data(p)

    ## Splitting and Classifying Training Data
    print "Extracted train Data" , X_train.shape,y_train.shape
    print "Extracted test Data" , X_test.shape, y_test.shape
    # Split Train Data
    X_train,X_val,y_train,y_val = split_data(p, X_train, y_train)

    print "\nSplit percentage",p,"%"
    print "Splitted train Data" , X_train.shape,y_train.shape
    print "splitted train Validation Data" , X_val.shape, y_val.shape
    #classify(X_train,X_val,y_train,y_val)
    
    ## Splitting  and Classifying Test Data
    #X_train,X_val,y_train,y_val = split_data(p, X_test, y_test)
    y_hat =MLP_Classify(X_train,X_val,y_train,y_val)
    
    #### TOOO SLOOOOW
    #SVM_Classify(X_train,X_val,y_train,y_val)
    ## SGD
    #y_hat = SGD_Classify(X_train,X_val,y_train,y_val)
    print (data_collection(X_val,y_val,y_hat))
