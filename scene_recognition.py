import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from scipy import stats
from pathlib import Path, PureWindowsPath
from sklearn.neighbors import KNeighborsClassifier

def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list
    
def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()

def get_tiny_image(img, output_size):
    W = output_size[0]
    H = output_size[1]
    feature = np.zeros([W,H])
    img_reduced = cv2.resize(img, (W,H))
    for w in range(W):
        for h in range(H):
            feature[w,h] = (np.mean(img_reduced[w,h]) - np.mean(img_reduced))/np.std(img_reduced)

    feature = np.reshape(feature, (W*H))
    return feature

def predict_knn(feature_train, label_train, feature_test, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(feature_train, label_train)
    label_test_pred = knn.predict(feature_test)
    label_test_pred = np.array([label_test_pred])

    label_test_pred = np.transpose(label_test_pred)
    return label_test_pred

def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    output_size = np.array([16,16])
    k=3
    label_train = np.zeros((np.size(label_train_list,0),1))
    label_test = np.zeros((np.size(label_test_list,0),1))
    feature_train = np.zeros((np.size(label_train_list,0),output_size[0]*output_size[1]))
    feature_test = np.zeros((np.size(label_test_list,0),output_size[0]*output_size[1]))

    for i in range(np.size(label_train_list,0)):
        feature_train[i,:] = get_tiny_image(cv2.imread(img_train_list[i]) , output_size)
        label_train[i] = label_classes.index(label_train_list[i]) + 1
    for i in range(np.size(label_test_list,0)):
        feature_test[i,:] = get_tiny_image(cv2.imread(img_test_list[i]) , output_size)
        label_test[i] = label_classes.index(label_test_list[i])+1

    label_test_pred = predict_knn(feature_train, label_train.ravel(), feature_test, k)
    confusion = np.zeros((15,15))
    accuracy = 0
    for i in range(np.size(label_test,0)):
        if label_test[i] == label_test_pred[i]:
            accuracy += 1
        for j in range(np.size(label_test,0)):
            temp1 = label_test[i]
            temp2 = label_test_pred[i]
            confusion[int(temp1-1),int(temp2-1)] += 1
    confusion = confusion/100
    accuracy=accuracy/np.size(label_test,0)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy

def compute_dsift(img, stride, size):
    img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    W = np.size(img_g,0) #220
    H = np.size(img_g,1) #293
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=.02)
    d_f = np.array([])

    tracker_i = -stride
    for i in range(W//stride):
        tracker_i += stride
        center_i=tracker_i+stride//2
        tracker_j = -stride
        for j in range(H//stride):
            tracker_j+=stride
            center_j=tracker_j+stride//2
            temp_img = img_g[tracker_i:tracker_i+stride,tracker_j:tracker_j+stride]
            kp = cv2.KeyPoint(center_i,center_j,size)
            kp,des = sift.compute(img_g,[kp])
            d_f = np.append(d_f,des)

    dense_feature = np.reshape(d_f,(np.size(d_f,0)//128,128))
    return dense_feature


def build_visual_dictionary(dense_feature_list, dic_size):
    kmeans = KMeans(n_clusters=dic_size).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_
    return vocab

def compute_bow(feature, vocab):
    hist = np.zeros((np.size(vocab,0),1))
    nbrs = NearestNeighbors(n_neighbors=2).fit(vocab)
    distance, indices = nbrs.kneighbors(feature)

    for i in range(np.size(indices,0)):
        temp = indices[i,0]
        hist[temp] += 1
    bow_feature = hist/np.linalg.norm(hist)
    return bow_feature

def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    k=7
    stride = 16
    size = 16
    dic_size = 100
    label_train = np.zeros((np.size(label_train_list,0),1))
    label_test = np.zeros((np.size(label_test_list,0),1))

# ******************************************************************************
    # Budild dense features
    # dense_feature_lists1 = np.array(([[]]))
    for i in range(np.size(img_train_list,0)):
        label_train[i] = label_classes.index(label_train_list[i]) + 1
        label_test[i] = label_classes.index(label_test_list[i]) + 1
    #
    #     temp=compute_dsift(cv2.imread(img_train_list[i]),stride,size)
    #     dense_feature_lists1 = np.append(dense_feature_lists1,temp)
    # dense_feature_lists1 = np.reshape(dense_feature_lists1,(np.size(dense_feature_lists1,0)//128,128))
    #
    # vocab = build_visual_dictionary(dense_feature_lists1,dic_size)
# ******************************************************************************
    # Make bows
    # bow_feature1 = np.zeros((np.size(img_train_list,0),dic_size))
    # for i in range(np.size(img_train_list,0)):
    #     feature = compute_dsift(cv2.imread(img_train_list[i]),stride,size)
    #     temp = compute_bow(feature,vocab)
    #     bow_feature1[i,:] = temp[:,0]
    #
    # bow_feature2 = np.zeros((np.size(img_test_list,0),dic_size))
    # for i in range(np.size(img_test_list,0)):
    #     feature_test = compute_dsift(cv2.imread(img_test_list[i]),stride,size)
    #     temp = compute_bow(feature_test,vocab)
    #     bow_feature2[i,:] = temp[:,0]
# ******************************************************************************
    # OR Load previously made bow features
    # vocab = np.loadtxt('vocab100')
    # bow_feature1 = np.loadtxt('bow_features_train100')
    # bow_feature2 = np.loadtxt('bow_features_test100')
# ******************************************************************************

    label_test_pred = predict_knn(bow_feature1, label_train.ravel(), bow_feature2, k)

    confusion = np.zeros((15,15))
    accuracy = 0
    for i in range(np.size(label_test,0)):
        if label_test[i] == label_test_pred[i]:
            accuracy += 1
        for j in range(np.size(label_test,0)):
            temp1 = label_test[i]
            temp2 = label_test_pred[i]
            confusion[int(temp1-1),int(temp2-1)] += 1
    confusion = confusion/100
    accuracy=accuracy/np.size(label_test,0)
#         print("k+1 = {}, accuracy = {}".format(k,accuracy))

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy

def predict_svm(feature_train, label_train, feature_test):
    labels_train = np.zeros((1500,15))
    y_pred = np.zeros((1500,15))
    for i in range(1500):
        label = label_train[i]
        labels_train[i,int(label[0])-1]+=1

    for i in range(15):
        clf = SVC(gamma=1)
        clf.fit(feature_train,labels_train[:,i])
        y_pred[:,i] = clf.decision_function(feature_test)

    label_test_pred=np.argmax(y_pred,axis=1)+1

    return label_test_pred

def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    label_train = np.zeros((np.size(label_train_list,0),1))
    label_test = np.zeros((np.size(label_test_list,0),1))
    for i in range(np.size(img_train_list,0)):
        label_train[i] = label_classes.index(label_train_list[i]) + 1
        label_test[i] = label_classes.index(label_test_list[i]) + 1

    bow_feature1 = np.loadtxt('bow_features_train100')
    bow_feature2 = np.loadtxt('bow_features_test100')

    label_test_pred = predict_svm(bow_feature1, label_train, bow_feature2)

    confusion = np.zeros((15,15))
    accuracy = 0
    for i in range(np.size(label_test,0)):
        if label_test[i] == label_test_pred[i]:
            accuracy += 1
        for j in range(np.size(label_test,0)):
            temp1 = label_test[i]
            temp2 = label_test_pred[i]
            confusion[int(temp1-1),int(temp2-1)] += 1
    confusion = confusion/100
    accuracy=accuracy/np.size(label_test,0)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


if __name__ == '__main__':
    # Replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
