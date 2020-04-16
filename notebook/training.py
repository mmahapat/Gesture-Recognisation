import pickle
import os
import numpy
from sklearn import decomposition
from sklearn import preprocessing
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

master_path = '../CSV'
labels = numpy.empty((0, 0))
column_to_drop = ["Frames#", "score_overall", "nose_score", "leftEye_score", "rightEye_score", "leftEar_score",
                  "rightEar_score",
                  "leftShoulder_score", "rightShoulder_score", "leftElbow_score", "rightElbow_score", "leftWrist_score",
                  "rightWrist_score", "leftHip_score", "rightHip_score", "leftKnee_score", "rightKnee_score",
                  "leftAnkle_score",
                  "rightAnkle_score"]
columns_to_retain = ["nose_x", "nose_y", "leftEye_x", "leftEye_y", "rightEye_x", "rightEye_y", "leftEar_x", "leftEar_y",
                     "rightEar_x", "rightEar_y", "leftShoulder_x", "leftShoulder_y", "rightShoulder_x",
                     "rightShoulder_y", "leftElbow_x", "leftElbow_y", "rightElbow_x", "rightElbow_y", "leftWrist_x",
                     "leftWrist_y", "rightWrist_x", "rightWrist_y", "leftHip_x", "leftHip_y", "rightHip_x",
                     "rightHip_y", "leftKnee_x", "leftKnee_y", "rightKnee_x", "rightKnee_y", "leftAnkle_x",
                     "leftAnkle_y", "rightAnkle_x", "rightAnkle_y"]
arr = numpy.empty((0, 52), float)
total_rows = 150
final_data = pd.DataFrame()

fileExists = os.path.exists("pickBeforePca")
# Label mapping for categories
label_to_category = {0: "buy", 1: "communicate", 2: "fun", 3: "hope", 4: "mother", 5: "really"}

if not fileExists:
    for category_label in label_to_category:
        temp_data = pd.DataFrame(columns=columns_to_retain)
        path = os.path.join(master_path, label_to_category[category_label])
        total_files = len(os.listdir(str(path)))
        i = 1
        for file in os.listdir(str(path)):
            data = pd.read_csv(os.path.join(path, file))
            data.drop(column_to_drop, axis=1, inplace=True)
            while data.shape[0] < total_rows:
                data = data.append(data, ignore_index=True)
            shape = data.shape
            if shape[0] > total_rows:
                data = data[:150]
            temp_data = temp_data.append(data, ignore_index=True)
            percent = (i / total_files) * 100
            print('\r', "%.2f" % round(percent, 2) + "% '" + label_to_category[category_label] + "' completed", end=' ')
            i += 1
        print()

        result = temp_data.values

        labels = numpy.append(labels, numpy.full((1, result.shape[0]), category_label))
        # print(pca_result.shape)
        final_data = final_data.append(pd.DataFrame(result), ignore_index=True)
    result = final_data.values
    pickBeforePca = open("pickBeforePca", 'wb')
    pickle.dump(result, pickBeforePca)
    labelsPickle = open("labelsPickle", 'wb')
    pickle.dump(labels, labelsPickle)

print()
print("Applying PCA")

result = pickle.load(open('pickBeforePca', 'rb'))
labels = pickle.load(open('labelsPickle', 'rb'))
scaler = preprocessing.StandardScaler()
scaler.fit(result)
scaled_result = scaler.transform(result)
pca = decomposition.PCA(n_components=25)
pca.fit(scaled_result)
pca_result = pca.transform(scaled_result)
# print(pca_result.shape)
test_size = 0.33
# seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(pca_result, labels, test_size=test_size,
                                                                    random_state=4)
print("Test data prepared. Go ahead and train you model")

print("Training using Decision Tree...")

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
decision_tree = open('../models/decision_tree', 'wb')
# source, destination
pickle.dump(model, decision_tree)
result = model.score(X_test, Y_test)

print("Decision Tree Accuracy: %.3f%%" % (result * 100.0))
print("Training Complete")

print("Training using KNeighbour Classifier...")

knn = KNeighborsClassifier(n_neighbors=200)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
knnPickle = open('../models/knnpickle_file', 'wb')
# source, destination
pickle.dump(knn, knnPickle)

print("KNN Accuracy: %.3f%%" % (metrics.accuracy_score(Y_test, y_pred) * 100))
print("Training Complete")

print("Training using Random Forest Classifier...")

rf = RandomForestClassifier(max_depth=10, random_state=0)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
rfPickle = open('../models/rfpickle_file', 'wb')
pickle.dump(rf, rfPickle)

print("Random Forest Accuracy: %.3f%%" % (metrics.accuracy_score(Y_test, y_pred) * 100))
print("Training Complete")
