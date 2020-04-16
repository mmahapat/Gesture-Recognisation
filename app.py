import json
import pickle
import numpy as np
from collections import Counter
import pandas as pd
from sklearn import decomposition
from flask import Flask, jsonify, request

app = Flask(__name__)


def json_to_csv(json_data):
    column_names = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
                    'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
                    'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x',
                    'leftShoulder_y',
                    'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
                    'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
                    'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
                    'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x',
                    'leftKnee_y',
                    'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
                    'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    rows = len(json_data)
    columns = len(column_names)
    csv_values = np.zeros((rows, columns))
    for i in range(rows):
        individual_element_data = [json_data[i]['score']]
        for individual_element in json_data[i]['keypoints']:
            individual_element_data.append(individual_element['score'])
            individual_element_data.append(individual_element['position']['x'])
            individual_element_data.append(individual_element['position']['y'])
        csv_values[i] = np.array(individual_element_data)
    pd.DataFrame(csv_values, columns=column_names)
    return csv_values


def get_reduced_bbject(inputObject):
    total_rows = 150
    result = pickle.load(open('/notebook/pickBeforePca', 'rb'))
    result = pd.DataFrame(result)
    while inputObject.shape[0] < total_rows:
        inputObject = inputObject.append(inputObject, ignore_index=True)
    shape = inputObject.shape
    if shape[0] > total_rows:
        inputObject = inputObject[:150]

    print(result.shape)
    result = result.append(inputObject)
    print(result.shape)
    scaler = preprocessing.StandardScaler()
    scaler.fit(result)
    scaled_result = scaler.transform(result)
    pca = decomposition.PCA(n_components=25)
    pca.fit(scaled_result)
    pca_result = pca.transform(scaled_result)
    pca_result = pd.DataFrame(pca_result)
    print(pca_result.tail(150))
    pca_result = pca_result[-150:]
    return pca_result.values


@app.route('/getPrediction', methods=['POST'])
def get_prediction():
    csv_data = json_to_csv(request.json)
    label_to_category = {0: "buy", 1: "communicate", 2: "fun", 3: "hope", 4: "mother", 5: "really"}

    prediction_by_model = {}
    predicted_data = get_reduced_bbject(csv_data)
    loaded_model = pickle.load(open('logistic_regression_file', 'rb'))
    result = loaded_model.predict(predicted_data)
    prediction_by_model["1"] = label_to_category[int(Counter(result).most_common(1)[0][0])]
    print(prediction_by_model)
    print(result)
    # loaded_model = pickle.load(open('logistic_regression_file', 'rb'))
    # result = loaded_model.predict(csv_data)
    # prediction_by_model["2"] = label_to_category[int(Counter(result).most_common(1)[0][0])]
    #
    # loaded_model = pickle.load(open('knnpickle_file', 'rb'))
    # result = loaded_model.predict(csv_data)
    # prediction_by_model["3"] = label_to_category[int(Counter(result).most_common(1)[0][0])]
    #
    # loaded_model = pickle.load(open('logistic_regression_file', 'rb'))
    # result = loaded_model.predict(csv_data)
    # prediction_by_model["4"] = label_to_category[int(Counter(result).most_common(1)[0][0])]

    return jsonify(prediction_by_model)


if __name__ == '__main__':
    app.run(debug=True, port=80)
