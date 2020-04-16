import json
import pickle
import numpy as np
from collections import Counter
import pandas as pd
from sklearn import decomposition, preprocessing
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
    return pd.DataFrame(csv_values, columns=column_names)


def get_reduced_bbject(inputObject):
    column_to_drop = ["Frames#", "score_overall", "nose_score", "leftEye_score", "rightEye_score", "leftEar_score",
                      "rightEar_score", "leftShoulder_score", "rightShoulder_score", "leftElbow_score",
                      "rightElbow_score", "leftWrist_score", "rightWrist_score", "leftHip_score", "rightHip_score",
                      "leftKnee_score", "rightKnee_score",
                      "leftAnkle_score", "rightAnkle_score", "leftHip_x", "leftHip_y", "rightHip_x",
                      "rightHip_y", "leftKnee_x", "leftKnee_y", "rightKnee_x", "rightKnee_y", "leftAnkle_x",
                      "leftAnkle_y", "rightAnkle_x", "rightAnkle_y"]
    total_rows = 150
    inputObject.drop(column_to_drop, axis=1, inplace=True)
    print(inputObject.shape)
    result = pickle.load(open('./notebook/pickBeforePca', 'rb'))
    result = pd.DataFrame(result)
    while inputObject.shape[0] < total_rows:
        inputObject = inputObject.append(inputObject, ignore_index=True)
    shape = inputObject.shape
    if shape[0] > total_rows:
        inputObject = inputObject[:150]
    result = result.append(pd.DataFrame(inputObject.values), ignore_index=True)
    scaler = preprocessing.StandardScaler()
    scaler.fit(result)
    scaled_result = scaler.transform(result)
    pca = decomposition.PCA(n_components=25)
    pca.fit(scaled_result)
    pca_result = pca.transform(scaled_result)
    pca_result = pd.DataFrame(pca_result)
    pca_result = pca_result[-150:]
    return pca_result.values


@app.route('/getPrediction', methods=['POST'])
def get_prediction():
    csv_data = json_to_csv(request.json)
    label_to_category = {0: "buy", 1: "communicate", 2: "fun", 3: "hope", 4: "mother", 5: "really"}

    prediction_by_model = {}
    predicted_data = get_reduced_bbject(csv_data)
    print(predicted_data.shape)
    loaded_model = pickle.load(open('./models/model_1', 'rb'))
    result = loaded_model.predict(predicted_data)
    print(result)
    prediction_by_model["1"] = label_to_category[int(Counter(result).most_common(1)[0][0])]
    loaded_model = pickle.load(open('./models/model_2', 'rb'))
    result = loaded_model.predict(predicted_data)
    prediction_by_model["2"] = label_to_category[int(Counter(result).most_common(1)[0][0])]

    loaded_model = pickle.load(open('./models/model_3', 'rb'))
    result = loaded_model.predict(predicted_data)
    prediction_by_model["3"] = label_to_category[int(Counter(result).most_common(1)[0][0])]

    loaded_model = pickle.load(open('./models/model_4.pkl', 'rb'))
    result = loaded_model.predict(predicted_data)
    prediction_by_model["4"] = label_to_category[int(Counter(result).most_common(1)[0][0])]

    response = app.response_class(response=json.dumps(prediction_by_model),
                                  status=200,
                                  mimetype='application/json')
    return response


if __name__ == '__main__':
    app.run(debug=True, port=80)
