# region: Import dependencies
from flask import Flask, request, jsonify, send_file
from sklearn import preprocessing
import traceback, sys, os, joblib, pandas as pd, numpy as np

# endregion: Import dependencies

app = Flask(__name__)

# Function to preprocess the data for the decision tree model
def preprocess_data_for_dtree(query_df):
    return query_df.select_dtypes(include=[np.number])

@app.route('/predict/dectree', methods=['POST'])
def predict_dectree():
    if decision_tree_model:
        try:
            json_data = request.get_json()
            query_df = pd.DataFrame(json_data)
            query_processed = preprocess_data_for_dtree(query_df)
            prediction = decision_tree_model.predict(query_processed)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Decision tree model is not loaded'}), 500

@app.route('/predict/linregress', methods=['POST'])
def predict_linregress():
    print(request.data)
    if linear_model and linear_model_features:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)

            scaler = preprocessing.MinMaxScaler()
            query['BIKE_COST_NORMALIZED'] = scaler.fit_transform(query(['BIKE_COST']))
            query['BIKE_SPEED_NORMALIZED'] = scaler.fit_transform(query(['BIKE_SPEED']))

            data_encoded_ori = pd.get_dummies(query.drop(['EVENT_UNIQUE_ID', 'LOCATION_TYPE', 'OCC_TIMESTAMP',
                                                          'REPORT_TIMESTAMP', 'BIKE_COST', 'BIKE_SPEED'], axis=1),
                                              drop_first=True)
            data_encoded = data_encoded_ori.astype(float)
            data_encoded = data_encoded.reindex(columns=linear_model_features, fill_value=0)

            print(data_encoded)

            prediction = list(linear_model.predict(data_encoded))
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})


@app.route('/predict/logregress', methods=['POST'])
def predict_logregress():
    pass


@app.route('/description', methods=['GET'])
def data_description():
    pass  # TODO


@app.route('/statistics', methods=['GET'])
def data_statistics():
    pass  # TODO


@app.route('/visualization', methods=['GET', 'POST'])
def data_visualization():
    pass  # TODO


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # input from command-line
    except:
        port = 5001  # default port

    # Load models
    try:
        linear_model = joblib.load('./models/linear_model.pkl')
        linear_model_features = joblib.load('./models/linear_model_features.pkl')
        logistic_model = joblib.load('./models/model_logistic.pkl')
        logistic_model_features = joblib.load('./models/model_logistic_features.pkl')
        decision_tree_model = joblib.load('./models/decision_tree_model.pkl')
        print('Models loaded')
    except:
        print('Models could not be loaded')
        exit()

    app.run(port=port, debug=True)
