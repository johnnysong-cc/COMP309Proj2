import joblib
import numpy as np
import pandas as pd
import sys
import traceback

from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing

from plotting.create_plot import getstats, create_plot

app = Flask(__name__)

df = pd.read_csv('./data/Bicycle_Thefts_Open_Data.csv')


def preprocess_data_for_dtree(query):
    # Convert to numeric and drop non-numeric columns
    query_numeric = query.select_dtypes(include=[np.number])
    return query_numeric


@app.route('/predict/dectree', methods=['POST'])
def predict_dectree():
    if decision_tree_model:
        try:
            json_ = request.json
            query = pd.DataFrame(json_)

            # Preprocess the data
            query_processed = preprocess_data_for_dtree(query)

            # Predict using the decision tree model
            prediction = decision_tree_model.predict(query_processed)
            return jsonify({'prediction': prediction.tolist()})
        except:
            return jsonify({'trace': traceback.format_exc()})


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
    print(request.data)
    if logistic_model and logistic_model_features:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)

            scaler = preprocessing.MinMaxScaler()
            query['BIKE_COST_NORMALIZED'] = scaler.fit_transform(query(['BIKE_COST']))
            query['BIKE_SPEED_NORMALIZED'] = scaler.fit_transform(query(['BIKE_SPEED']))

            data_encoded_ori = pd.get_dummies(query.drop(
                ['EVENT_UNIQUE_ID', 'LOCATION_TYPE', 'OCC_TIMESTAMP',
                 'REPORT_TIMESTAMP', 'BIKE_COST', 'BIKE_SPEED'], axis=1),
                drop_first=True)

            data_encoded = data_encoded_ori.astype(float)
            data_encoded = data_encoded.reindex(columns=logistic_model_features, fill_value=0)

            print(data_encoded)

            prediction = list(logistic_model.predict(data_encoded))
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})


@app.route('/description', methods=['GET'])
def data_description():
    print(df.describe())
    return f"{df.describe()}"


@app.route('/statistics', methods=['GET'])
def data_statistics():
    return jsonify(getstats(df))


@app.route('/visualization', methods=['GET'])
def data_visualization():
    stats = getstats(df)

    image_paths = []
    for key in stats:
        if stats[key]:
            image_path = create_plot(stats[key], key.capitalize())
            image_paths.append(image_path)
        else:
            print(f"Data for {key} is empty or None.")

    html_content = '<h1>Data Visualizations</h1>'
    for key in stats:
        image_b64 = create_plot(stats[key], key.capitalize())
        html_content += f'<img src="{image_b64}" alt="{key.capitalize()}"><br>'

    return html_content


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html')


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
