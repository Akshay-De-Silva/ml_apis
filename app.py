from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/getLogReg', methods=['POST'])
def getLogReg():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    response = {
        "BATI": weight_e,
        "RAF1_CPU" : weight_e,
        "RAF1_GPU" : weight_e,
        "RAF1_RAM" : weight_e,
        "F1CI" : weight_e,
        "BRMSETI" : 0,
        "RARS_CPU" : 0,
        "RARS_GPU" : 0,
        "RARS_RAM" : 0,
        "RMSPECI" : 0,
        "Accuracy" : weight_p,
        "F1Score" : weight_p,
        "Precision" : weight_p,
        "Recall" : weight_p,
        "MSE" : 0,
        "RSME" : 0,
        "MAE" : 0,
        "R2" : 0
    }

    return jsonify(response)

@app.route('/getLinReg', methods=['POST'])
def getLinReg():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    response = {
        "BATI": 0,
        "RAF1_CPU" : 0,
        "RAF1_GPU" : 0,
        "RAF1_RAM" : 0,
        "F1CI" : 0,
        "BRMSETI" : weight_e,
        "RARS_CPU" : weight_e,
        "RARS_GPU" : weight_e,
        "RARS_RAM" : weight_e,
        "RMSPECI" : weight_e,
        "Accuracy" : 0,
        "F1Score" : 0,
        "Precision" : 0,
        "Recall" : 0,
        "MSE" : weight_p,
        "RSME" : weight_p,
        "MAE" : weight_p,
        "R2" : weight_p
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)