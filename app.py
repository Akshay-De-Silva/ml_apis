from flask import Flask, request, jsonify
from flask_cors import CORS
import ml

app = Flask(__name__)
app.config['DEBUG'] = False
CORS(app)

@app.route('/getLogReg', methods=['POST'])
def getLogReg():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getLogReg(weight_e, weight_p)

    return jsonify(results)

@app.route('/getLinReg', methods=['POST'])
def getLinReg():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getLinReg(weight_e, weight_p)

    return jsonify(results)

@app.route('/getKnn', methods=['POST'])
def getLinReg():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getLinReg(weight_e, weight_p)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)