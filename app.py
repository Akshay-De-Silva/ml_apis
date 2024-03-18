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
def getKnn():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getKnn(weight_e, weight_p)

    return jsonify(results)

@app.route('/getDt', methods=['POST'])
def getDt():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getDt(weight_e, weight_p)

    return jsonify(results)

@app.route('/getRf', methods=['POST'])
def getRf():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getRf(weight_e, weight_p)

    return jsonify(results)

@app.route('/getNb', methods=['POST'])
def getNb():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getNb(weight_e, weight_p)

    return jsonify(results)

@app.route('/getSvm', methods=['POST'])
def getSvm():
    data = request.get_json()
    weight_e = data.get('weight_e')
    weight_p = data.get('weight_p')

    results = ml.getSvm(weight_e, weight_p)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)