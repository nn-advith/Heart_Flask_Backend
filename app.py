from flask import Flask, jsonify, request
import simplejson as json
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Server running!"


@app.route("/predict_data", methods=['POST'])
def index():
    ip =json.loads(request.data)
    classifier = ip[-1]
    print(classifier)
    ip.pop(-1)
    print(ip)
    if classifier == 0:
        filename = ('./models/LR.sav') 
    elif classifier == 1:
        filename = ('./models/RFC.sav') 
    loaded_model = pickle.load(open(filename, 'rb'))
    res = loaded_model.predict([ip])

    return {"res": str(res[0])}


if __name__ == "__main__":
    app.run(debug=True)