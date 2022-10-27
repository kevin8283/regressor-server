import operator
from flask import Blueprint, jsonify, request
from keras.models import load_model
import pandas as pd
import pickle

model = load_model("src/saved_models/regressor.h5")
vectorizer = pickle.load(open("src/saved_models/vectorizer.pickle", "rb"))

regressor = Blueprint(name="regressor", import_name=__name__)

@regressor.route('/predict', methods=['POST'])
def predict():

    # retrieve body data from input JSON
    body = request.get_json()
    inputs = body["data"]
    # compute result and return as JSON
    x_test = vectorizer.transform(
        pd.DataFrame([inputs],
        columns=["serie", "mlg", "frs", "eng", "hg", "phl", "math", "pc", "svt", "parcours"]
    )
    .to_dict(orient="records")
    ).toarray() 

    output = model.predict(x_test).flatten().tolist()[0]
    return jsonify(output)

@regressor.route('/recommend', methods=['POST'])

def recommend():
    fields = ["asecna", "droit", "egs", "ens", "espa", "flsh", "medecine", "misa"]
    
    body = request.get_json()
    inputs = body["data"]
    # compute result and return as JSON

    result = []

    for field in fields:
      x = vectorizer.transform(
        pd.DataFrame([[*inputs, field]],
        columns=["serie", "mlg", "frs", "eng", "hg", "phl", "math", "pc", "svt", "parcours"]
      )
      .to_dict(orient="records")
      ).toarray()

      y = model.predict(x).flatten().tolist()[0]
      result.append({"field": field, "score": y})
    
    result.sort(key=operator.itemgetter("score"))
    result.reverse()

    return jsonify(result)
