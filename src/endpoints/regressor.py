from flask import Blueprint, jsonify, request
from keras.models import load_model
from operator import itemgetter
import numpy as np

orientation_model = load_model("src/saved_models/orientation.h5")
deliberation_model = load_model("src/saved_models/deliberation2.h5")

regressor = Blueprint(name="regressor", import_name=__name__)

@regressor.route('/deliberation/predict', methods=['POST'])
def predict():

    body = request.get_json()
    inputs = body["data"]
    
    x = [
          inputs["gr_etu"], inputs["gr_ens"],
          inputs["gr_pat"], inputs["cr_pol"],
          inputs["cr_san"], inputs["cr_mto"],
          inputs["heure_etude_hebd"]
        ]

    result = deliberation_model.predict([x], batch_size=400).flatten()[0]
  
    return jsonify({"result": float(result)})

@regressor.route('/orientation/recommend', methods=['POST'])

def recommend():
    fields = ["mto", "gpci", "gmi", "en", "tco", "btp", "geol", "petr", "ge", "igat"]
    
    body = request.get_json()
    inputs = body["data"]
    # compute result and return as JSON

    grades = [
          inputs["mech"], inputs["chem"],
          inputs["math"], inputs["elec"],
          inputs["sc_terre"], inputs["trigo"],
          inputs["alg_bool"], inputs["opt"],
          inputs["dess_tech"], inputs["env"]
        ]
    
    x1 = np.asarray(fields)
    x2 = np.asarray([grades for _, g in enumerate(fields)])

    results = orientation_model.predict([x1, x2], batch_size=400).flatten()
    
    results = [{"field": fields[i], "score": float(results[i])} for i in range(0, 10)]

    
  

    results.sort(key=itemgetter("score"))
    results.reverse()

    return jsonify(results)
