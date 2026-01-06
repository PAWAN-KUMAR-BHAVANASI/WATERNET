from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_water_quality

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    values = [
        data["ph"],
        data["turbidity"],
        data["tds"],
        data["hardness"],
        data["chloride"],
        data["nitrate"]
    ]

    drink, irri = predict_water_quality(values)

    return jsonify({
        "drinking_water": "Safe" if drink == 1 else "Unsafe",
        "irrigation_water": "Suitable" if irri == 1 else "Unsuitable"
    })

if __name__ == "__main__":
    app.run(debug=True)
