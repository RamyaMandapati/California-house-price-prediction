
from flask import Flask, render_template, url_for, flash, redirect

from flask import request
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():

    model = joblib.load('California_Model.pkl')
    float_ = [float(x) for x in request.form.values()]
    final_ = [np.array(float_)]
    prediction = model.predict(final_)
    output = round(prediction[0], 2)
    pred = 'affording home in California is  $ {}'.format(output)

    return(render_template("result.html", prediction_text=pred))


if __name__ == "__main__":
    # Use below for AWS EC2 deployment
    app.run(host='0.0.0.0',port=8080)
