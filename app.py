import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('crop.pkl')

@app.route('/')
def home():
    title = 'Kheti.com - Home'
    return render_template('index.html',title=title)

@app.route('/form')
def form():
    title = 'Kheti.com - Crop'
    return render_template('form.html',title=title)

@app.route('/cropresult')
def cropresult():
    title = 'Kheti.com - Cropresult'
    return render_template('cropresult.html',title=title)

@app.route('/layout')
def layout():
    title = 'Kheti.com - layout'
    return render_template('layout.html',title=title)

@app.route('/predict', methods=['POST'])
def predict():
    title = 'Kheti Kaksha - Crop Recommendation'
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('cropresult.html', prediction=output, title=title )



if __name__ == "__main__":
    app.run(debug=True)
 