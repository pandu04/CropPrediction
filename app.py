from flask import Flask, request, render_template
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

# Load the model and preprocessor
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = str.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction[0][0])

if __name__ == "__main__":
    app.run(debug=True)