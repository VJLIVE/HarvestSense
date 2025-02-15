from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
import pandas as pd

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
df = pd.read_csv('yield_df.csv')

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    areas = sorted(df['Area'].dropna().unique().tolist())  # Ensure correct column name
    items = sorted(df['Item'].dropna().unique().tolist())  # Ensure correct column name

    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('predict.html', prediction=prediction[0][0], areas=areas, items=items)

    return render_template('predict.html', prediction=None, areas=areas, items=items)

# Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
