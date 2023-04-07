from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)

model = pickle.load(open('laptop.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for the prediction result
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the HTML form
    os = request.form['os']
    ram = (request.form['ram'])
    processor = request.form['processor']
    storage = (request.form['storage'])

    # Convert the input values into a pandas DataFrame
    query = np.array([processor, ram, os, storage])
    query = query.reshape(1,-1)
    p = model.predict(query)[0]
#print(p)
    p=np.exp(p)
    p=round(p)
    # Render the HTML template with the prediction result
    return render_template('home.html', predicted_price=p)

if __name__ == '__main__':
    app.run(debug=True)
