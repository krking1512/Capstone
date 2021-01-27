# imports
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
import text_processing
# initialize the flask app
app = Flask('my_app')

@app.route('/')# Flask will run the function directly below this decorator when you access this route '/'


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/submit')
def submit():
    user_input = request.args
    tweet = str(user_input['PotentialTweet'])

    X_test = text_processing.tok_and_lem_input(tweet)

    model = pickle.load(open('./Models/n_bayes_2', 'rb'))
# read a binary file
    preds = f'{round(model.predict(X_test)[0], 2):,}'
    if preds == "1":
        pred = 'Sarcastic'
    elif preds == '0':
        pred = 'Not Sarcastic'
    else:
        pred = 'there was an error'
    return render_template('results.html', prediction = pred, data_processed= X_test[0], data = tweet)

# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if this file gets run from the terminal then run what's below
    app.run(debug=True)
