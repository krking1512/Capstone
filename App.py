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

    model_nb = pickle.load(open('./Models/n_bayes_2', 'rb'))
    model_svc = pickle.load(open('./Models/svc_model','rb'))
# read a binary file
    preds_nb = f'{model_nb.predict(X_test)[0]:,}'
    if preds_nb == "1":
        pred_nb = 'sarcastic'
    elif preds_nb == '0':
        pred_nb = 'not Sarcastic'
    else:
        pred_nb = 'there was an error'
    preds_svc = f'{model_svc.predict(X_test)[0]:,}'
    if preds_svc == "1":
        pred_svc = 'sarcastic'
    elif preds_svc == '0':
        pred_svc = 'not Sarcastic'
    return render_template('results.html', prediction_nb = pred_nb, prediction_svc = pred_svc, data_processed= X_test[0], data = tweet)

# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if this file gets run from the terminal then run what's below
    app.run(debug=True)
