# imports
import pickls
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

# initialize the flask app
app = Flask('myApp', template_folder = 'templates')

### route 1: hello world
# define the route
@app.route('/')
# create the controller
def home():
    return render_template("home.html")

### route 4: show a form to the user
@app.route("/form")
def form():
    # use flask's render_template function to display the html page
    return render_template("form.html")

@app.route("/submit")
def make_predictions():
    # load the form data from the incoming request
    user_input = request.args
    input_dict = {0:user_input['PotentialTweet']}
    df = pd.DataFrame(input_dict, index = input_dict.keys())
    def tok_and_lem(dataframe, new_column, input_column,handles = True, tok = True, lem = True):
        tokenizer = TweetTokenizer(strip_handles = handles)
        lemmatizer = WordNetLemmatizer()
        #tokenizing
        dataframe[new_column] = [tokenizer.tokenize(row) for row in dataframe[input_column]]
        #lemmatizing and joining the text back together 
        for row in dataframe.index:
            dataframe[new_column].iloc[row] = ' '.join([lemmatizer.lemmatize(word) for word in dataframe[new_column][row]])
        #example of what the data looks like
        return dataframe[new_column]
    tok_and_lem(df,'text_processed',0)
    model = pickle.load(open('./Models/n_bayes_2','rb')
    preds = model.predict(df['text_processed'])[0]
    return render_template("results.html", data = input_dict.values(), pred = preds)
    
# run the app
if __name__ == '__main__':
    app.run(debug = True)
