from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = pickle.load(open('model.pkl', 'rb'))
        
        # Get values through input bars
        slength = request.form.get("slength")
        swidth = request.form.get("swidth")
        plength = request.form.get("plength")
        pwidth = request.form.get("pwidth")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[slength, swidth, plength, pwidth]], columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

if __name__ == '__main__':
     app.run(port=8080)