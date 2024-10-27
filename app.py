from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# load the model we trained
model = joblib.load('model.joblib')

######## if you're using tensorflow ########
# in the file containing your model, save the model
# model.save('my_model.keras')
# in app.py:
# import tensorflow as tf
# new_model = tf.keras.models.load_model('my_model.keras')
# OR
# model.save('my_model.h5')
# new_model = tf.keras.models.load_model('my_model.h5')

######## if you're using pytorch ########
# in the file containing your model, save the model
# torch.save(model.state_dict(), PATH)
# in app.py:
# import torch
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, weights_only=True))
# model.eval()

######## if you're using HuggingFace ########
# from transformers import pipeline
# sentiment_analyzer = pipeline("sentiment-analysis")

# create a route for both get and post requests
@app.route('/', methods=['GET', 'POST'])
def predict():
    # initialize prediction variable
    prediction = None
    
    # check if the request is a post request (form submission)
    if request.method == 'POST':
        # receive the input for 'num_rooms' from the form and convert it to float
        num_rooms = float(request.form['num_rooms'])
        
        # make a prediction using the model and store the result
        prediction = model.predict([[num_rooms]])[0]
    
    # render the 'index.html' template and pass the prediction to it
    return render_template('index.html', prediction=prediction)

# run the app
if __name__ == '__main__':
    app.run()
