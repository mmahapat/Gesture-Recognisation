## Gesture-Recognisation
* CSV of the training data should be present in appropriate folder in the base directory
* Run notebook/Training.ipynb file to train and create models
* Models are created in the models folder

##### After models are trained and Created
##Run the flask server
* Api contract is `POST` `localhost:7777/getPrediction`
* Expects json body like in `hello.json` file present.
* Running server `python app.py` 
