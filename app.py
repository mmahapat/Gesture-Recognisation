from flask import Flask

app = Flask(__name__)


@app.route('/getPrediction', methods=['POST', 'GET'])
def get_prediction():
    return {"1": "Test", "2": "Test", "3": "Test", "4": "Test"}


if __name__ == '__main__':
    app.run()
