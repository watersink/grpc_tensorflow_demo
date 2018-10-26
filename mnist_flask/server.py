import flask
import os,sys
import io
from PIL import Image

sys.path.append("../train_test_mnist/")
from test_mnist import MNIST


mnist=MNIST()

app = flask.Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            predict_result,predict_probability = mnist.interface(image)
            data["predict_result"] = str(predict_result[0])
            data["predict_probability"] = str(predict_probability)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    print(data)
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading tensorflow model and Flask starting server..."
        "please wait until server has fully started"))
    app.run('0.0.0.0', 5000)