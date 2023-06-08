import os
import tensorflow as tf
import numpy as np

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename #untuk menamai file jika ada yang pake spasi
from tensorflow import keras


app = Flask(__name__)

# memastikan file yang masuk adalah PNG, JPEG, atau JPG
app.config["ALLOWED_EXTENSIONS"] = set(["png", "jpg", "jpeg"])
app.config["UPLOAD_FOLDER"] = 'static/uploads/'

#load model h5
model = keras.models.load_model("model.h5", compile=False)


#mengecek apakah file yang masuk adalah  PNG, JPEG, atau JPG, kalau bukan, file ditolak
def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1] in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    return jsonify({
        "status": {
            "code" : 200,
            "message" : "Succes fetching the API!"
        }
    }), 200

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    #image harus POST, jika tidak maka NOT ALLOWED
    if request.method == "POST":
        image = request.files["image"]

        #jika ada foto PNG/JPG/JPEG akan diproses oleh model, jika tidak ada akan ditulak
        if image and allowed_file(image.filename):
            image.save(os.path.join( #save imagenya
                app.config["UPLOAD_FOLDER"], secure_filename(image.filename)))
            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"], secure_filename(image.filename))
            
            img = keras.utils.load_img(image_path, target_size = (300,300))
            x = keras.utils.img_to_array(img)
            x /= 255.
            x = np.expand_dims(x, axis= 0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size = 20)
            print(classes[0])

            result =""
            if classes[0] > 0.5:
                result = "Normal"
            else:
                result = "Pothole"
            a = classes[0].tolist()
            return jsonify({
                "status": {
                    "code"  : 200,
                    "message": "Succses predicting the image!"
                },
                "data": {
                    "image" : "http://localhost:5000/" + image_path,
                    "result": "Normal accuracy= " + str(a) + " status: " + result
                }
            })
        else:
            return jsonify({
                "status": {
                    "code"  : 400,
                    "message": "Please input PNG/JPG/JPEG!"
                }
            })
    else:
        return jsonify({
            "status": {
                "code" : 403,
                "message" : "Method not allowed!"
            }
        }), 403


if __name__ == "__main__":
    app.run()
    
    