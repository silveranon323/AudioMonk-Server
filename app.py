from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import io
import pickle
import os
from predictor.Metadata import getmetadata
from sklearn.preprocessing import MinMaxScaler

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
genre_type = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}
scaler = MinMaxScaler()
app = Flask(__name__)
CORS(app)
with open("models.p", "rb") as f:
    models = pickle.load(f)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file part received")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(file_path)

    a = getmetadata(file_path)
    d1 = np.array(a)
    data1 = models["norma"].transform([d1])
    predicted_genre = models["svm"].predict(data1)

    try:

        return jsonify(
            {
                "message": "Audio file received successfully",
                "filename": file.filename,
                "duration": 0.2,
                "predicted_genre": genre_type[predicted_genre[0]],
            }
        )

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500


if __name__ == "__main__":
    app.run()
