from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image  # Tambahan untuk validasi grayscale

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Ukuran input model
img_size = (150, 150)

# Load model saat server dijalankan
model = load_model('./model/pneumonia_detection_optimized_model.h5')

# Mapping label
class_names = ['NORMAL', 'PNEUMONIA']

def is_grayscale(pil_image):
    """Cek apakah gambar grayscale (indikasi X-ray)."""
    grayscale_img = pil_image.convert("L")
    rgb_img = pil_image.convert("RGB")
    diff = np.sum(np.array(rgb_img) - np.array(grayscale_img.convert("RGB")))
    return diff < 1000  # Nilai threshold ini bisa disesuaikan

@app.route('/', methods=['GET', 'POST'])
def index():
    print('API Berjalan')
    prediction = None
    if request.method == 'POST':
        # Ambil file dari form upload
        img_file = request.files.get('image')
        if img_file:
            # Amankan nama file
            filename = secure_filename(img_file.filename)

            # Simpan file di folder static/
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(img_path)

            # Buka gambar untuk validasi grayscale
            pil_image = Image.open(img_path)
            if not is_grayscale(pil_image):
                prediction = {
                    'label': 'UNKNOWN',
                    'confidence': "0.00%",
                    'image_path': img_path
                }
                return render_template('index.html', prediction=prediction)

            # Proses gambar untuk prediksi
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prob = model.predict(img_array)[0][0]
            label = class_names[1] if prob > 0.5 else class_names[0]
            confidence = prob if prob > 0.5 else 1 - prob

            # Threshold keyakinan
            threshold = 0.60
            if confidence < threshold:
                label = "UNKNOWN"

            prediction = {
                'label': label,
                'confidence': f"{confidence * 100:.2f}%",
                'image_path': img_path
            }

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
