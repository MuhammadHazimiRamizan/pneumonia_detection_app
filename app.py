import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os # Although not strictly needed for Gradio's image handling, good practice if local paths were involved

# Suppress TensorFlow logging messages if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ukuran input model
img_size = (150, 150)

# Path to the model
# In Hugging Face Spaces, your model should be in the same directory or a subdirectory.
# Ensure 'model/pneumonia_detection_optimized_model.h5' exists relative to app.py
model_path = './pneumonia_detection_optimized_model.h5'

# Load model saat server dijalankan
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure './pneumonia_detection_optimized_model.h5' exists.")
    # Exit or handle gracefully, as the app won't function without the model
    exit()

# Mapping label
class_names = ['NORMAL', 'PNEUMONIA']

def is_grayscale(pil_image):
    """
    Cek apakah gambar grayscale (indikasi X-ray).
    Membandingkan perbedaan antara gambar asli (dikembalikan ke RGB) dan versi grayscale-nya.
    Jika perbedaannya kecil, berarti gambar tersebut cenderung grayscale.
    """
    # Convert image to grayscale (L mode) and then back to RGB for comparison
    grayscale_img = pil_image.convert("L").convert("RGB")
    rgb_img = pil_image.convert("RGB") # Ensure the input is also RGB for consistent comparison

    # Calculate the sum of absolute differences between the RGB and grayscale versions
    # A small difference indicates it was already very close to grayscale.
    diff = np.sum(np.abs(np.array(rgb_img) - np.array(grayscale_img)))
    
    # This threshold might need fine-tuning based on your dataset characteristics.
    # A value of 1000 is a heuristic; images with color will have much higher differences.
    return diff < 1000

def predict_pneumonia(input_image: Image.Image):
    """
    Fungsi prediksi untuk model pneumonia.
    Menerima gambar PIL.Image, melakukan preprocessing, validasi grayscale,
    dan mengembalikan label, tingkat keyakinan, serta karakteristik radiologis.
    """
    # Initialize diagnostic_text to empty
    diagnostic_text = ""

    if input_image is None:
        return "No Image Uploaded", "N/A", "Silakan unggah gambar."

    # Validate if the image is grayscale (proxy for X-ray)
    if not is_grayscale(input_image):
        label = "UNKNOWN"
        confidence_str = "0.00%"
        diagnostic_text = """
        <div class="section-info mt-4">
            <h6 style="color: #6c757d;">Gambar Tidak Dikenali sebagai X-ray Dada:</h6>
            <p>Silakan unggah ulang dengan gambar X-ray dada posterior-anterior (PA) yang valid.</p>
        </div>
        """
        return "UNKNOWN (Not Grayscale/X-ray)", "0.00%", diagnostic_text

    # Preprocess the image for the model
    img = input_image.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # If your model was trained with normalized inputs (e.g., /255.0), apply it here:
    # img_array = img_array / 255.0

    # Make prediction
    prob = model.predict(img_array)[0][0]

    # Determine label and confidence
    if prob > 0.5:
        label = class_names[1]  # PNEUMONIA
        confidence = prob
    else:
        label = class_names[0]  # NORMAL
        confidence = 1 - prob   # Confidence for the NORMAL class

    # Apply confidence threshold
    threshold = 0.60
    if confidence < threshold:
        label = "UNKNOWN"
        diagnostic_text = """
        <div class="section-info mt-4">
            <h6 style="color: #6c757d;">Gambar Tidak Dikenali / Keyakinan Rendah:</h6>
            <p>Silakan unggah ulang dengan gambar X-ray dada posterior-anterior (PA) yang valid.</p>
        </div>
        """
    elif label == 'PNEUMONIA':
        diagnostic_text = """
        <div class="section-info mt-4">
            <h6 style="color: #dc3545;">Karakteristik Radiologis Pneumonia:</h6>
            <ul>
                <li>Infiltrat atau konsolidasi opasitas putih di jaringan paru</li>
                <li>Pola lobar atau bronkopneumonia</li>
                <li>Bisa disertai volume loss dan air bronchogram</li>
            </ul>
        </div>
        """
    elif label == 'NORMAL':
        diagnostic_text = """
        <div class="section-info mt-4">
            <h6 style="color: #28a745;">Karakteristik Radiologis Paru Normal:</h6>
            <ul>
                <li>Lapang paru jernih, tidak terdapat infiltrat</li>
                <li>Kontur jantung dan diafragma normal</li>
                <li>Distribusi vaskular merata</li>
            </ul>
        </div>
        """

    # Format confidence as a percentage string
    confidence_str = f"{confidence * 100:.2f}%"

    return label, confidence_str, diagnostic_text

# Set up the Gradio Interface
iface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil", label="Unggah Gambar X-ray"),
    outputs=[
        gr.Textbox(label="Prediksi"),
        gr.Textbox(label="Keyakinan"),
        gr.Markdown(label="Karakteristik Radiologis") # New output for the diagnostic text
    ],
    title="Deteksi Pneumonia dari Gambar X-ray",
    description="Unggah gambar X-ray dada skala abu-abu untuk mendapatkan prediksi Pneumonia. Gambar yang tidak terdeteksi sebagai skala abu-abu akan ditolak.",
    examples=[
        # You can add example image paths here for easier testing in Hugging Face Spaces
        # e.g., ["path/to/normal_xray.jpeg"], ["path/to/pneumonia_xray.jpeg"]
        # Ensure these paths exist in your Hugging Face Space repository.
    ],
    allow_flagging="auto", # Allows users to "flag" inputs/outputs for review
    theme=gr.themes.Soft() # A modern, soft theme
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()