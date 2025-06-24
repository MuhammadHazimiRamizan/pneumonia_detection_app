**Deteksi Otomatis Gambar X-Ray Dada untuk Pneumonia Menggunakan CNN**

Proyek ini bertujuan untuk mengembangkan model klasifikasi citra medis menggunakan Convolutional Neural Networks (CNN) untuk mendeteksi pneumonia pada gambar X-ray dada secara otomatis.

**🧠 Deskripsi Proyek**

Model ini dilatih menggunakan dataset Chest X-Ray yang berisi citra dada dengan label Normal dan Pneumonia. Dengan augmentasi data dan teknik klasifikasi berbasis CNN, model dilatih untuk mempelajari fitur-fitur penting dari gambar.

**🛠️ Langkah-Langkah Training Model**

**1. Persiapan Lingkungan**

- Import library seperti TensorFlow, Keras, Matplotlib, dan Scikit-learn.
- Konfigurasi GPU.

**2. Load Dataset**

- Dataset diambil dari sumber online atau lokal (chest_xray).
- Dataset dibagi ke dalam folder train, val, dan test.

**3. Preprocessing & Augmentasi**

- Gunakan layer augmentasi: RandomFlip, RandomRotation, RandomZoom, dll.
- Rescaling piksel gambar ke rentang [0, 1].

**4. Arsitektur CNN**

- Layer convolutional: Conv2D → BatchNormalization → Activation → MaxPooling → Dropout.
- Fully connected layers: Flatten → Dense → Output Layer (Sigmoid/Softmax).

**5. Kompilasi Model**

- Optimizer: Adam
- Loss Function: binary_crossentropy
- Metrics: Accuracy

**6. Training**

- Menggunakan EarlyStopping dan ReduceLROnPlateau. 
- Perhitungan class_weight untuk mengatasi data imbalance.

**7. Evaluasi**

- Akurasi, confusion matrix, classification report.
- Visualisasi hasil prediksi pada gambar X-ray.

**8. Simpan Model**

- Model disimpan sebagai file .h5.
