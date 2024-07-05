import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

# Memuat data dari Excel
data = pd.read_excel('Data_Rumah.xlsx')

# Mengatur fitur dan target
X = data[['luas_tanah', 'jumlah_kamar', 'usia_bangunan', 'lokasi_numerik']]
y = data['harga']

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Membuat model SVM
model = SVR(kernel='linear')
model.fit(X_train_scaled, y_train)

# Menyimpan model dan scaler ke dalam file
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    pickle.dump(scaler, file)

from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle  # Used to load the model

app = Flask(__name__)

# Load the pre-trained SVM model and scaler
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    scaler = pickle.load(model_file)  # Load scaler

@app.route('/')
def index():
    return render_template('index.html')  # Assumed to be in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    # Receive data from the form
    data = {
        'luas_tanah': request.form.get('landSize', type=float),
        'jumlah_kamar': request.form.get('bedrooms', type=float),
        'usia_bangunan': request.form.get('buildingAge', type=float),
        'lokasi_numerik': request.form.get('location', type=float)
    }
    df = pd.DataFrame([data])
    features = scaler.transform(df)

    # Make prediction
    prediction = model.predict(features)

    # Return prediction price
    return jsonify({'prediksi_harga': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
