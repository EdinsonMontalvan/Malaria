import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import preprocessing
import time

# Cargar el modelo entrenado
MODEL_PATH = 'C:/Users/EDINSON/Desktop/Software/Malaria/models/modelo2_malaria.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((50, 50))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Función para hacer la predicción
def predict(image):
    processed_image = preprocess_image(image)
    start_time = time.time()  # Inicio del temporizador
    prediction = model.predict(processed_image)
    end_time = time.time()  # Fin del temporizador
    classes = ['Falciparum', 'Vivax', 'No infectado']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    execution_time = end_time - start_time  # Calcular el tiempo de ejecución
    return predicted_class, confidence, execution_time

# Interfaz de Streamlit
st.set_page_config(page_title="Clasificador de Malaria", layout="centered")
st.title("Clasificador de Células de Malaria 🦠")
st.write("Sube una imagen para clasificarla como *Falciparum*, *Vivax* o *No infectado*.")

# Aplicar CSS para mejorar el diseño
st.markdown(
    """
    <style>
    .stImage > img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border: 2px solid #d9d9d9;
        border-radius: 8px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        border-radius: 8px;
    }
    .stUploadButton>button {
        color: white;
        background-color: #2196F3;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        border: none;
    }
    .stMarkdown {
        font-size: 16px;
        font-weight: bold;
    }
        .result-card {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #333333;
    }
    .result-card h2 {
        margin: 0;
        font-size: 24px;
        color: #00796b;
    }
    </style>
    """, unsafe_allow_html=True
)

# Subir una imagen
uploaded_file = st.file_uploader("Elige una imagen", type=['jpg', 'jpeg', 'png', 'tiff'])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    
    # Centrando la imagen y ajustando el tamaño
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='' style='width: 300px; height: auto; border-radius: 50px;'/>
        </div>
        """.format(st.image(image, caption='Imagen cargada', use_column_width=False, width=300)),
        unsafe_allow_html=True
    )

    # Botón para clasificar
    if st.button("Clasificar"):
        with st.spinner('Clasificando la imagen...'):
            predicted_class, confidence, execution_time = predict(image)
            st.success('Clasificación completada')

    # Mostrar resultados finales
            st.markdown(f"""
                <div class="result-card">
                    <h2>Resultado Final: {predicted_class}</h2>
                    <p>Tiempo Total de Ejecución: {execution_time:.2f} segundos</p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.write("Por favor, sube una imagen para comenzar la clasificación.")

