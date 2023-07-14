

def calcular_flujo_optico(frame_anterior, frame_actual, tamano_ventana=15):
    # Convertir los frames a escala de grises
    anterior_gris = np.mean(frame_anterior, axis=2)
    actual_gris = np.mean(frame_actual, axis=2)

    # Calcular el gradiente en las direcciones x e y
    gradiente_x = np.gradient(anterior_gris)
    gradiente_y = np.gradient(anterior_gris)

    # Definir la ventana para el cálculo del flujo óptico
    mitad_ventana = tamano_ventana // 2

    # Inicializar matrices vacías para el flujo óptico en las direcciones x e y
    flujo_x = np.zeros(anterior_gris.shape)
    flujo_y = np.zeros(anterior_gris.shape)

    # Iterar sobre cada píxel de la imagen
    for i in range(mitad_ventana, anterior_gris.shape[0] - mitad_ventana):
        for j in range(mitad_ventana, anterior_gris.shape[1] - mitad_ventana):
            # Calcular la región de interés para el píxel
            roi_anterior = anterior_gris[i - mitad_ventana: i + mitad_ventana + 1, j - mitad_ventana: j + mitad_ventana + 1]
            gradiente_x_roi = gradiente_x[i - mitad_ventana: i + mitad_ventana + 1, j - mitad_ventana: j + mitad_ventana + 1]
            gradiente_y_roi = gradiente_y[i - mitad_ventana: i + mitad_ventana + 1, j - mitad_ventana: j + mitad_ventana + 1]

            # Aplanar las matrices para realizar operaciones de matriz
            roi_anterior = roi_anterior.flatten()
            gradiente_x_roi = gradiente_x_roi.flatten()
            gradiente_y_roi = gradiente_y_roi.flatten()

            # Construir la matriz A y el vector b para el sistema de ecuaciones lineales
            A = np.vstack((gradiente_x_roi, gradiente_y_roi)).T
            b = -roi_anterior-

            # Resolver el sistema de ecuaciones lineales
            v = np.linalg.lstsq(A, b, rcond=None)[0]

            # Actualizar el flujo óptico para el píxel
            flujo_x[i, j] = v[0]
            flujo_y[i, j] = v[1]

    return flujo_x, flujo_y

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf

# Construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="ruta al video de entrada")
ap.add_argument("-o", "--output", required=True,
    help="ruta al video de salida")
ap.add_argument("-m", "--model", required=True,
    help="ruta al modelo entrenado")
ap.add_argument("-l", "--labels", required=True,
    help="ruta al archivo de etiquetas")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="probabilidad mínima para filtrar detecciones débiles")
args = vars(ap.parse_args())

# Cargar el modelo entrenado y las etiquetas
model = tf.keras.models.load_model(args["model"])
with open(args["labels"], "r") as f:
    LABELS = f.read().splitlines()

# Cargar el video
video = cv2.VideoCapture(args["input"])

# Obtener propiedades del video
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Inicializar el escritor de video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(args["output"], fourcc, fps, (width, height), True)

# Definir el tamaño de la imagen para el redimensionamiento
TAMANO_IMG = 100

# Procesar cada cuadro del video
while True:
    # Leer el siguiente cuadro
    ret, frame = video.read()

    # Romper el bucle si se llega al final del video
    if not ret:
        break

    # Convertir el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocesar el cuadro
    preprocessed_frame = cv2.resize(gray, (TAMANO_IMG, TAMANO_IMG))
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=-1) / 255.0

    # Realizar predicciones en el cuadro
    predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    class_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    # Filtrar predicciones basadas en el umbral de confianza
    mask = confidences > args["confidence"]
    class_indices = class_indices[mask]
    confidences = confidences[mask]

    # Escalar las coordenadas del cuadro delimitador al tamaño del cuadro
    boxes = np.array([(0, 0, 0, 0)])  # Cuadros ficticios, actualiza esto con tus cuadros delimitadores reales

    # Aplicar supresión de no máxima para seleccionar las detecciones más relevantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["confidence"])

    # Dibujar cuadros delimitadores y etiquetas
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = LABELS[class_indices[i]]
        color = (0, 255, 0)  # Color verde para los cuadros delimitadores
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidences[i]:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Escribir el cuadro en el video de salida
    writer.write(frame)

    # Mostrar el cuadro
    cv2.imshow("Cuadro", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos del video
video.release()
writer.release()
cv2.destroyAllWindows()
