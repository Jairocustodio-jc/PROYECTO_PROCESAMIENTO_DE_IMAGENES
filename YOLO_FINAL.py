# importar los paquetes necesarios
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="ruta al video de entrada")
ap.add_argument("-o", "--output", required=True,
	help="ruta al video de salida")
ap.add_argument("-y", "--yolo", required=True,
	help="ruta base al directorio YOLO")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="probabilidad mínima para filtrar detecciones débiles")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="umbral al aplicar supresión de no máxima")
args = vars(ap.parse_args())

# cargar las etiquetas de clase COCO en las que se entrenó el modelo YOLO
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# inicializar una lista de colores para representar cada etiqueta de clase posible
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derivar las rutas a los pesos y la configuración del modelo YOLO
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# cargar el detector de objetos YOLO entrenado en el conjunto de datos COCO (80 clases)
# y determinar solo los nombres de las capas *de salida* que necesitamos de YOLO
print("[INFO] cargando YOLO desde el disco...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# inicializar el flujo de video, el puntero al archivo de video de salida y
# las dimensiones del fotograma
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# intentar determinar el número total de fotogramas en el archivo de video
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} fotogramas totales en el video".format(total))

# se produjo un error al intentar determinar el total
# número de fotogramas en el archivo de video
except:
	print("[INFO] no se pudo determinar el número de fotogramas en el video")
	print("[INFO] no se puede proporcionar un tiempo de finalización aproximado")
	total = -1

# bucle sobre los fotogramas del flujo de video del archivo
while True:
	# leer el siguiente fotograma del archivo
	(grabbed, frame) = vs.read()

	# si el fotograma no se capturó, entonces hemos llegado al final
	# del flujo
	if not grabbed:
		break

	# si las dimensiones del fotograma están vacías, captúrelas
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construir un blob a partir del fotograma de entrada y luego realizar un
	# paso hacia adelante del detector de objetos YOLO, dándonos nuestras cajas delimitadoras
	# y probabilidades asociadas
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# inicializar nuestras listas de cajas delimitadoras detectadas, confidencias
	# y IDs de clase, respectivamente
	boxes = []
	confidences = []
	classIDs = []

	# recorrer cada una de las salidas de capa
	for output in layerOutputs:
		# recorrer cada una de las detecciones
		for detection in output:
			# extraer la ID de clase y la confidencia (es decir, probabilidad)
			# de la detección actual del objeto
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filtrar las predicciones débiles asegurándose de que la detección
			# probabilidad sea mayor que la probabilidad mínima
			if confidence > args["confidence"]:
				# escalar las coordenadas de la caja delimitadora de nuevo en relación a
				# el tamaño de la imagen, teniendo en cuenta que YOLO
				# realmente devuelve el centro (coordenadas x, y) de
				# la caja delimitadora seguido del ancho y
				# altura de las cajas
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# usar las coordenadas del centro (x, y) para derivar la esquina superior
				# e izquierda de la caja delimitadora
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# actualizar nuestras listas de coordenadas de cajas delimitadoras,
				# confidencias e IDs de clase
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# aplicar supresión de no máxima para suprimir cajas delimitadoras débiles y superpuestas
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# asegurarse de que al menos una detección exista
	if len(idxs) > 0:
		# recorrer los índices que estamos manteniendo
		for i in idxs.flatten():
			# extraer las coordenadas de la caja delimitadora
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# dibujar un rectángulo de caja delimitadora y una etiqueta en el fotograma
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# comprobar si el escritor de video es None
	if writer is None:
		# inicializar nuestro escritor de video
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		# alguna información sobre el procesamiento de un solo fotograma
		if total > 0:
			elap = (end - start)
			print("[INFO] un solo fotograma tardó {:.4f} segundos".format(elap))
			print("[INFO] tiempo total estimado para terminar: {:.4f}".format(
				elap * total))

	cv2.imshow("Video", frame)
	if cv2.waitKey(1) == ord('q'):
		break

	# escribir el fotograma de salida en disco
	writer.write(frame)

# liberar los punteros de archivo
print("[INFO] limpiando...")
writer.release()
vs.release()
