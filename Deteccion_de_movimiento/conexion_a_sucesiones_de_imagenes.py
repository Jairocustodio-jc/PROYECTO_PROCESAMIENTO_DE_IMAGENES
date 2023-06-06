# Importación de librerías
import numpy as np
import cv2

# Capturamos el vídeo
cap = cv2.VideoCapture('prueba_movimiento_estatico.mp4')

# Llamada al método
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)

# Deshabilitamos OpenCL, si no hacemos esto no funciona
cv2.ocl.setUseOpenCL(False)

while(1):
	# Leemos el siguiente frame
	ret, frame = cap.read()

	# Si hemos llegado al final del vídeo salimos
	if not ret:
		break

	# Aplicamos el algoritmo
	fgmask = fgbg.apply(frame)

	# Copiamos el umbral para detectar los contornos
	contornosimg = fgmask.copy()

	# Buscamos contorno en la imagen
	im, contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# Recorremos todos los contornos encontrados
	for c in contornos:
		# Eliminamos los contornos más pequeños
		if cv2.contourArea(c) < 500:
			continue

		# Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
		(x, y, w, h) = cv2.boundingRect(c)
		# Dibujamos el rectángulo del bounds
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Mostramos las capturas
	cv2.imshow('Camara',frame)
	cv2.imshow('Umbral',fgmask)
	cv2.imshow('Contornos',contornosimg)

	# Sentencias para salir, pulsa 's' y sale
	k = cv2.waitKey(30) & 0xff
	if k == ord("s"):
		break

# Liberamos la cámara y cerramos todas las ventanas
cap.release()
cv2.destroyAllWindows()