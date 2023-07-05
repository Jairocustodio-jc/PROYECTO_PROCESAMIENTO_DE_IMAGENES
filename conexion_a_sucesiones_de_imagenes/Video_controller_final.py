import cv2

from cv2_tools.Management import ManagerCV2

class VideoController():
    @staticmethod
    def CONVERTIR_IMAGEN(IMAGEN):
        # Convertir a grises
        IMAGEN_GRIS=cv2.cvtColor(IMAGEN,cv2.COLOR_BGR2GRAY)
        # SUPRESION DE RUIDO
        return cv2.GaussianBlur(IMAGEN_GRIS,(21,21),0)
    @staticmethod
    def CONVERTIR_GRIS_IMAGEN(IMAGEN):
         return cv2.cvtColor(IMAGEN,cv2.COLOR_BGR2GRAY)
    @staticmethod
    def ENCONTRAR_CONTORNOS(IMAGEN, AREA_MINIMA=500):
        IMG_CONTORNOS = IMAGEN.copy()
        CONTORNOS, JERARQUIA = cv2.findContours(IMG_CONTORNOS, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        CONTORNOS_VALIDOS = []

        for i, CONTORNO in enumerate(CONTORNOS):
            # Puedes verificar si un contorno con índice i está dentro de otro comprobando si JERARQUIA[0, i, 3] es igual a -1 o no.
            # Si es diferente de -1, significa que el contorno está dentro de otro y queremos ignorarlo.
            if JERARQUIA[0, i, 3] != -1:
                continue

            # Ignorar contornos pequeños
            if cv2.contourArea(CONTORNO) < AREA_MINIMA:
                continue

            # Rectángulo del contorno
            X, Y, ANCHO, ALTO = cv2.boundingRect(CONTORNO)
            CONTORNOS_VALIDOS.append((X, Y, X + ANCHO, Y + ALTO))

        return CONTORNOS_VALIDOS
    @staticmethod
    def OBTENER_CONTORNOS_Y_CENTRO(IMAGEN, AREA_MINIMA=500):
        IMG_CONTORNOS = IMAGEN.copy()
        CONTORNOS, JERARQUIA = cv2.findContours(IMG_CONTORNOS, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        CONTORNOS_VALIDOS = []

        for i, CONTORNO in enumerate(CONTORNOS):
            # Puedes verificar si un contorno con índice i está dentro de otro comprobando si JERARQUIA[0, i, 3] es igual a -1 o no.
            # Si es diferente de -1, significa que el contorno está dentro de otro y queremos ignorarlo.
            if JERARQUIA[0, i, 3] != -1:
                continue

            # Ignorar contornos pequeños
            if cv2.contourArea(CONTORNO) < AREA_MINIMA:
                continue

            # Rectángulo del contorno
            X, Y, ANCHO, ALTO = cv2.boundingRect(CONTORNO)

            momentos = cv2.moments(CONTORNO)
            CENTRO_X = int(momentos["m10"] / momentos["m00"])
            CENTRO_Y = int(momentos["m01"] / momentos["m00"])

            CONTORNOS_VALIDOS.append((X, Y, X + ANCHO, Y + ALTO, (CENTRO_X, CENTRO_Y)))

        return CONTORNOS_VALIDOS
    def __init__(self, VIDEO, STREAM, FPS, DILATAR=False, DETECTAR_ESCENAS=False, NOMBRE='DetectorMovimiento'):
        self.DILATAR = DILATAR
        self.NOMBRE = NOMBRE
        self.GESTOR_CV2 = ManagerCV2(cv2.VideoCapture(VIDEO),is_stream=STREAM, fps_limit=FPS, detect_scenes=DETECTAR_ESCENAS)
        self.GESTOR_CV2.add_keystroke(27, 1,exit=True) # Salir cuando se presione `Esc`
        self.GESTOR_CV2.add_keystroke(ord(' '), 1, 'action')

    def EJECUTAR(self):
        for IMAGEN in self.GESTOR_CV2:
            IMAGEN = self.next_frame(IMAGEN)
            cv2.imshow(self.NOMBRE, IMAGEN)
        cv2.destroyAllWindows()

    def next_frame(self, IMAGEN, *args):
        return IMAGEN

if __name__ == "__main__":
    import argparse

    # Creamos un objeto ArgumentParser para definir y analizar los argumentos de línea de comandos
    parser = argparse.ArgumentParser()

    # Definimos los argumentos de línea de comandos que se pueden pasar al script
    parser.add_argument('-v', '--video', default=0,
        help='video/flujo de entrada (por defecto 0, es tu cámara web principal)')

    parser.add_argument('-s', '--stream',
        help='si se pasa, significa que el video es un flujo de transmisión',
        action='store_true')

    parser.add_argument('-f', '--fps', default=0,
        help='parámetro entero que indica el límite de FPS (por defecto 0, significa sin límite)',
        type=int)

    # Analizamos los argumentos de línea de comandos proporcionados y los almacenamos en 'args'
    args = parser.parse_args()

    # Verificamos si el argumento 'args.video' es una cadena de dígitos y lo convertimos a entero si es así
    if type(args.video) is str and args.video.isdigit():
        args.video = int(args.video)

    # Creamos una instancia de la clase 'VideoController' con los valores proporcionados por los argumentos
    md = VideoController(args.video, args.stream, args.fps)
    
    # Llamamos al método 'run()' en el objeto 'md' para iniciar la ejecución del controlador de video
    md.EJECUTAR()




