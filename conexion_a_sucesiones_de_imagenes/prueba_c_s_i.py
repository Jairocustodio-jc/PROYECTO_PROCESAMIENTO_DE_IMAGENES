import cv2
from cv2_tools.Management import ManagerCV2
import argparse

class VideoController():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', default=0,
                        help='video/flujo de entrada (por defecto 0, es tu cámara web principal)')
    parser.add_argument('-s', '--stream',
                        help='si se pasa, significa que el video es un flujo de transmisión',
                        action='store_true')
    parser.add_argument('-f', '--fps', default=0,
                        help='parámetro entero que indica el límite de FPS (por defecto 0, significa sin límite)',
                        type=int)
    args = parser.parse_args()

    if type(args.video) is str and args.video.isdigit():
        args.video = int(args.video)

    md = VideoController(args.video, args.stream, args.fps)
    md.EJECUTAR()
