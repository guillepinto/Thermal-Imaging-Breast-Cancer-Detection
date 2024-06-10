# Data manipulation
import cv2

def crop_breast(img):
  img_copy = img.copy().astype('uint8')

  # Crear un binario invertido de la imagen
  _, thresh = cv2.threshold(img_copy, 1, 255, cv2.THRESH_BINARY)

  # Encontrar contornos
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Si no se encuentran contornos, retornar la imagen original
  if not contours:
    return img

  # Encontrar el rectángulo delimitador que contiene el objeto
  x, y, w, h = cv2.boundingRect(contours[0])

  # Recortar la imagen utilizando el rectángulo delimitador
  cropped_image = img[y:y+h, x:x+w]

  # Hago resize (para que todas las imagenes tengan el mismo tamaño)
  cropped_image = cv2.resize(cropped_image, (112, 112))

  return cropped_image