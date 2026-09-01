import cv2
import numpy as np

# Escala do instrumento
angulo_min = -120
angulo_max = 120

valor_min = 0
valor_max = 1.6

img = cv2.imread(r"C:\Users\guilh\instrument-vision-system\dataset\train\Analog_Instruments\analog_instrument4.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detecta bordas
edges = cv2.Canny(blur, 50, 150)

# Detecta linhas
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=10
)

h, w = img.shape[:2]
cx = w // 2
cy = h // 2

maior_dist = 0
ponteiro = None

for line in lines:
    x1, y1, x2, y2 = line[0]

    d1 = np.hypot(x1 - cx, y1 - cy)
    d2 = np.hypot(x2 - cx, y2 - cy)

    if min(d1, d2) < 30:
        comprimento = np.hypot(x2 - x1, y2 - y1)

        if comprimento > maior_dist:
            maior_dist = comprimento
            ponteiro = (x1, y1, x2, y2)

x1, y1, x2, y2 = ponteiro

if np.hypot(x1 - cx, y1 - cy) < np.hypot(x2 - cx, y2 - cy):
    px, py = x2, y2
else:
    px, py = x1, y1

dx = px - cx
dy = cy - py

angulo = np.degrees(np.arctan2(dy, dx))

valor = np.interp(
    angulo,
    [angulo_min, angulo_max],
    [valor_min, valor_max]
)

print(f"Ângulo: {angulo:.2f}°")
print(f"Leitura: {valor:.2f}")