import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mountain.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

row, col = int(height / 2), int(width / 2)
LPF = np.zeros((height, width, 2), np.uint8)
LPF[row - 50:row + 50, col - 50:col + 50] = 1
LPF_shift = dft_shift * LPF
LPF_ishift = np.fft.ifftshift(LPF_shift)
LPF_img = cv2.idft(LPF_ishift)
LPF_img = cv2.magnitude(LPF_img[:, :, 0], LPF_img[:, :, 1])
out = 20*np.log(cv2.magnitude(LPF_shift[:, :, 0], LPF_shift[:, :, 1]))

HPF = np.ones((height, width, 2), np.uint8)
HPF[row - 50:row + 50, col - 50:col + 50] = 0
HPF_shift = dft_shift * HPF
HPF_ishift = np.fft.ifftshift(HPF_shift)
HPF_img = cv2.idft(HPF_ishift)
HPF_img = cv2.magnitude(HPF_img[:, :, 0], HPF_img[:, :, 1])
out2 = 20*np.log(cv2.magnitude(HPF_shift[:, :, 0], HPF_shift[:, :, 1]))


plt.subplot(151), plt.imshow(gray, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(152), plt.imshow(LPF_img, cmap='gray')
plt.title('LPF'), plt.xticks([]), plt.yticks([])
plt.subplot(153), plt.imshow(out, cmap='gray')
plt.title('out1'), plt.xticks([]), plt.yticks([])
plt.subplot(154), plt.imshow(HPF_img, cmap='gray')
plt.title('HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(155), plt.imshow(out2, cmap='gray')
plt.title('out2'), plt.xticks([]), plt.yticks([])

plt.show()
