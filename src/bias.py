import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf


def downsampleImage(img):
    import scipy.signal
    F = 0.25*np.ones((2, 2))
    return scipy.signal.convolve2d(img, F, 'valid')[::2, ::2]


img = 255 * np.ones((512, 512))
X, Y = np.meshgrid(np.arange(512), np.arange(512))
img[((X-300)**2+(Y-300)**2)<40**2] = 0

img = cv2.resize(img, (256,256))
cv2.imwrite('bias6-0.png', img.astype('uint8'))
img = cv2.resize(img, (128,128))
cv2.imwrite('bias7-0.png', img.astype('uint8'))

'''
img = plt.imread('bias2-0.png') * 255
print(img)
img = np.reshape(img,(256,256,1))
im = []
im.append(img)
img = np.asarray(im)
print(img.shape)
vv = tf.nn.avg_pool2d(tf.convert_to_tensor(img, dtype=tf.float32), ksize=(2, 2), strides=2, padding='VALID').numpy().astype(np.uint8)

plt.imshow(np.squeeze(vv[0]))
plt.show()
cv2.imwrite('bias3-0.png', np.squeeze(vv[0]).astype('uint8'))
'''
'''
img = plt.imread('bias3-0.png')
plt.imshow(np.squeeze(img), cmap='gray')
plt.plot(10.63 + 64, 64 + 10.63, 'r+', color='red')
plt.plot(11 + 64, 64 + 11, 'r+', color='red')
plt.title('Prevision')
plt.show()
'''