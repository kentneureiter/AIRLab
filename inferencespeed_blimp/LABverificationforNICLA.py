import sensor
from ulab import numpy as np

sensor.reset()
sensor.set_pixformat(sensor.LAB)   # capture in LAB directly
sensor.set_framesize(sensor.HQVGA)
sensor.skip_frames(time=1000)

img = sensor.snapshot()
data = img.bytearray()

print(len(data))          # should be 240 * 160 * 3 = 115200
arr = np.frombuffer(data, dtype=np.uint8)
print(arr.shape)          # should be (115200,)
arr = arr.reshape((160, 240, 3))
print(arr.shape)          # should be (160, 240, 3)

# Peek at one cell's A and B values
cell = arr[0:11, 0:11, :]
print("A mean raw:", np.mean(cell[:, :, 1]))   # raw byte, need to subtract 128 later
print("B mean raw:", np.mean(cell[:, :, 2]))
