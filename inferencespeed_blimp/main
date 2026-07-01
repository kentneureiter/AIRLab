# main - By: kentneureiter
import sensor
import time
from pyb import USB_VCP

# Camera setup
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.HQVGA)
sensor.skip_frames(time=2000)

usb = USB_VCP()

MAGIC = b'\xFF\xFE\xFD\xFC'  # 4-byte header so Mac knows a frame is starting

print("NiclaVision ready - waiting for receiver.py on Mac")

while True:
    img = sensor.snapshot()
    data = img.bytearray()       # 240 * 160 * 2 = 76800 bytes of raw RGB565

    usb.write(MAGIC)             # send header
    usb.write(data)              # send raw frame

    # Wait for Mac to acknowledge before capturing next frame
    while usb.any() == 0:
        pass
    usb.read(1)
