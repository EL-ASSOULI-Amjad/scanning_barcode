import cv2
from pyzbar.pyzbar import decode
import streamlit as st

def decode_barcode():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    while True:
        success, frame = camera.read()
        if not success:
            print("Error, camera not available.")
        for code in decode(frame):
            print(code)
            x = code
        cv2.imshow("test-cam", frame)
        cv2.waitKey(1)
        
    print(f"info :{x}")

