import pyvirtualcam
import numpy as np
import cv2 as cv
import win32gui, win32ui, win32con

def get_screenshot(): 
    hwnd = win32gui.FindWindow(None, 'desktop-app')
    w = 1920
    h = 1080
    window_rect = win32gui.GetWindowRect(hwnd)
    w = window_rect[2] - window_rect[0]
    h = window_rect[3] - window_rect[1]
    border_pixels = 8
    titlebar_pixels = 31
    w = w - (border_pixels * 2)
    h = h - titlebar_pixels - border_pixels
    cropped_x = border_pixels
    cropped_y = titlebar_pixels

    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, (cropped_x, cropped_y), win32con.SRCCOPY)

    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    img = img[...,:3]
    img = np.ascontiguousarray(img)

    return img

def list_window_names():
    def winEnumHandler(hwnd, ctx):
        if win32gui.isWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))
    win32gui.EnumWindows(winEnumHandler, None)

frame = get_screenshot()
fmt = pyvirtualcam.PixelFormat.BGR
with pyvirtualcam.Camera(width=frame.shape[1], height=frame.shape[0], fps=20, fmt=fmt) as cam:
    while True:
        frame = get_screenshot()
        print(f'Using virtual camera: {cam.device}')

        cam.send(frame)
        cam.sleep_until_next_frame()