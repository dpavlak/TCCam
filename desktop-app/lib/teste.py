import pyvirtualcam
import numpy as np
import cv2 as cv
import win32gui, win32ui, win32con

def get_screenshot(): 
    hwnd = win32gui.FindWindow(None, 'Black Desert - 414534')
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

    # Free Resources
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


while True:
    screenCapture = get_screenshot()

    cv.imshow('teste', screenCapture)
    if cv.waitKey(1) == ord('f'):
        cv.destroyAllWindows()
        break

'''
with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        screenCapture = pyautogui.screenshot()
        screenCapture = np.array(screenCapture)
        screenCapture = cv.cvtColor(screenCapture, cv.COLOR_RGB2BGR)

        cv.imshow('teste', screenCapture)

        frame[:] = cam.frames_sent % 255  
        cam.send(frame)
        cam.sleep_until_next_frame()
'''