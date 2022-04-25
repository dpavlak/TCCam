'''
import os
directory = os.getcwd() + "\src\win-dshow\virtualcam-install.bat"
print(directory)

def RunAsAdmin(path_to_file,*args):
    os.system(r'Powershell -Command "Start-Process "'+path_to_file+'"'+ 
                ' -ArgumentList @('+str(args)[1:-1]+')'+ 
                ' -Verb RunAs"' 
    )

RunAsAdmin(directory,'arg1','arg2')

#install = subprocess.check_call("virtualcam-install.bat", cwd=directory, shell=True)
'''

import pyvirtualcam
import numpy as np

with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
    print(f'Using virtual camera: {cam.device}')
    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while True:
        frame[:] = cam.frames_sent % 255  
        cam.send(frame)
        cam.sleep_until_next_frame()
