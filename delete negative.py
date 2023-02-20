####Delete file#####
'''
import os
import numpy as np
import cv2
os.chdir('C:\\Users\\ASPLAB\\unet\\inputs\\BGD\\masks\\0')
imagefolder = 'C:\\Users\\ASPLAB\\unet\\inputs\\BGD\\images'
maskfolder = 'C:\\Users\\ASPLAB\\unet\\inputs\\BGD\\masks\\0'
print(os.getcwd())
imageext = ".jpeg"
for count, f in enumerate(os.listdir()):
    name, ext = os.path.splitext(f)
    mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    total = np.sum(mask)
    if total <= 5:
        image = os.path.join(imagefolder, f'{name}{imageext}')
        os.remove(f)
        os.remove(image)
'''

### invert color

import os
import numpy as np
import cv2
os.chdir('C:\\Users\\ASPLAB\\unet\\inputs\\BKK+\\images')
imagefolder = 'C:\\Users\\ASPLAB\\unet\\inputs\\BKK+\\test'
print(os.getcwd())
imageext = ".bmp"
for count, f in enumerate(os.listdir()):
    print(f)
    name, ext = os.path.splitext(f)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img_invert = abs(255-img)
    img_invert[img_invert >= 250] = 0
    newpath = os.path.join(imagefolder, f'{name}{imageext}')
    cv2.imwrite(newpath, img_invert)