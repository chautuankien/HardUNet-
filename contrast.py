#directory = 'C:\\Users\\anhng\\unet++\\inputs\\BKK\\images\\test\\'
folder = 'C:\\Users\\anhng\\unet\\inputs\\BKK\\images\\'
import glob, os, shutil
import cv2
from PIL import Image, ImageEnhance
for file_path in glob.glob(os.path.join(folder, '*.*')):


    # -----Reading the image-----------------------------------------------------
    img = cv2.imread(file_path, 1)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)


    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))


    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    name = os.path.basename(file_path)
    des = os.path.join(folder, 'a')
    des = os.path.join(des, name)
    print (des)
    cv2.imwrite(des, final)

    # _____END_____#
