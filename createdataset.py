import glob, os, shutil, cv2
import numpy as np

folder = 'F:\\annotations\\'

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders
print (fast_scandir(folder))
for folderpath in fast_scandir(folder):
    print ('loop')
    newfile = np.zeros([128, 128])
    for file_path in glob.glob(os.path.join(folderpath, '*.*')):

        temp = cv2.imread(file_path, 0)
        newfile = newfile + temp
        newfilename = os.path.basename(file_path)
        newfilename = file_path.rsplit('_defect', 1)[0]
        newfilename = newfilename + '.png'


    cv2.imwrite(newfilename, newfile)

'''for folderpath in fast_scandir(folder):

    for file_path in glob.glob(os.path.join(folderpath, '*.*')):
        filename = os.path.basename(file_path)
        filename = file_path.rsplit('_defect', 1)[0]
        filename = filename + '.png'
        newfile = np.zeros([128, 128])
        temp = cv2.imread(file_path,0)
        newfile = newfile + temp
        a = np.clip (newfile,0,255)
        print (folderpath)
        #cv2.imwrite(filename, a)

    new_dir = file_path.rsplit('_defect', 1)[0]
    try:
        os.makedirs(os.path.join(folder, new_dir))
    except WindowsError:
        # Handle the case where the target dir already exist.
        pass
    shutil.move(file_path, os.path.join(new_dir, os.path.basename(file_path)))'''