from PIL import Image
import os
new_ext = ".png"

def tile(f, dir_in, dir_out):
    name, ext = os.path.splitext(f)
    img = Image.open(os.path.join(dir_in, f))
    new_img = img.resize((128, 128))
    out = os.path.join(dir_out, f'{name}{new_ext}')
    new_img.save(out)

os.chdir('C:\\Users\\ASPLAB\\unet\\inputs\\BKK 60 full\\masks')
print(os.getcwd())
dir_in = 'C:\\Users\\ASPLAB\\unet\\inputs\\BKK 60 full\\masks'
dir_out = 'C:\\Users\\ASPLAB\\unet\\inputs\\BKK\\masks'


for count, f in enumerate(os.listdir()):
    tile(f,dir_in,dir_out)