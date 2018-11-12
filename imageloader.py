from PIL import Image
import os
# import cv2
import numpy as np
from array import array

def load_images_from_folder(folder):
    print("Loading images from coil-100 dataset ......")
    pix_val = []
    for filename in os.listdir(folder):
        # print(filename)
        img = Image.open(folder+filename)        
        if img is not None:
            pix = list(img.getdata())
            pix_val.append(pix)
        img.close()
    return pix_val

# im = Image.open(r"/Users/linjian/Documents/coil-100/obj1__0.png")
folder = "/Users/linjian/Documents/coil-100/"
pix_val = load_images_from_folder(folder)

pix = np.asarray(pix_val)
print(pix.shape)

pix_out = np.reshape(pix,(7200,128,128,3)).astype(float)

output_file = open('coil-100.bin', 'wb')
print("Print out pixels ......")
pix_out.tofile(output_file)
output_file.close()

