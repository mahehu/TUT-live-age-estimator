# -*- coding: utf-8 -*-

import os
import cv2

"""
Generates pre-scaled output images from source images, that will be shown in the application as celebrity match.
The use case for this is that you might want to compare cropped/aligned face features to cropped/aligned celebrity
features. They might not look very pleasant though, so this gives the option to use unprocessed images in visualization. 
"""

output_size = 300 # A compromise to preserve image quality while getting rid of huge images that would slow down the
# program. The output images are resized further in the application, hopefully to a smaller size than this.

if __name__ == '__main__':
    path = "/home/tommola/TUT-live-age-estimator/recognizers/celebrities/data/FinnishCelebs_unprocessed"
    print(os.path.abspath(path))
    print(os.path.basename(path))
    print(os.path.split(path))
    #print(os.path.join(os.path.split(path)[0:-1]))
    basepath, celebfolder = os.path.split(path)
    os.makedirs(os.path.join(basepath, "visualization_" + celebfolder), exist_ok=True)

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                celebname = os.path.split(root)[1]


                img = cv2.imread(root + os.sep + filename)
                newpath = os.path.join(basepath, "visualization_" + celebfolder, celebname)
                os.makedirs(newpath, exist_ok=True)

                w, h = img.shape[0:2]

                if "Cheek" in celebname:
                    print(w, h)
                    print(filename)

                if w >= h:
                    new_h = output_size
                    new_w = int(output_size*w/h)
                else:
                    new_w = output_size
                    new_h = int(output_size*h/w)

                img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(newpath + os.sep + filename), img)


