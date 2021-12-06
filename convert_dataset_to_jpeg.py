import os
import sys
from PIL import Image

ego2hands_root_dir = '/home/alex/Documents/Data/Ego2Hands/train'
#'C:/School/Alex/PoseDatasets/Ego2HandsPose/train_orig'
#'/home/alex/Documents/Data/Ego2Hands/train'
count = 0
for root, dirs, files in os.walk(ego2hands_root_dir):
    for file_name in files:
        if file_name.endswith(".png") and "_" not in file_name:
            if os.path.exists(os.path.join(root, 'mano.npy')):
                img_png = Image.open(os.path.join(root, file_name))
                img_png = img_png.convert('RGB')
                img_png.save(os.path.join(root, file_name).replace('png', 'jpg'))
                count += 1
                if count % 100 == 0:
                    print("Converted count = {}".format(count))

print("Finished count = {}".format(count))
