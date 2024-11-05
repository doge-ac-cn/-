import os
import json
import cv2
import numpy as np
from labelme import utils
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
from os.path import splitext
from pathlib import Path

image_folder = "image_npy"
mask_folder = 'masks'
image_output_folder = "imgs"
os.makedirs(image_output_folder, exist_ok=True)
mask_output_folder = "mask_png"
os.makedirs(mask_output_folder, exist_ok=True)

for category in os.listdir(image_folder):
    category_path = os.path.join(image_folder, category)


    if os.path.isdir(category_path):
        for patient_folder in tqdm(os.listdir(category_path)):
            patient_path = os.path.join(category_path, patient_folder)
            if os.path.isdir(patient_path) and os.path.isdir (os.path.join(mask_folder,category,patient_folder)):
                for image_file in os.listdir(patient_path):
                    image_path = os.path.join(image_folder,category,patient_folder,image_file)
                    mask_path = os.path.join(mask_folder,category,patient_folder,image_file)
                    image_array,mask_array = np.load(image_path),np.load(mask_path)
                    
                    
                    image_png = (image_array*255).astype(np.uint8)
                    mask_png = (mask_array*255).astype(np.uint8)

                    image_save_path = os.path.join(image_output_folder,category+patient_folder+os.path.splitext(image_file)[0] + ".png")
                    mask_save_path = os.path.join(mask_output_folder,category+patient_folder+os.path.splitext(image_file)[0] + ".png")
                    Image.fromarray(image_png).save(image_save_path)
                    Image.fromarray(mask_png).save(mask_save_path)


hcm = [name[3:] for name in os.listdir(image_output_folder) if 'HCM' in name ]
htn = [name[3:] for name in os.listdir(image_output_folder) if 'HTN' in name ]
wrong_pics = [pic for pic in hcm if pic in htn]
print(wrong_pics,hcm,htn)

for pic in wrong_pics:
    os.remove(os.path.join(image_output_folder, 'HCM'+pic))
    os.remove(os.path.join(mask_output_folder, 'HCM'+splitext(pic)[0]+'_mask.png'))
    os.remove(os.path.join(image_output_folder, 'HTN'+pic))
    os.remove(os.path.join(mask_output_folder, 'HTN'+splitext(pic)[0]+'_mask.png'))

# wrongsize_pics =["HCM20190603 wang lei1.png","HCM20190603 wang lei2.png","HCM20190603 wang lei3.png"]

# for pic in wrongsize_pics:
#     os.remove(os.path.join(image_output_folder, pic))
#     # os.remove(os.path.join('masks',splitext(pic)[0]+'_mask.png'))

def create_3c_file(dir):
    img_name_list = []
    if dir == image_output_folder:
        for pic in os.listdir(dir):
            print (pic,pic[:-5])
            if pic[:-5] not in img_name_list:
                img_name_list .append(pic[:-5])
    else:
        for pic in os.listdir(dir):
            print (pic,pic[:-5])
            if pic[:-9] not in img_name_list:
                img_name_list .append(pic[:-5])

    for name in img_name_list:
        img_file = list(Path(dir).glob(name + '*'))
        img_file.sort()
        # print(name,img_file)
        image = cv2.imread(str(img_file[0]))
        image_total = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

        image_total[:, :, 0] = cv2.imread(str(img_file[0]))[:, :, 0]
        # print (cv2.imread(str(img_file[1])).shape)
        image_total[:, :, 1] = cv2.imread(str(img_file[1]))[:, :, 0]
        image_total[:, :, 2] = cv2.imread(str(img_file[2]))[:, :, 0]

        if dir == image_output_folder:
            save_path = os.path.join(dir+'_3/', name + '.png')
        else:
            save_path = os.path.join(dir+'_3/', name + '_mask.png')
        cv2.imwrite(save_path, image_total)

create_3c_file(image_output_folder)
create_3c_file(mask_output_folder)