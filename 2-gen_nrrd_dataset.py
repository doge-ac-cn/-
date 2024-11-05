import os
import json
import cv2
import numpy as np
from labelme import utils
from tqdm import tqdm
import SimpleITK as sitk


image_folder = "image_npy"
mask_folder = 'masks'
output_folder = "image_nrrd"
os.makedirs(output_folder, exist_ok=True)

for category in os.listdir(image_folder):
    category_path = os.path.join(image_folder, category)
    if os.path.isdir(category_path):
        os.makedirs(os.path.join(output_folder,category), exist_ok=True)
        for patient_folder in tqdm(os.listdir(category_path)):
            os.makedirs(os.path.join(output_folder,category,patient_folder), exist_ok=True)
            patient_path = os.path.join(category_path, patient_folder)
            if os.path.isdir(patient_path) and os.path.isdir (os.path.join(mask_folder,category,patient_folder)):
                for image_file in os.listdir(patient_path):
                    image_path = os.path.join(image_folder,category,patient_folder,image_file)
                    mask_path = os.path.join(mask_folder,category,patient_folder,image_file)
                    image_array,mask_array = np.load(image_path),np.load(mask_path)
                    
                    image_sitk = sitk.GetImageFromArray(image_array)
                    mask_sitk = sitk.GetImageFromArray(mask_array)
                    print (image_path,np.unique(mask_sitk))
                    image_save_path = os.path.join(output_folder,category,patient_folder,os.path.splitext(image_file)[0] + "_image.nrrd")
                    mask_save_path = os.path.join(output_folder,category,patient_folder,os.path.splitext(image_file)[0] + "_mask.nrrd")

                    sitk.WriteImage (image_sitk,image_save_path)
                    sitk.WriteImage (mask_sitk,mask_save_path)


hcm = os.listdir(os.path.join(output_folder,'HCM'))
htn = os.listdir(os.path.join(output_folder,'HTN'))
wrong_pics = [pic for pic in hcm if pic in htn]
print(wrong_pics)

import shutil

for pic in wrong_pics:
    shutil.rmtree(os.path.join(os.path.join(output_folder,'HCM'), pic))
    shutil.rmtree(os.path.join(os.path.join(output_folder,'HTN'), pic))


def clean_empty_dir (dir_name):


    for dir in os.listdir(os.path.join(output_folder,dir_name)):
        if len(os.listdir(os.path.join(output_folder,dir_name,dir)))==0:
            print ('remove empty',os.path.join(output_folder,dir_name,dir))
            shutil.rmtree(os.path.join(output_folder,dir_name,dir))

clean_empty_dir('HCM')
clean_empty_dir('HTN')
clean_empty_dir('NOR')