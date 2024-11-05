import os
import json
import cv2
import numpy as np
from labelme import utils
from tqdm import tqdm

def create_mask(json_file, image_shape):
    with open(json_file, 'r') as f:
        data = json.load(f)

    shapes = data['shapes']
    label_name_to_value = {'background': 0,'0':1,'1':0}  # 背景标签为0

    # for shape in shapes:
    #     label_name = shape['label']
    #     if label_name not in label_name_to_value:
    #         label_name_to_value[label_name] = len(label_name_to_value)
    
    print (label_name_to_value)
    # labels = [label_name_to_value[shape['label']] for shape in shapes]
    mask = utils.shapes_to_label(image_shape, shapes, label_name_to_value)

    return mask

def main():
    input_folder = "label"
    output_folder = "masks"
    os.makedirs(output_folder, exist_ok=True)

    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        if os.path.isdir(category_path):
            for patient_folder in tqdm(os.listdir(category_path)):
                patient_path = os.path.join(category_path, patient_folder)
                if os.path.isdir(patient_path):
                    for json_file in os.listdir(patient_path):
                        if json_file.endswith('.json'):
                            json_path = os.path.join(patient_path, json_file)
                            image_path = os.path.splitext(json_path)[0] + ".png"
                            image = cv2.imread(image_path)
                            mask = create_mask(json_path, image.shape[:2])
                            
                            os.makedirs(os.path.join(output_folder,category), exist_ok=True)
                            os.makedirs(os.path.join(output_folder,category,patient_folder), exist_ok=True)


                            output_path = os.path.join(output_folder,category, patient_folder,f"{os.path.splitext(json_file)[0]}.npy")
                            np.save(output_path, mask[0])

if __name__ == "__main__":
    main()
