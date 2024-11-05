import os
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
## https://github.com/AIM-Harvard/pyradiomics
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve,ConfusionMatrixDisplay
import copy
import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract features from ROI
def extract_features_from_roi(image,mask):
    settings = {'normalize': True}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    # extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    features = extractor.execute(image, mask)

    # 筛选出数值和数组类型的特征，排除字符串等非数值非数组类型的特征
    filtered_features = {k: v for k, v in features.items() if isinstance(v, (int, float, np.number)) or isinstance(v, np.ndarray)}
    
    return filtered_features


label_dict = {'HCM':0,'HTN':1,'NOR':2}
# Function to load data
def load_data(data_dir):
    labels = []
    features = []
    df = pd.DataFrame()
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        label = class_dir
        for patient_dir in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            # Load DICOM images
            images = [os.path.join(patient_path, f) for f in os.listdir(patient_path) if f.endswith('_image.nrrd')]
            # Load ROI masks
            masks = [os.path.join(patient_path, f) for f in os.listdir(patient_path) if f.endswith('_mask.nrrd')]
            
            features_single_patient = []
            for image, mask in zip(images, masks):
                
                # image_sitk,mask_sitk = sitk.GetArrayFromImage(sitk.ReadImage(image)) , sitk.GetArrayFromImage(sitk.ReadImage(mask))
                # print (np.unique(mask_sitk))
                # print (image_sitk.shape,mask_sitk.shape)
                featureVector = extract_features_from_roi(image, mask)
                features_single_patient.extend(featureVector.values())
                print ('patient_path',len(features_single_patient),patient_path)

            features.append(copy.deepcopy(np.array(list(features_single_patient))))
            labels.append(label_dict[label])
            # print (labels)
    for feature in features:
        if len(feature)!=273:
            print (feature)
    return np.array(features), np.array(labels)

# Main function
def main():
    data_dir = './image_nrrd'
    
    # Load data
    X, y = load_data(data_dir)

    np.save('feature.npy',X)
    np.save('label.npy',y)


if __name__ == "__main__":
    main()
