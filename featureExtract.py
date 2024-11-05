import pandas as pd
import os
from radiomics import featureextractor
import SimpleITK as sitk
basePath = r'F:\delta-radiomics0904\second'
folders = os.listdir(basePath)
print(folders)

df = pd.DataFrame()
extractor = featureextractor.RadiomicsFeatureExtractor()
for folder in folders:
    files = os.listdir(os.path.join(basePath,folder))
    print(files)
    for file in files:
        if file.endswith('image.nrrd'):
            imageFile = os.path.join(basePath,folder,file)
        if file.endswith('label.nrrd'):
            maskFile = os.path.join(basePath,folder,file)
#     print(imageFile, maskFile)
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = [1, 1, 1]  # unit: mm
    settings['interpolator'] = sitk.sitkNearestNeighbor
    settings['normalize'] = True
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    featureVector = extractor.execute(imageFile, maskFile)
    featureVector = extractor.execute(imageFile, maskFile)
    df_new = pd.DataFrame.from_dict(featureVector.values()).T
    df_new.columns = featureVector.keys()
    df = pd.concat([df,df_new])
df.to_excel(os.path.join(basePath,'results_new.xlsx'))