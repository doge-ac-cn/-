# 修改 1-paint_roi.py 的 label_name_to_value = {'background': 0,'0':1,'1':1}  # 背景标签为0
python 1-paint_roi.py
python 2-gen_nrrd_dataset.py
python 3-get_feature.py