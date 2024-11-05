# 创建一个空字典来存储结果
results = {}

# 打开结果文件
with open("result_new.log", "r") as file:
    lines = file.readlines()

# 循环遍历文件中的每一行
i = 0
while i < len(lines):
    # 提取ROI区域、特征降维方法和模型名
    roi = lines[i].split("：")[1].split("，")[0].strip()
    feature_reduction = lines[i].split("：")[2].split("，")[0].strip()
    model_name = lines[i].split("：")[3].strip()

    # 提取模型评价指标
    metrics = lines[i+1:i+6]  # 提取下面6行的指标信息

    # 提取准确度、F1分数和AUC值
    metrics_dict = {}
    for metric in metrics:
        # print (metric.split(":"))
        metric_name, metric_value = metric.split(":")[0],metric.split(":")[1]
        metrics_dict[metric_name.strip()] = float(metric_value.strip())

    # 将提取的信息存储为字典
    result_info = metrics_dict

    # 构建嵌套字典，如果ROI区域和特征降维方法已经存在，则将新的模型结果添加到现有的字典中
    if roi in results:
        if feature_reduction in results[roi]:
            results[roi][feature_reduction][model_name] = result_info
        else:
            results[roi][feature_reduction] = {model_name: result_info}
    else:
        results[roi] = {feature_reduction: {model_name: result_info}}

    # 移动到下一个ROI区域的数据
    i += 6



roi_avg_metrics = {}

for roi, roi_data in results.items():
    roi_metrics_sum = {
        "Accuracy": 0,
        "F1 Score": 0,
        "sensitivity": 0,
        "specificity": 0,
        "AUC": 0
    }
    num_models = 0

    for feature_reduction, model_data in roi_data.items():
        for model_name, metrics in model_data.items():
            # 计算每个指标的总和
            for metric_name, metric_value in metrics.items():
                roi_metrics_sum[metric_name] += metric_value

            num_models += 1

    # 计算平均值
    roi_avg_metrics[roi] = {metric_name: metric_sum / num_models for metric_name, metric_sum in roi_metrics_sum.items()}

# 打印每个 ROI 区域的平均指标
for roi, avg_metrics in roi_avg_metrics.items():
    print(f"ROI: {roi}")
    for metric_name, metric_value in avg_metrics.items():
        print(f"{metric_name}: {metric_value}")
    print()



feature_avg_metrics = {}

for roi_data in results.values():
    for feature_reduction, model_data in roi_data.items():
        feature_metrics_sum = {
            "Accuracy": 0,
            "F1 Score": 0,
            "sensitivity": 0,
            "specificity": 0,
            "AUC": 0
        }
        num_models = 0

        for model_name, metrics in model_data.items():
            # 计算每个指标的总和
            for metric_name, metric_value in metrics.items():
                feature_metrics_sum[metric_name] += metric_value

            num_models += 1

        # 计算平均值
        if feature_reduction in feature_avg_metrics:
            for metric_name, metric_sum in feature_metrics_sum.items():
                feature_avg_metrics[feature_reduction][metric_name] += metric_sum
            feature_avg_metrics[feature_reduction]['NumModels'] += num_models
        else:
            feature_avg_metrics[feature_reduction] = {metric_name: metric_sum for metric_name, metric_sum in feature_metrics_sum.items()}
            feature_avg_metrics[feature_reduction]['NumModels'] = num_models

# 计算平均值
for feature_reduction, metrics in feature_avg_metrics.items():
    num_models = metrics.pop('NumModels')
    feature_avg_metrics[feature_reduction] = {metric_name: metric_sum / num_models for metric_name, metric_sum in metrics.items()}

# 打印每种特征提取手段的平均指标
for feature_reduction, avg_metrics in feature_avg_metrics.items():
    print(f"Feature Reduction: {feature_reduction}")
    for metric_name, metric_value in avg_metrics.items():
        print(f"{metric_name}: {metric_value}")


roi_feature_avg_metrics = {}

for roi, roi_data in results.items():
    for feature_reduction, model_data in roi_data.items():
        roi_feature_combination = f"{roi}_{feature_reduction}"
        roi_feature_metrics_sum = {
            "Accuracy": 0,
            "F1 Score": 0,
            "sensitivity": 0,
            "specificity": 0,
            "AUC": 0
        }
        num_models = 0

        for model_name, metrics in model_data.items():
            # 计算每个指标的总和
            for metric_name, metric_value in metrics.items():
                roi_feature_metrics_sum[metric_name] += metric_value

            num_models += 1

        # 计算平均值
        if roi_feature_combination in roi_feature_avg_metrics:
            for metric_name, metric_sum in roi_feature_metrics_sum.items():
                roi_feature_avg_metrics[roi_feature_combination][metric_name] += metric_sum
            roi_feature_avg_metrics[roi_feature_combination]['NumModels'] += num_models
        else:
            roi_feature_avg_metrics[roi_feature_combination] = {metric_name: metric_sum for metric_name, metric_sum in roi_feature_metrics_sum.items()}
            roi_feature_avg_metrics[roi_feature_combination]['NumModels'] = num_models

# 计算平均值
for roi_feature_combination, metrics in roi_feature_avg_metrics.items():
    num_models = metrics.pop('NumModels')
    roi_feature_avg_metrics[roi_feature_combination] = {metric_name: metric_sum / num_models for metric_name, metric_sum in metrics.items()}

# 打印不同特征提取手段与roi区域组合的平均指标
for roi_feature_combination, avg_metrics in roi_feature_avg_metrics.items():
    print(f"ROI and Feature Reduction Combination: {roi_feature_combination}")
    for metric_name, metric_value in avg_metrics.items():
        print(f"{metric_name}: {metric_value}")
    print()
