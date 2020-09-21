
## Summary of chest X-ray screening program:

### System requirments:
- Python 3.5
- CUDA 10.0
- cudnn 7.1.2
- nvidia drivers 410.78

Dependence and requirments：
[requirements.txt](./requirements.txt)

### HARDWARE: (The following specs were used to create the original solution)
- Ubuntu 16.04.4 LTS
- 40 vCPUs, 64 GB memory
- 4 x NVIDIA 1080Ti
Installation guide:

### project structure：
    ├── configs
    │   ├── path_configs.json      (path configuration)
    ├── data                       (cross validation dataset)
    ├── snapshot_path              (model saving path)
    ├── images                     (path of image in README.md)
    ├── src                        (code of training and inference)
    │   ├── inceptionv3_5k         (pretrained model on No.6 hospital data)
    │   ├── dataset                (loading dataset)
    │   ├── models                 (models)
    │   ├── train_model_10_class.py(training model on No.6 hospital data and chexpert data)
    │   ├── train_model_25_class.py(training model on No.6 hospital data)
    │   ├── train_model_chexpert.py(training model on chexpert data)
    │   ├── inference.py           (forcast and generate heatmap)
    │   ├── inference_cpu.py       (inference on cpu device)
    │   ├── inference_gpu.py       (inference on gpu device)
    │   ├── inference.ipynb        (inference on notebook)
    ├── README.md
    └── requirements.txt


## Data 

Following link provides test data and pretrained model for downloading:

http://202.120.39.3:25480/item

CRADI_label_sample.xlsx

25 classes' labels of local hospital data
Labels are PICC implant, aortic abnormalities, aortic arteriosclerosis, aortic unfolding, cardiomegaly, cavity, consolidation, emphysema, hilar adenopathy, interstitial involvement, mass, pacemaker implant, patchy consolidation, pleural abnormalities, pleural adhesion, pleural calcification, pleural effusion, pleural thickening, pneumothorax, prominent bronchovascular marking, pulmonary edema, pulmonary nodule, pulmonary parenchymal calcification, scoliosis, small consolidation.


### Propose of project

**Background**

X-ray chest film is listed as one of the routine physical examination items.X-ray photography, which is fast, simple and economical, has become the preferred choice for chest examination.
X-ray chest film can clearly record the general pathological changes of the lung, such as pulmonary inflammation, mass, tuberculosis, etc. X-ray photography can be used to observe the lesions in the parts with little difference in thickness and density by using different densities of human tissues. Compared with chest fluoroscopy, X-ray image is more clear, and can find subtle lesions; the objective record of image data is conducive to the review and comparison of disease diagnosis and treatment; the radiation dose of patients receiving fluoroscopy is also relatively larger.

**Propose**
According to the chest X-ray images, a variety of examination findings were detected, and the heatmap of the corresponding category of concerned areas was generated.


Implement path and method

- 1). train multi-label classification model by CNN
- 2). generate heatmap by using grad-cam


**Evaluation function:**

auc and f1 score


**Demo:**

```  
python3 inference.py
```



### Instruction of use:

How to run code:

Some training sample data can be found under /images

Run the training model

```  
python3 train_model_25_class.py
```      
Predict with training model and draw a heatmap

```  
python3 inference.py
```

#### Setup
[config]
add path information in (./configs/path_configs_25.json)
- train_label_path: path of training dataset label
- k_fold_path: K fold path
- train_img_path: training dataset path
- test_img_path: test dataset path
- snapshot_path: model saving path

#### Install
```
bash
pip3 install -r requirements.txt
```

#### How to run code

1. train 25 multi-class classifation model on local data
```  
python3 train_model_25_class.py
```      
2. generate heatmap on trained model
```  
python3 inference.py
```  
3. inference file on cpu device
```  
inference_cpu.py
```  
4. inference file on gpu device
```  
inference_gpu.py
```


#################################################################################################################################################################################

# 胸片疾病筛查项目总结

## 一. 技术栈

### 框架和库：
[requirements.txt](./requirements.txt)

### 项目结构：
    ├── configs
    │   ├── path_configs.json      (设置路径文件)
    ├── data                       (交叉验证数据集)
    ├── snapshot_path              (模型保存路径)
    ├── images                     (README.md图片路径)
    ├── src                        (训练和预测的代码)
    │   ├── inceptionv3_5k         (在六院数据上预训练的模型)
    │   ├── dataset                (读取数据代码)
    │   ├── models                 (模型代码)
    │   ├── train_model_10_class.py(用六院数据和chexpert数据训练模型)
    │   ├── train_model_25_class.py(用六院数据训练模型)
    │   ├── train_model_chexpert.py(用chexpert数据训练模型)
    │   ├── inference.py           (预测和生成热力图)
    │   ├── inference_cpu.py       (在cpu上预测)
    │   ├── inference_gpu.py       (在gpu上预测)
    │   ├── inference.ipynb        (在notebook上预测)
    ├── README.md
    └── requirements.txt


### 数据目录：
```
/data/raw_data_repository/chestXray
```
包含六院数据90k例 和cheXpert 200k例

- chext_expert_local_10_classes.csv 为六院数据和cheXpert数据合并得到的10个类的标签
其中var_0到var_9对应的标签为不张，实变，心影增大，气胸，肺水肿，肺内占位，肺炎，胸腔积液，骨折。
- cleaned_report_25_classes_extraction.csv 为六院数据的25个类的标签
其中var_0到var_24对应的标签为气胸，肺气肿，肺内钙化，PICC， 动脉弓迂曲，动脉弓钙化，动脉异常，小片影，心影增大，斑片影，肺内阴影，空洞，肺内占位，肺纹理增多，水肿，肺结节，肺门异常，胸腔积液，胸膜增厚，胸膜粘连，胸膜钙化，胸膜异常，脊柱侧弯，起搏器植入后，间质改变。

> chexpert
  >+ chexpert (cheXpert 数据)
  >+ local_data (六院数据)
  >+ all_png_512 (cheXpert + 六院数据)

## 二. 业务部分

### 项目背景
X线胸片，通俗地被称为“拍片”，也列为常规体检的检查项目之一。X线摄影的快捷、简便、经济的优势日渐突出，成为胸部检查的优先选择。
X线胸片能清晰地记录肺部的大体病变，如肺部炎症、肿块、结核等。X线摄片利用人体组织的不同密度可观察到厚度和密度差别较小部位的病变。相比胸部透视，X线胸片显像更清楚，能发现细微的病变；影像资料的客观记录有利于疾病诊治的复查对比；患者接受透视的射线剂量也相对更大。

### 项目计划目标
目标：根据胸部X-ray图片检测多种检查所见，并做出相应类别关注区域的热力图。

### 实施路径和方法
- 1). 用深度卷积神经网络训练多标签分类模型
- 2). 用grad-cam方法做出类别的关注区域热力图


#### 评价指标
分类metrics 为auc 和 f1 score

## 三．工程化部分
### 可执行代码
#### HARDWARE: (The following specs were used to create the original solution)
- Ubuntu 16.04.4 LTS
- 40 vCPUs, 64 GB memory
- 4 x NVIDIA 1080Ti

#### SOFTWARE:
- Python 3.5
- CUDA 10.0
- cudnn 7.1.2
- nvidia drivers 410.78

#### Setup
[config](./configs/path_configs_25.json)中设置数据路径
- train_label_path: 训练集标签地址
- k_fold_path: 数据集划分路径
- train_img_path: 训练集图像路径
- test_img_path: 测试集图像路径
- snapshot_path: 模型保存路径

#### Install
```bash
pip3 install -r requirements.txt
```

#### How to run code

2. 训练本地数据的25个类的多标签分类模型
```  
python3 train_model_25_class.py
```      
3. 用已训练的模型预测，并画出热力图
```  
python3 inference.py
```  
4. cpu版本的服务部署代码
```  
inference_cpu.py
```  
5. gpu版本的服务部署代码
```  
inference_gpu.py
```  
    
#### 训练集输出文档


|字段| 表示 |
|  ----  | ----  |
|epochID|训练步数|
|lr|学习率|
|trainLoss|训练集损失| 
|valLoss|验证集损失| 
|epoch_time|每步运行时间| 
|mean(val_f1)|所有类别的平均F1 SCORE|
|mean(val_auc)|所有类别的平均AUC|
|val_threshold|每个类别的最优阈值| 
|val_f1|每个类别的F1 SCORE|
|val_auc|每个类别的AUC|
|precision_list|每个类别的precision|
|recall_list|每个类别的recall|

