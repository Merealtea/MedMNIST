# MedMNIST
SJTU AI2612 Final HomeWork

Author：王宇昊 陈星元

Med_main.py：导入数据、数据预处理、训练模型、预测结果
             使用时将filepath改为MedMNIST数据集所在位置
             
ResNet.py：ResNet模型，包含了ResNet18和50两种模型

Result：包含了十个数据集在ResNet18和50的训练结果

        结果里面包含了：train集的acc、auc和loss的文本记录和可视化
        
                        valid集的acc和auc文本记录
                        
                        test集的acc和auc的文本记录

Retina新策略结果：包含了我们对Retina有序回归任务实施了新的训练策略后得到的模型结果
