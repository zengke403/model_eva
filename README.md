# 二分类模型评估类

## 简介:

用于二分类概率类模型评估，包含ROC曲线、增益曲线、提升曲线、洛伦兹曲线、ks曲线等绘制方法

## 示例:

ROC曲线:
    
    from eva import binary_evaluation
    
    binary_evaluation(y_true = test_target,y_pred = test_est_p).plot_roc(color='red')
    binary_evaluation(y_true = train_target,y_pred = train_est_p).plot_roc(color='blue')
    
## 输入:

    **y_true**:一维np.array对象，实际值
    **y_pred**:一维np.array对象，预测概率

## 方法：

    **confusion_matrix()**: 计算混淆矩阵的指标   
    **threhold()**: 产生cut-off阈值
    **index()**: 计算不同cut-off阈值下的指标
    **plot_roc()**: 绘制ROC曲线
    **plot_gains()**: 绘制增益曲线，常用于营销类二分类模型
    **plot_lift()**: 绘制提升曲线，常用于营销类二分类模型
    **plot_lorenz()**: 绘制洛伦兹曲线，常用于信用风险建模
    **plot_ks()**: 绘制ks曲线，常用于信用风险建模 
    **plot_afdr_adr()**: 绘制afdr_adr曲线，常用于欺诈侦测模型
    **plot_precision_recall()**: 绘制精准率-召回率曲线，功能与ROC曲线类似，但在样本不平衡时，ROC曲线更佳稳健                              

## 绘图参数

    **color**: str,颜色
    **legend**: str,图例名
    **title**: str,标题
    **xlabel**: str,x轴标签
    **ylabel**: str,y轴标签
