#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:15:42 2017

@author: zengke
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

class binary_evaluation:
    """二分类模型评估类
    输入:
        y_true:一维np.array对象，实际值
        y_pred:一维np.array对象，预测概率
    方法：
        confusion_matrix(): 计算混淆矩阵的指标   
        threhold(): 产生cut-off阈值
        index(): 计算不同cut-off阈值下的指标
        plot_roc(): 绘制ROC曲线
        plot_gains(): 绘制收益曲线，常用于营销类二分类模型
        plot_lift(): 绘制提升曲线，常用于营销类二分类模型
        plot_lorenz(): 绘制洛伦兹曲线，常用于信用风险建模
        plot_ks(): 绘制ks曲线，常用于信用风险建模 
        plot_afdr_adr(): 绘制afdr_adr曲线，常用于欺诈侦测模型
        plot_precision_recall(): 绘制精准率-召回率曲线，功能与ROC曲线类似，但在样本不平衡时，ROC曲线更佳稳健                              
    """
    
    def __init__(self,y_true,y_pred):
        
        self.true = y_true
        self.pred = y_pred
        
    
    def confusion_matrix(self,pred):
        
        # 产生混淆矩阵的四个指标
        tn, fp, fn, tp = metrics.confusion_matrix(self.true,pred).ravel()
        
        # 产生衍生指标
        fpr = fp/(fp + tn) #假真率／特异度
        tpr = tp/(tp + fn) #灵敏度／召回率
        depth = (tp + fp)/(tn+fp+fn+tp) #Rate of positive predictions.
        ppv = tp/(tp + fp) #精准率
        lift = ppv/((tp + fn)/(tn+fp+fn+tp)) #提升度
        afdr = fp/tp #(虚报／命中)／好账户误判率
        return(fpr,tpr,depth,ppv,lift,afdr)


    def threhold(self):
       
        min_score = min(self.pred) #预测概率最大值
        max_score = max(self.pred) #预测概率最小值
        
         # 产生分阈值
        thr = np.linspace(min_score, max_score, 100)
        return(thr)

        
    def index(self):
        """计算不同cutoff下的相关统计量
        涉及统计量为：
            fpr: 假正率/特异度
            tpr: 真正率／灵敏度／召回率
            depth: 深度／响应率
            ppv: 精准率／命中率
            lift: 提升度
            afdr: 误判率
        """
        
        # 产生cutoff阈值 
        threhold = self.threhold()
        
        # 产生指标列表
        fpr_list = []
        tpr_list = []
        depth_list = []
        ppv_list = []
        lift_list = []
        afdr_list = []
        
        # 遍历每一个cutoff下的指标，并输出结果
        for i in threhold:
            
            # cutoff下产生0，1预测值
            pred = (self.pred > i).astype('int64')
            
            #产生指标
            fpr,tpr,depth,ppv,lift,afdr = self.confusion_matrix(pred = pred)
            
            #append指标                
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            depth_list.append(depth)
            ppv_list.append(ppv)
            lift_list.append(lift)
            afdr_list.append(afdr)
            
        return(fpr_list,tpr_list,depth_list,ppv_list,lift_list,afdr_list)
 
        
    def plot_roc(self,color='red',legend='model',title='Roc_Curve',xlabel='FPR',ylabel='TPR'):
        """绘制ROC曲线
        X轴 : fpr 
        Y轴 : tpr
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """
        
        # 计算auc
        auc = round(metrics.roc_auc_score(self.true,self.pred),2) 
            
        # 计算fpr(x轴)和tpr(y轴)
        fpr,tpr,_,_,_,_ = self.index()       
        
        #绘图
        plt.plot(fpr,tpr,
                 color = color,
                 label = legend + ' auc: {}'.format(auc),
                 linestyle = '-')

        plt.plot([0,1],[0,1],
                 color = 'black',
                 linestyle = '--')    

        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.legend(loc=4)
        
    
    def plot_gains(self,color='red',legend='model',title='Gain_Curve',xlabel='Depth/RPP',ylabel='ppv'):
        """绘制增益曲线，该曲线常用于营销类模型
        X轴 : depth
        Y轴 : ppv
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """  
        
        # 计算depth(x轴)和ppv(y轴)
        _,_,depth,ppv,_,_ = self.index()
        
        # 绘图
        plt.plot(depth,ppv,
                 color = color,
                 linestyle = '-',
                 label = legend)
        
        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=1)
        
        
    def plot_lift(self,color='red',legend='model',title='Lift_Curve',xlabel='Depth/RPP',ylabel='Lift'):
        """绘制提升曲线，该曲线常用于营销类模型
        X轴 : depth
        Y轴 : lift
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """    
        
        # 计算depth(x轴)和lift(y轴)
        _,_,depth,_,lift,_ = self.index()
        
        # 绘图
        plt.plot(depth,lift,
                 color = color,
                 linestyle = '-',
                 label = legend)

        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=1)    
        
        
    def plot_lorenz(self,color='red',legend='model',title='Lorenz_Curve',xlabel='Depth/RPP',ylabel='True Postive Rate'):
        """绘制洛伦兹曲线，该曲线常用于信用风险建模
        X轴 : depth
        Y轴 : tpr
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """  
        
        # 计算depth(x轴)和tpr(y轴)
        _,tpr,depth,_,_,_ = self.index()
        
        #绘图
        plt.plot(depth,tpr,
                 color = color,
                 linestyle = '-',
                 label = legend)
        
        plt.plot([0,1],[0,1],
                 color = 'black',
                 linestyle = '--') 

        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=4)

        
    def plot_ks(self,color='red',legend='model',title='KS_Curve',xlabel='Depth/RPP',ylabel='KS statistic'):
        """绘制KS曲线，该曲线常用于信用风险建模
        X轴 : depth
        Y轴 : ks
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """
        
        # 计算depth(x轴)和fpr,tpr(y轴)
        fpr,tpr,depth,_,_,_ = self.index()
        ks = np.array(tpr) - np.array(fpr)
        ks_stats = round(max(ks),2) * 100
        
        #绘图            
        plt.plot(depth,ks,
                 color = color,
                 linestyle = '-',
                 label = legend + ' KS : {}'.format(ks_stats))

        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        plt.legend(loc=3)
        
        
    def plot_afdr_adr(self,color='red',legend='model',xlabel='afdr',ylabel='recall',title='AFDR_ADR_Curve'):
        """绘制afdr_adr曲线,该曲线常用于欺诈侦测模型
        X轴 : afdr
        Y轴 : adr／tpr/RECALL
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """
        
        # 计算afdr(x轴)和adr(y轴)
        _,tpr,_,_,_,afdr = self.index()
        
        #绘图       
        plt.plot(afdr,tpr,
                 color = color,
                 linestyle = '-',
                 label = legend)

        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=4)        

        
    def plot_precision_recall(self,color='red',legend='model',xlabel='recall',ylabel='precision',title='Precision_Recall_Curve'):
        """绘制precision_recall曲线,该曲线功能类似于ROC，但样本不平衡时ROC曲线更佳稳健
        X轴 : tpr/recall
        Y轴 : ppv
        参数 :
            color: 颜色
            legend: 图例名
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        """     
        
        # 计算fpr(x轴)和ppv(y轴)
        _,tpr,_,ppv,_,_ = self.index()
        
        #绘图      
        plt.plot(tpr,ppv,
                 color = color,
                 linestyle = '-',
                 label = legend)

        plt.title(title)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 
        plt.legend(loc=3)    