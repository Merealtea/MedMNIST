# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:26:12 2021

@author: Yuhao Wang Xingyuan Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from Resnet import Resnet
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score

def preprocess(img):
    #预处理
    if len(img.shape) == 3:
        img = np.expand_dims(img,3)
    img = img / 255
    return img

def get_img_lab(data,task,kind):
    '''
    将npz文件中的images和label分别打包成dict格式img和lab
    img和lab的columns为train，test，val
    img['train']代表训练集图片
    lab['train']代表训练集标签
    '''
    train = data['train_images']
    test = data['test_images']
    val = data['val_images']
    
    train = preprocess(train)
    val = preprocess(val)
    test = preprocess(test)
    train = train.transpose((0,3,1,2))
    test = test.transpose((0,3,1,2))
    val = val.transpose((0,3,1,2))
    img = {'train':train,'test':test,'val':val}
    
    train = data['train_labels']
    test = data['test_labels']
    val = data['val_labels']
    if task == 1:
        train = train.reshape(-1,kind)
        test = test.reshape(-1,kind)
        val = val.reshape(-1,kind)
    elif task == 2:
        label_new = np.zeros((train.shape[0],kind))
        k = np.arange(0,kind,1)
        for i in range(train.shape[0]):
            label_new[i,:] = np.where(k<train[i],1,0)
        train = label_new
        
        label_new = np.zeros((test.shape[0],kind))
        k = np.arange(0,kind,1)
        for i in range(test.shape[0]):
            label_new[i,:] = np.where(k<test[i],1,0)
        test = label_new
        
        label_new = np.zeros((val.shape[0],kind))
        k = np.arange(0,kind,1)
        for i in range(val.shape[0]):
            label_new[i,:] = np.where(k<val[i],1,0)
        val = label_new
    else:
        train = train.reshape(-1)
        test = test.reshape(-1)
        val = val.reshape(-1)
    lab = {'train':train,'test':test,'val':val}
    return img, lab

def get_ACC(score,label,task):
    if task:
        normalize = nn.Sigmoid()
    else:
        normalize = nn.Softmax(dim = 1)
    score = normalize(score.cpu()).detach().numpy()
    
    if task == 1:
        score = np.where(score>0.5,1,0)
        acc = 0
        for i in range(score.shape[1]):
            acc += np.mean(score[:,i] == label[:,i])
        acc = acc / score.shape[1]
    elif task == 2:
        score = np.where(score>0.5,1,0).sum(axis = 1)
        label = np.sum(label,axis = 1)
        acc = np.mean(score == label)

    else:
        score = np.argmax(score,axis = 1)
        acc = np.mean(score == label)
    return acc

def get_AUC(score,label,task,Clss):
    if task:
        normalize = nn.Sigmoid()
    else:
        normalize = nn.Softmax(dim = 1)
    score = normalize(score).cpu().detach().numpy()
    if task :
        auc = 0
        for i in range(score.shape[1]):
            auc += roc_auc_score(label[:,i], score[:,i])
        auc = auc / score.shape[1]
    elif Clss == 2:
        auc = roc_auc_score(label,score[:,1])
    else:
        auc = 0
        for i in range(Clss):
            l = np.where(label == i,1,0)
            auc +=roc_auc_score(l, score[:,i])
        auc = auc / Clss
    return auc

def saveresult(train_acc,train_auc,train_loss,val_acc,val_auc,test_acc,test_auc,task):
    with open("{}_result.txt".format(task),"a+") as f:
        str1=" ".join([str(x) for x in train_acc])
        str2=" ".join([str(x) for x in train_auc])
        str3=" ".join([str(x) for x in train_loss])
        f=open("{}_result.txt".format(task),'a+')
        f.write(task+'\n'+'train_acc:['+str1+']'+'\n'+'train_auc:['+str2+']'+'\n'+ \
            'train_loss:['+str3+']'+'\n'+'val_acc:'+str(val_acc)+'     val_auc:'+str(val_auc)+\
            '\n'+'test_acc:'+str(test_acc)+'     test_auc:'+str(val_auc)+'\n'+'\n')

def main():
    name = {'breast': 'breastmnist.npz','chest':'chestmnist.npz',
            'derma':'dermamnist.npz', 'oct':'octmnist.npz',
            'axial':'organmnist_axial.npz','coronal':'organmnist_coronal.npz',
            'sagittal':'organmnist_sagittal.npz','path':'pathmnist.npz',
            'pneumonia':'pneumoniamnist.npz','retina':'retinamnist.npz'}
    
    filepath = '/MedMNIST的数据集路径/'
    keys = name.keys()
    for key in keys:
        file = name[key]
        with np.load(filepath + file, allow_pickle=True) as f :
            kind = {'breast':(1,2,0),'chest':(1,14,1),'derma':(3,7,0),'oct':(1,4,0),
                'axial':(1,11,0),'coronal':(1,11,0),'sagittal':(1,11,0),'path':(3,9,0),
                'pneumonia':(1,2,0),'retina':(3,4,2)} #第三位 1代表多label二分类
            config = [kind[key][0],kind[key][1],kind[key][2],50]
            img , lab = get_img_lab(f,config[2],config[1])
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Resnet(config).to(device)
    
        train = torch.FloatTensor(img['train'])
        train_lab = lab['train']
        test = torch.FloatTensor(img['test'])
        test_lab = lab['test']
        if config[2]:
            train_lab = torch.FloatTensor(train_lab)
        else:
            train_lab = torch.LongTensor(train_lab)
        
        ##########====存储的数据变量====#############
        loss_train = []                            #
        acc_train = []                             #
        auc_train = []                             #
        acc_test = None                            #
        auc_test = None                            #
        acc_val = None                             #
        auc_val = None                             #
        ############################################

        train_num = train.shape[0]
        opt = torch.optim.Adam(model.parameters(),lr = 0.001)
        epoches = 100
        batch_size = 64

        for i in tqdm(range(epoches)):
            for j in range(train_num // batch_size+1):
                inputs = train[j*batch_size:(j+1)*batch_size,:,:,:].to(device)
                label = train_lab[j*batch_size:(j+1)*batch_size].to(device)
                
                opt.zero_grad()
                outputs = model.forward(inputs)
                loss = model.loss(outputs,label)
                loss.backward()
                opt.step()
                loss_train.append(loss.item())
            if i % 5 == 0:
                torch.cuda.empty_cache()
                out = torch.FloatTensor(np.array([x for x in range(config[1])])).reshape(1,-1)
                with torch.no_grad():
                    for j in range(train_num //batch_size+1):
                        inputs = train[j*batch_size:(j+1)*batch_size,:,:,:].to(device)
                        outputs = model.forward(inputs)
                        outputs = outputs.cpu()
                        out = torch.cat([out,outputs],dim = 0)
                    label = train_lab.detach().numpy()
                    acc_train.append(get_ACC(out[1:], label, config[2]))
                    auc_train.append(get_AUC(out[1:], label, config[2], config[1]))

        valid = torch.FloatTensor(img['val']).to(device)
        val_lab = lab['val']
        val_num = valid.shape[0]
        
        torch.cuda.empty_cache()
        out = torch.FloatTensor(np.array([x for x in range(config[1])])).reshape(1,-1)
        with torch.no_grad():
            for j in range(val_num //batch_size + 1):
                inputs = valid[j*batch_size:(j+1)*batch_size,:,:,:].to(device)
                outputs = model.forward(inputs).cpu()
                out =torch.cat([out,outputs],dim = 0)
            acc_val = get_ACC(out[1:], val_lab, config[2])
            auc_val = get_AUC(out[1:], val_lab, config[2], config[1])
    
        test_num = test.shape[0]
        torch.cuda.empty_cache()
        out = torch.FloatTensor(np.array([x for x in range(config[1])])).reshape(1,-1)
        with torch.no_grad():
            for i in range(test_num // batch_size + 1):
                inputs = test[i*batch_size:(i+1)*batch_size,:,:,:].to(device)
                outputs = model.forward(inputs).cpu()
                out =torch.cat([out,outputs],dim = 0)
            acc_test = get_ACC(out[1:], test_lab, config[2])
            auc_test = get_AUC(out[1:], test_lab, config[2], config[1])
        print('==>输出结果')
    
        print('训练集ACC：{}'.format(acc_train[-1]))
        print('训练集AUC：{}'.format(auc_train[-1]))
        print('测试集ACC：{}'.format(acc_test))
        print('测试集AUC：{}'.format(auc_test))
        print('验证集ACC：{}'.format(acc_val))
        print('验证集AUC：{}'.format(auc_val))
        
        plt.figure()
        plt.plot(loss_train)
        plt.title('Train - Loss')
        plt.xlabel('Epoches')
        plt.ylabel('Loss')
        plt.savefig('{}_Train_loss.jpg'.format(key))
        plt.show()
    
        x = np.arange(0,epoches,5)
        plt.figure()
        plt.plot(x,acc_train)
        plt.title('Train - ACC')
        plt.xlabel('Epoches/5')
        plt.ylabel('Accuracy')
        plt.savefig('{}_Train_acc.jpg'.format(key))
        plt.show()
        plt.figure()
        plt.plot(x,auc_train)
        plt.title('Train - AUC')
        plt.xlabel('Epoches/5')
        plt.ylabel('AUC')
        plt.savefig('{}_Train_auc.jpg'.format(key))
        plt.show()
        saveresult(acc_train,auc_train,loss_train,acc_val,auc_val,acc_test,auc_test,key)
    
if __name__ == "__main__":
    main()