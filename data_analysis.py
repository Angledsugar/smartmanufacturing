import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sensor1 = pd.read_csv('./data/g1_sensor1.csv', names = ['time', 'normal', 'type1', 'type2', 'type3'])
sensor2 = pd.read_csv('./data/g1_sensor2.csv', names = ['time', 'normal', 'type1', 'type2', 'type3'])
sensor3 = pd.read_csv('./data/g1_sensor3.csv', names = ['time', 'normal', 'type1', 'type2', 'type3'])
sensor4 = pd.read_csv('./data/g1_sensor4.csv', names = ['time', 'normal', 'type1', 'type2', 'type3'])

from scipy import interpolate

x_new = np.arange(0, 140, 0.001)
y_new1 = []; y_new2 = []; y_new3 = []; y_new4 = []

for item in ['normal', 'type1', 'type2', 'type3']:
 f_linear1 = interpolate.interp1d(sensor1['time'], sensor1[item], kind='linear')
 y_new1.append(f_linear1(x_new))
 f_linear2 = interpolate.interp1d(sensor2['time'], sensor2[item], kind='linear')
 y_new2.append(f_linear2(x_new))
 f_linear3 = interpolate.interp1d(sensor3['time'], sensor3[item], kind='linear')
 y_new3.append(f_linear3(x_new))
 f_linear4 = interpolate.interp1d(sensor4['time'], sensor4[item], kind='linear')
 y_new4.append(f_linear4(x_new))

sensor1 = pd.DataFrame(np.array(y_new1).T, columns = ['normal', 'type1', 'type2', 'type3'])
sensor2 = pd.DataFrame(np.array(y_new2).T, columns = ['normal', 'type1', 'type2', 'type3'])
sensor3 = pd.DataFrame(np.array(y_new3).T, columns = ['normal', 'type1', 'type2', 'type3'])
sensor4 = pd.DataFrame(np.array(y_new4).T, columns = ['normal', 'type1', 'type2', 'type3'])

normal_ = pd.concat([sensor1['normal'], sensor2['normal'], sensor3['normal'],
sensor4['normal']], axis=1)
type1_ = pd.concat([sensor1['type1'], sensor2['type1'], sensor3['type1'],
sensor4['type1']], axis=1)
type2_ = pd.concat([sensor1['type2'], sensor2['type2'], sensor3['type2'],
sensor4['type2']], axis=1)
type3_ = pd.concat([sensor1['type3'], sensor2['type3'], sensor3['type3'],
sensor4['type3']], axis=1)
normal_.columns = ['s1', 's2', 's3', 's4']; type1_.columns = ['s1', 's2', 's3', 's4']
type2_.columns = ['s1', 's2', 's3', 's4']; type3_.columns = ['s1', 's2', 's3', 's4']

plt.figure(figsize = (10, 4))
plt.scatter(range(0,300), normal_['s1'][:300], label="class = "+str(1), marker='o', s=5)
plt.scatter(range(0,300), type1_['s1'][:300], label="class = "+str(2), marker='o', s=5)

plt.legend(loc="lower right")
plt.xlabel("Sensor", fontsize=15)
plt.ylabel("Sensor Value", fontsize=15)
plt.show()
plt.close()

names = ['s1','s2','s3','s4']
cm = np.corrcoef(normal_[names].values.T)
sns.set(font_scale=0.8)
sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
yticklabels=names, xticklabels=names, cmap=plt.cm.Blues)

M =15
normal_s1 = np.convolve(normal_['s1'], np.ones(M), 'valid') / M 
normal_s1 = normal_s1.reshape(len(normal_s1),1)

normal_s2 = np.convolve(normal_['s2'], np.ones(M), 'valid') / M 
normal_s2 = normal_s2.reshape(len(normal_s2),1)

normal_s3 = np.convolve(normal_['s3'], np.ones(M), 'valid') / M 
normal_s3 = normal_s3.reshape(len(normal_s3),1)

normal_s4 = np.convolve(normal_['s4'], np.ones(M), 'valid') / M 
normal_s4 = normal_s4.reshape(len(normal_s4),1)

type1_s1 = np.convolve(type1_['s1'], np.ones(M), 'valid') / M 
type1_s1 = type1_s1.reshape(len(type1_s1),1)

type1_s2 = np.convolve(type1_['s2'], np.ones(M), 'valid') / M
type1_s2 = type1_s2.reshape(len(type1_s2),1)

type1_s3 = np.convolve(type1_['s3'], np.ones(M), 'valid') / M 
type1_s3 = type1_s3.reshape(len(type1_s3),1)

type1_s4 = np.convolve(type1_['s4'], np.ones(M), 'valid') / M 
type1_s4 = type1_s4.reshape(len(type1_s4),1)

type2_s1 = np.convolve(type2_['s1'], np.ones(M), 'valid') / M 
type2_s1 = type2_s1.reshape(len(type2_s1),1)

type2_s2 = np.convolve(type2_['s2'], np.ones(M), 'valid') / M 
type2_s2 = type2_s2.reshape(len(type2_s2),1)

type2_s3 = np.convolve(type2_['s3'], np.ones(M), 'valid') / M 
type2_s3 = type2_s3.reshape(len(type2_s3),1)

type2_s4 = np.convolve(type2_['s4'], np.ones(M), 'valid') / M 
type2_s4 = type2_s4.reshape(len(type2_s4),1)

type3_s1 = np.convolve(type3_['s1'], np.ones(M), 'valid') / M 
type3_s1 = type3_s1.reshape(len(type3_s1),1)

type3_s2 = np.convolve(type3_['s2'], np.ones(M), 'valid') / M 
type3_s2 = type3_s2.reshape(len(type3_s2),1)

type3_s3 = np.convolve(type3_['s3'], np.ones(M), 'valid') / M
type3_s3 = type3_s3.reshape(len(type3_s3),1)

type3_s4 = np.convolve(type3_['s4'], np.ones(M), 'valid') / M 
type3_s4 = type3_s4.reshape(len(type3_s4),1)

normal_temp = np.concatenate((normal_s1,normal_s2,normal_s3,normal_s4), axis =1)
type1_temp = np.concatenate((type1_s1,type1_s2,type1_s3,type1_s4), axis =1)
type2_temp = np.concatenate((type2_s1,type2_s2,type2_s3,type2_s4), axis =1)
type3_temp = np.concatenate((type3_s1,type3_s2,type3_s3,type3_s4), axis =1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(normal_)
normal = scaler.transform(normal_temp)
type1 = scaler.transform(type1_temp)
type2 = scaler.transform(type2_temp)
type3 = scaler.transform(type3_temp)

print(normal)
print('------------------------------------------------')
print('normal data size = ', normal.shape)

normal = normal[30000:130000][:]
type1 = type1[30000:130000][:]
type2 = type2[30000:130000][:]
type3 = type3[30000:130000][:]
print(normal)
print('------------------------------------------------')
print('normal data size = ', normal.shape)

normal_train = normal[:][:60000]; normal_valid = normal[:][60000:80000]; 
normal_test = normal[:][80000:]
type1_train = type1[:][:60000]; type1_valid = type1[:][60000:80000]; 
type1_test = type1[:][80000:]
type2_train = type2[:][:60000]; type2_valid = type2[:][60000:80000]; 
type2_test = type2[:][80000:]
type3_train = type3[:][:60000]; type3_valid = type3[:][60000:80000]; 
type3_test = type3[:][80000:]
train = np.concatenate((normal_train,type1_train,type2_train,type3_train))
valid = np.concatenate((normal_valid,type1_valid,type2_valid,type3_valid))
test = np.concatenate((normal_test,type1_test,type2_test,type3_test))
print("train data의 형태:", train.shape)
print("valid data의 형태:", valid.shape)
print(" test data의 형태:", test.shape)

train_label = np.concatenate((np.full((60000,1),0), np.full((60000,1),1), np.full((60000,1),2), np.full((60000,1),3)))
valid_label = np.concatenate((np.full((20000,1),0), np.full((20000,1),1), np.full((20000,1),2), np.full((20000,1),3)))
test_label = np.concatenate((np.full((20000,1),0), np.full((20000,1),1), np.full((20000,1),2), np.full((20000,1),3)))

idx = np.arange(train.shape[0]); np.random.shuffle(idx);
train = train[:][idx]; train_label = train_label[:][idx]
idx_v = np.arange(valid.shape[0]); np.random.shuffle(idx_v);
valid = valid[:][idx_v]; valid_label = valid_label[:][idx_v]
idx_t = np.arange(test.shape[0]); np.random.shuffle(idx_t);
test = test[:][idx_t]; test_label = test_label[:][idx_t]

x_train = torch.from_numpy(train).float()
y_train = torch.from_numpy(train_label).float().T[0]
x_valid = torch.from_numpy(valid).float()
y_valid = torch.from_numpy(valid_label).float().T[0]
x_test = torch.from_numpy(test).float()
y_test = torch.from_numpy(test_label).float().T[0]
print("변경 전")
train

print("변경 후")
x_train
