import pandas as pd
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader,Dataset
import re
sentence_ZH_train = []
sentence_ZH_valid=[]
sentence_ZH_test=[]
sentence_TL =[]
sentence_TL_deIN = []
sentence_TL_deIN_valid=[]
sentence_TL_deOP = []
sentence_TL_deOP_valid = []

ZH_dataframe=pd.read_csv("train-ZH.csv",encoding="utf-8").drop(["id"],axis=1).values
ZH_dataframe_train=np.squeeze(ZH_dataframe)[:int(len(ZH_dataframe)*0.9)]                  #切割train資料數量
ZH_dataframe_valid=np.squeeze(ZH_dataframe)[int(len(ZH_dataframe)*0.9):]
# print(len(ZH_dataframe))
TL_dataframe=pd.read_csv("train-TL.csv",encoding="utf-8").drop(["id"],axis=1).values
TL_dataframe_train=np.squeeze(TL_dataframe)[:int(len(TL_dataframe)*0.9)]
TL_dataframe_valid=np.squeeze(TL_dataframe)[int(len(TL_dataframe)*0.9):]
# print(len(TL_dataframe))
ZH_dataframe_test=pd.read_csv("test-ZH-nospace.csv",encoding="utf-8").drop(["id"],axis=1).values
ZH_dataframe_test=np.squeeze(ZH_dataframe_test)  


for ZH_string in ZH_dataframe_train:
    tem_list=[]
    # sentence_ZH_train.append(ZH_string.strip().split())
    for word in ZH_string.strip('\s').split():
        tem_list.extend(list(word))
    sentence_ZH_train.append(tem_list)
    
for ZH_string in ZH_dataframe_valid:
    tem_list=[]
    for word in ZH_string.strip('\s').split():
        tem_list.extend(list(word))
    sentence_ZH_valid.append(tem_list)
    
for TL_string in TL_dataframe_train:
    sentence_TL.append(['START']+list(TL_string)+['STOP'])   #拿來做字典用
    sentence_TL_deIN.append(['START']+list(TL_string))
    sentence_TL_deOP.append(list(TL_string)+['STOP'])

for TL_string in TL_dataframe_valid:
    sentence_TL_deIN_valid.append(['START']+list(TL_string))
    sentence_TL_deOP_valid.append(list(TL_string)+['STOP'])

for ZH_string in ZH_dataframe_test:
    sentence_ZH_test.append(list(ZH_string))

    
word_count_ZH=Counter([word for sent in sentence_ZH_train for word in sent])
word_count_TL=Counter([word for sent in sentence_TL for word in sent])
ZH_mostcommonword=word_count_ZH.most_common(len(word_count_ZH))        #會變成[('字','頻率').......]
TL_mostcommonword=word_count_TL.most_common(len(word_count_TL))
word_dict_ZH={w[0]:index + 2 for index,w in enumerate(ZH_mostcommonword)}  #常用字字典{'字':index, ....}
word_dict_TL= {w[0]:index + 2 for index,w in enumerate(TL_mostcommonword)} 
word_dict_ZH['UNK'] = 1 # 在新的字典word_dict_ZH增加一个字符UNK
word_dict_ZH['PAD'] = 0 # 在新的字典word_dict_ZH增加一个字符PAD
word_dict_TL['UNK'] = 1 # 在新的字典word_dict_TL增加一个字符UNK
word_dict_TL['PAD'] = 0 # 在新的字典word_dict_TL增加一个字符PAD
src_vocab_size = len(word_dict_ZH)
tgt_vocab_size = len(word_dict_TL)   
index_dict_ZH={v:k for k,v in word_dict_ZH.items()}
index_dict_TL={v:k for k,v in word_dict_TL.items()}

code_num_ZH = [[word_dict_ZH.get(word,1) for word in sent] for sent in sentence_ZH_train] # 从中文句子中獲得句子，再從每条句子獲得每个词，再进行對應编码替换,get()從字典中找key，如果找不到則返回默認值
code_num_ZH_valid = [[word_dict_ZH.get(word,1) for word in sent] for sent in sentence_ZH_valid]
code_num_ZH_test = [[word_dict_ZH.get(word,1) for word in sent] for sent in sentence_ZH_test]
code_num_TL_deIN=[[word_dict_TL.get(word,1) for word in sent] for sent in sentence_TL_deIN]
code_num_TL_deIN_valid=[[word_dict_TL.get(word,1) for word in sent] for sent in sentence_TL_deIN_valid]
code_num_TL_deOP=[[word_dict_TL.get(word,1) for word in sent] for sent in sentence_TL_deOP]
code_num_TL_deOP_valid=[[word_dict_TL.get(word,1) for word in sent] for sent in sentence_TL_deOP_valid]

def Padding(cod_num,mode):
    Lenth=[len(sent) for sent in cod_num]
    max_lenth=max(Lenth)
    # print(max_lenth)
    if mode=='ZH':
        for i in range(len(Lenth)):   
            padding_len=max_lenth-Lenth[i]   
            list_0=[0]*padding_len
            cod_num[i]=cod_num[i]+list_0
        return cod_num
    elif mode=='TL_IN':
        for i in range(len(Lenth)):
            padding_len=max_lenth-Lenth[i]   #要補0的數量
            list_0=[0]*padding_len
            cod_num[i]=cod_num[i]+list_0
        return cod_num
    elif mode=='TL_OP':
        for i in range(len(Lenth)):
            padding_len=max_lenth-Lenth[i]   #要補0的數量
            list_0=[0]*padding_len
            cod_num[i]=cod_num[i]+list_0
        return cod_num

enc_inputs=torch.LongTensor(Padding(code_num_ZH,'ZH'))
enc_inputs_valid=torch.LongTensor(Padding(code_num_ZH_valid,'ZH'))
enc_inputs_test=torch.LongTensor(Padding(code_num_ZH_test,'ZH'))
dec_inputs=torch.LongTensor(Padding(code_num_TL_deIN,'TL_IN'))
dec_inputs_valid=torch.LongTensor(Padding(code_num_TL_deIN_valid,'TL_IN'))
dec_outputs =torch.LongTensor(Padding(code_num_TL_deOP,'TL_OP'))
dec_outputs_valid =torch.LongTensor(Padding(code_num_TL_deOP_valid,'TL_OP'))
TL_MAXlen_valid=dec_inputs_valid.shape[-1]

def globalvariable(mode):
    if mode=='train':
        return enc_inputs,dec_inputs,dec_outputs
    elif mode=='valid':
        return enc_inputs_valid,dec_inputs_valid,dec_outputs_valid
    elif mode=='test':
        return enc_inputs_test

class MyDataSet(Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

class MyDataSet_test(Dataset):
  def __init__(self, enc_inputs):
    super(MyDataSet_test, self).__init__()
    self.enc_inputs = enc_inputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx]

# train_loader = DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 128, True)
# for a,b,c in train_loader:
#     print(a.shape)