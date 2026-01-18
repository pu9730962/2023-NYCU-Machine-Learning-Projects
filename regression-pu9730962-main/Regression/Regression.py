import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
from torch.optim import RMSprop,Adam,SGD
from sklearn import preprocessing
import numpy

train_dataframe=pd.read_csv('train-v3.csv').values
valid_dataframe=pd.read_csv('valid-v3.csv').values
test_dataframe=pd.read_csv('test-v3.csv',header=None).drop(index=0,columns=0).values

train_standardization_input=torch.tensor(preprocessing.scale(train_dataframe[:,1:]))
valid_standardization_input=torch.tensor(preprocessing.scale(valid_dataframe[:,1:]))
test_standardization_input=torch.tensor(preprocessing.scale(test_dataframe))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Sequential(
            nn.Linear(20,500),
            nn.BatchNorm1d(500,eps=1e-05,momentum=0.1,affine=True),
            nn.Dropout(p=0.25),
            nn.PReLU(),
            nn.Linear(500,500),
            nn.BatchNorm1d(500,eps=1e-05,momentum=0.1,affine=True),
            nn.Dropout(p=0.25),
            nn.PReLU(),
            nn.Linear(500,1),
            nn.PReLU()
            )
    def forward(self,input):
        return self.linear(input)

def validation(device,net,Loss):
    global valid_standardization_input
    global valid_dataframe
    with torch.no_grad():
        valid_input=valid_standardization_input.to(device)
        valid_label=torch.tensor(valid_dataframe[:,:1]).to(device)
        net.eval()
        valid_output=net(valid_input.float())
        valid_loss=Loss(valid_output,valid_label).item()
    return valid_loss

def train(device,Batch=325,Epoch=500,lr=0.0005):
    global train_standardization_input,train_dataframe
    net=Net()
    net.to(device)
    train_dataset=TensorDataset(train_standardization_input,torch.tensor(train_dataframe[:,:1]))
    train_dataloader=DataLoader(train_dataset,batch_size=Batch,shuffle=True,drop_last=True)
    Loss=nn.L1Loss()
    optimizer=RMSprop(net.parameters(),lr=lr,weight_decay=1e-8,momentum=0.99)
    #optimizer=SGD(net.parameters(),lr=lr,momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr,eps=1e-08)
    train_losslist=[]
    valid_losslist=[]
    best_loss=float("inf")
    for epoch in range(Epoch):
        for train_input,train_label in train_dataloader:
            net.train()
            train_input=train_input.to(device).float()
            train_label=train_label.to(device).float()
            optimizer.zero_grad()
            train_output=net(train_input)
            train_loss=Loss(train_output,train_label)
            train_loss.backward()
            optimizer.step()
            train_losslist.append(train_loss.item())

            valid_loss=validation(device,net,Loss)
            print(f'validloss is {valid_loss}')
            valid_losslist.append(valid_loss)
            if valid_loss<best_loss:
                best_loss=valid_loss
                torch.save(net.state_dict(),'Regression_weight.pth')      
        torch.cuda.empty_cache()    
    print(f'The lowest of validationloss is {best_loss}')
    
    plt.figure(figsize=(20,10))
    plt.plot(range(Epoch*40),train_losslist,'g.-',label='train_loss')
    plt.plot(range(Epoch*40),valid_losslist,'r.-',label='valid_loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.03,0),loc=3,borderaxespad=0)
    plt.pause(5)
   
def test():
    global test_standardization_input
    with torch.no_grad():
        net=Net()
        net.to(device)
        net.load_state_dict(torch.load('Regression_weight.pth'))
        net.eval()
        test_standardization_input=test_standardization_input.to(device)
        test_output=net(test_standardization_input.float())
    return test_output

if __name__ =="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_tensor=torch.zeros(6485).to(device)
    for i in range(10):
        train(device)
        test_output=test().squeeze(dim=1)
        loss_tensor=loss_tensor+test_output
    avgloss=(loss_tensor/10).cpu().numpy().tolist()
    loss_dataframe={"id":list(range(1,6486)),
                    "price":avgloss}
    loss_dataframe=pd.DataFrame(loss_dataframe)
    print(loss_dataframe)
    loss_dataframe.to_csv("Submission.csv",index=False)