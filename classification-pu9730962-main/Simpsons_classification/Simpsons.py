import Simpsons_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from torch.optim import SGD,Adam,AdamW
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet152_Weights
from torchvision import transforms

class SimpsonsNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.Resnet152=models.resnet152(pretrained=True)
        self.Resnet152=models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.Resnet152.fc=nn.Sequential(
           nn.Linear(2048,1024),
            nn.BatchNorm1d(1024,eps=1e-05, momentum=0.1, affine=True),
            nn.PReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024,50)
                                        )
    def forward(self,input):
        return self.Resnet152(input)
    
def validation(net,device):
    with torch.no_grad():
        net.eval()
        valid_dataloader=DataLoader(Simpsons_dataset.valid(),batch_size=64,shuffle=True)
        correct_totalnum=0
        valid_predictlist=[]
        valid_labellist=[]
        for valid_data,valid_label in valid_dataloader:
            valid_labellist.extend(valid_label.tolist())
            valid_data=valid_data.to(device)
            valid_label=valid_label.to(device)
            output=net(valid_data)
            outputmax=torch.argmax(output,dim=1)
            valid_predictlist.extend(outputmax.cpu().tolist())
            correct_num=len(outputmax[outputmax==valid_label])
            correct_totalnum=correct_totalnum+correct_num
            torch.cuda.empty_cache()
    return correct_totalnum,valid_predictlist,valid_labellist

def train(device,Batch=30,Epoch=20,lr=0.001):
    net=SimpsonsNet()
    param_list=[]
    for param_name,param in net.named_parameters():
        param_list.append(param_name)
    net.to(device)
    net.train()
    train_dataloader=DataLoader(Simpsons_dataset.train(),batch_size=Batch,shuffle=True)
    Loss_func=nn.CrossEntropyLoss()
    for param_name,param in net.named_parameters():
        if param_name not in param_list[len(param_list)-7:]:
            param.requires_grad=False
    train_accuracylist=[]
    train_losslist=[]
    valid_accuracylist=[]
    best_valaccuracy=0
    for epoch in range(Epoch):
        correct_totalnum=0
        if epoch == 6:
            lr=0.0005
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-37:len(param_list)-7]:
                    param.requires_grad=True
        elif epoch ==8:
            lr=0.0002
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-117:len(param_list)-37]:
                    param.requires_grad=True
        elif epoch ==10:
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-198:len(param_list)-117]:
                    param.requires_grad=True
        elif epoch ==12:
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-279:len(param_list)-198]:
                    param.requires_grad=True
        elif epoch ==14:
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-364:len(param_list)-279]:
                    param.requires_grad=True
        elif epoch ==16:
            lr=0.0001
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-439:len(param_list)-364]:
                    param.requires_grad=True
        elif epoch ==18:
            for param_name,param in net.named_parameters():
                if param_name in param_list[len(param_list)-469:len(param_list)-439]:
                    param.requires_grad=True
                    
        parameter=filter(lambda p:p.requires_grad,net.parameters())
        #optimizer=SGD(parameter, lr=lr, momentum=0.9, nesterov = True)
        #optimizer = Adam(parameter, lr=lr,eps=1e-08)
        optimizer=AdamW(parameter, lr=lr, amsgrad=True)
        # scheduler=exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3)
        for i,(train_data,train_label) in enumerate(train_dataloader):
            train_data=train_data.to(device)
            train_label=train_label.to(device)
            output=net(train_data)
            loss=Loss_func(output,train_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputmax=torch.argmax(output,dim=1)
            correct_num=len(outputmax[outputmax==train_label])
            correct_totalnum=correct_totalnum+correct_num
            print(f'Batch{i+1} of loss:{loss.item()}')
        train_accuracylist.append(correct_totalnum/76929)
        train_losslist.append(loss.item())
        valid_correct_totalnum,valid_predictlist,valid_labellist=validation(net,device)
        valid_accuracy=valid_correct_totalnum/20000
        if valid_accuracy>best_valaccuracy:
            best_valaccuracy=valid_accuracy
            valid_best_predictlist=valid_predictlist
            torch.save(net.state_dict(),'Simpsons_weight.pth')
        valid_accuracylist.append(valid_accuracy)
        print(f'train_accuracy of epoch{epoch+1} is {correct_totalnum/76929}')
        print(f'Validation_accuracy of epoch{epoch+1} is {valid_accuracy}')
        torch.cuda.empty_cache()
    plt.figure(figsize=(20,10))
    plt.plot(range(Epoch),train_accuracylist,'g.-',label='train_accuracy')
    plt.plot(range(Epoch),train_losslist,'r.-',label='train_loss')
    plt.plot(range(Epoch),valid_accuracylist,'b.-',label='validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend(bbox_to_anchor=(1.01,0),loc=3,borderaxespad=0)
    plt.savefig('accuracy.png')
    plt.pause(5)
    return valid_best_predictlist,valid_labellist

def test(device): 
    with torch.no_grad():
        test_dataloader=DataLoader(Simpsons_dataset.test(),batch_size=64)
        Net=SimpsonsNet()
        Net.to(device)
        Net.load_state_dict(torch.load('Simpsons_weight.pth'))
        Net.eval()
        predict_list=[]
        for test_data in test_dataloader:
            test_data=test_data.to(device)
            output=Net(test_data)
            outputmax=torch.argmax(output,dim=1)
            name_dict={0:'abraham_grampa_simpson',1:'agnes_skinner',2:'apu_nahasapeemapetilon',3:'barney_gumble',4:'bart_simpson',
                       5:'brandine_spuckler',6:'carl_carlson',7:'charles_montgomery_burns',8:'chief_wiggum',9:'cletus_spuckler',
                       10:'comic_book_guy',11:'disco_stu',12:'dolph_starbeam',13:'duff_man',14:'edna_krabappel',15:'fat_tony',
                       16:'gary_chalmers',17:'gil',18:'groundskeeper_willie',19:'homer_simpson',20:'jimbo_jones',21:'kearney_zzyzwicz',
                       22:'kent_brockman',23:'krusty_the_clown',24:'lenny_leonard',25:'lionel_hutz',26:'lisa_simpson',27:'lunchlady_doris',
                       28:'maggie_simpson',29:'marge_simpson',30:'martin_prince',31:'mayor_quimby',32:'milhouse_van_houten',33:'miss_hoover',
                       34:'moe_szyslak',35:'ned_flanders',36:'nelson_muntz',37:'otto_mann',38:'patty_bouvier',39:'principal_skinner',
                       40:'professor_john_frink',41:'rainier_wolfcastle',42:'ralph_wiggum',43:'selma_bouvier',44:'sideshow_bob',
                       45:'sideshow_mel',46:'snake_jailbird',47:'timothy_lovejoy',48:'snake_jailbird',49:'troy_mcclure'}
            for i in range(len(outputmax)):
                predict_list.append(name_dict[outputmax[i]])
            torch.cuda.empty_cache()
        print(len(predict_list))
        accuracy_dataframe={"id":list(range(1,10792)),
                            "character":predict_list}
        accuracy_dataframe=pd.DataFrame(accuracy_dataframe)
        print(accuracy_dataframe)
        accuracy_dataframe.to_csv("Submission.csv",index=False)

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    valid_best_predictlist,valid_labellist=train(device)
    test(device)
    plt.figure(figsize=(42,40))
    confusionmatrix=confusion_matrix(valid_labellist,valid_best_predictlist)
    confusionmatrix=confusionmatrix/confusionmatrix.astype(float).sum(axis=1).reshape(50,1)
    classes=["abrahamgrampasimpson","agnesskinner","apunahasapeemapetilon","barneygumble","bartsimpson","brandinespuckler","carlcarlson","charlesmontgomeryburns","chiefwiggum","cletusspuckler","comicbookguy","discostu","dolphstarbeam","duffman","ednakrabappel","fattony","garychalmers","gil","groundskeeperwillie","homersimpson","jimbojones","kearneyzzyzwicz","kentbrockman","krustytheclown","lennyleonard","lionelhutz","lisasimpson","lunchladydoris","maggiesimpson","margesimpson","martinprince","mayorquimby","milhousevanhouten","misshoover","moeszyslak","nedflanders","nelsonmuntz","ottomann","pattybouvier","principalskinner","professorjohnfrink","rainierwolfcastle","ralphwiggum","selmabouvier","sideshowbob","sideshowmel","snakejailbird","timothylovejoy","troymcclure","waylon_smithers"]
    confusionmatrix=pd.DataFrame(confusionmatrix,classes,classes)
    sns.heatmap(confusionmatrix,annot=True)
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig('heatmap.png')
    plt.pause(5)
    
    #可視畫參數圖
    weight=torch.load('Simpsons_weight.pth')
    for i in range(64):
        img_torch=weight['Resnet152.conv1.weight'][i,:,:,:]
        img_PIL=transforms.ToPILImage()(img_torch)
        plt.subplot(8,8,i+1)
        plt.imshow(img_PIL)
        plt.axis('off')
    plt.suptitle('Visualization')
    plt.savefig('Visualization1.jpg')