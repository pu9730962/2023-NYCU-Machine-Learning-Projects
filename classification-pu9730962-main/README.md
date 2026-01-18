Simpsons classification
=
Author:Chi-Chia Huang
-
**資料前處理**  
首先將train_data資料夾下的照片名稱利用csv檔分類成train_img.csv和valid_img.csv，然後再創建train_label.csv和valid_label.csv，此外也將test_data資料夾下的照片名稱寫入test.csv檔，以方便資料前處理  
  
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/data_processing.jpg) 
25列-31列將照片以一定機率做圖像轉換，32列是將PIL類型照片轉為Tensor並且將pixel的值範圍從[0,255]轉為[0,1]，33再將RGB三個維度的pixel值的平均拉到[0.485, 0.456, 0.406]，標準差為[0.229, 0.224, 0.225]，因為Pytorch提供的Resnet pretrained-weight是在ImageNet資料集下訓練，其照片的RGB平均和標準差分別為[0.485, 0.456, 0.406]、[0.229, 0.224, 0.225]，所以如果將自己的資料標準化到此範圍內，會比較容易收斂!
驗證集以及測試集也利用相同方法讀取照片，並且做transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])的轉換
  
**主程式**  
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/Net.jpg)  
19列為使用Pytorch提供ResNet152的pretrained-weight，21列-25列為更改最後ResNet152最後的全連接層的神經元數目並且再多加一層全連接層  
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/train.jpg)  
58列-60列，將ResNet152 pretrained好的層先凍結訓練，先行訓練最後兩層的全連接層，從67列開始，慢慢的解凍前面層並且調低learning rate，切記不能一次解凍太多層，不然會導致網路很難收斂。  

以上是本次最重要的部分，其它程式碼在Simpsons.py可見，大部分的架構和Project1相似，則不多加詳述，以下附上train loss、train accuracy以及validation accuracy趨勢圖   
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/accuracy.png)   
   
**Heatmap以及第一層捲機參數可視化圖**
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/heatmap_code.jpg) 
176列-177列利用sklearn.metrics import confusion_matrix，先計算出confusion-matrix，再利用seaborn轉為heatmap，以下為結果  
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/heatmap.png)   
   
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/Visualization_code.jpg)  
將預先存好的model參數讀進來，可以發現它為一個OrderedDict，利用key:'Resnet152.conv1.weight'可以找出第一層捲機的權重(channel=64)，並將64個權重Tensor轉為PIL照片並顯示出來，以下為結果  
![image](https://github.com/MachineLearningNTUT/classification-pu9730962/blob/main/Picture/Visualization.jpg)  
