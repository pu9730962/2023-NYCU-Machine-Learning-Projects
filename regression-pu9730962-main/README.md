House price predict(Regression)
=====================================================================================================================================================================
Author:Chi-Chia Huang
--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
**資料前處理**  
![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/data.jpg)  
1.將yr_build刪除，因為房屋建造年不會直接影響價錢，而是屋齡會影像價錢，所以我改成Yr_house且裡面的值為sale_yr-yr_built    
2.將yr_renovation刪除，因為翻修的年份不會直接影響價錢，而是有沒有翻修過，所以我改成renovation，裡面是one-hot vector，有整修就是[1,0]，沒整修就是[0,1]  
3.將zipcode刪除，因為郵遞區號不會直接影響價錢，必須要去看郵遞區號對應到美國哪一區並轉成one-hot vector，但在本次我先將其刪除  
下圖為修改過後的feature  
![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/processing_data.jpg)    


**程式碼**  
![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code1.jpg)  
上圖為本次有用到的函式庫  

![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code2.jpg)  
前三列分別將train.csv、valid.csv、test.csv讀進來並利用.valuse轉成二維Array，後三列裡用Scikit-learn函式將價錢以外的features個別正規化，以免各個Feature的尺度不一導致訓練效果降低。  

![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code3.jpg)  
利用三層線性隱藏層網路，每層後面都利用Batchnormalization()、Dropout()、PRelu()。Batchnormalization()可以避免每層輸出後各Feature尺度不一的狀況，達到加快學習的效果，Dropout()則是避免網路overfitting，PRelu()是使網路變成非線性，且PRelu其X軸小於0的斜率是可以學習的。  

![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code4.jpg)  
將驗證模式寫成函式方便呼叫，.eval()專門拿來用在驗證以及測試階段，會自動忽略Batchnormalization()和Dropout()。  

![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code5.jpg)  
將訓練模式寫成函式  
第五、六列拿來打包輸入資料和其對應的價錢，並且可按照設定的Batcht漸進的抽出資料  
第八、九、十列是不同的optimizer可以選擇使用   
第十四列-三十列為訓練網路階段並在每次更新後丟入驗證集並計算loss，且保存讓驗證集loss最小的那組weights  
第三十四列-四十列為畫出每次跌代後，訓練集和測試集的loss下降趨勢，觀察是否overfitting  

![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code6.jpg)  
將測試模式寫成函式，使用.eval()模式，並且讀取在訓練時保存的最佳wwights來預測房價  

![image](https://github.com/MachineLearningNTUT/regression-pu9730962/blob/main/Picture/code7.jpg)  
在本區主程式內實際執行，呼叫train()、test()函式，最後的預測答案是由10次predict的結果取平均
第九列-第十三列是將predict的結果轉成dataframe形式再轉成csv'檔
