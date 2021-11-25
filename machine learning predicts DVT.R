
###### machine learning predicts DVT ######

#########  一、单因素和Lasso,取交集  ######### 

# DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC
# DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer

library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

## tune cart 无DD，cp=0.001,AUC=0.647；
set.seed(7)
cartgrid <- expand.grid(.cp=seq(0.001, 0.150, by=0.004))

cart_1 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data=dataset, method="rpart", metric="ROC", tuneGrid=cartgrid, trControl=fitControl)
print(cart_1)
plot(cart_1)

## tune cart 有DD，cp=0.001,AUC=0.713；
set.seed(7)
cartgrid <- expand.grid(.cp=seq(0.001, 0.150, by=0.004))
cart_2 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer, data=dataset, method="rpart", metric="ROC", tuneGrid=cartgrid, trControl=fitControl)
print(cart_2)
plot(cart_2)

## tune RF 无DD，mtry = 1,AUC=0.716；
library(import)

set.seed(7)
rfgrid <- expand.grid(.mtry=c(1, 2, 3, 4, 11))
RF_1 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
              data=dataset, method="parRF", metric="ROC", tuneGrid=rfgrid, trControl=fitControl)
print(RF_1)
plot(RF_1)

## tune RF 有DD，mtry = 1,AUC=0.768；
set.seed(7)
rfgrid <- expand.grid(.mtry=c(1, 2, 3, 4, 11))
RF_2 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
              data=dataset, method="parRF", metric="ROC", tuneGrid=rfgrid, trControl=fitControl)
print(RF_2)
plot(RF_2)

## tune svm 无DD，sigma = 0.025,C= 3, AUC=0.670；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_1 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, 
               data=dataset, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_1)
plot(svm_1)

## tune svm 有DD，sigma = 0.025,C= 2, AUC=0.743；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_2 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
               data=dataset, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_2)
plot(svm_2)

# 对连续性变量标准化后再看结果

## tune svm 无DD，sigma = 0.025,C= 3, AUC=0.670；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_3 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, 
               data=dataset.svm, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_3)
plot(svm_3)

## tune svm 有DD，sigma = 0.025,C= 2, AUC=0.743；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_4 <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
               data=dataset.svm, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_4)
plot(svm_4)

#######  跟是否将变量标准化，结果无差异

dataset$DVT <- dataset01$DVT
dataset$DVT <- ifelse(dataset$DVT==1,"有深静脉血栓","无深静脉血栓")

#加载包
library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

# 数据集分十折
set.seed(7)
folds <- createFolds(y=dataset[,59],k=10)

# 给定参数
max1=0; max2=0; max3=0;
max4=0; max5=0; max6=0;
num1=0; num2=0; num3=0;
num4=0; num5=0; num6=0
auc_value1 <-as.numeric()
auc_value2 <-as.numeric()
auc_value3 <-as.numeric()
auc_value4 <-as.numeric()
auc_value5 <-as.numeric()
auc_value5.new <-as.numeric()
auc_value6 <-as.numeric()

# 训练集建模和测试集调优
# LDA 不联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre1 <- lda(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data=fold_train)
  fold_predict1 <- predict(fold_pre1, newdata=fold_test, type="response") 
  fold_predict1 <- as.data.frame(fold_predict1)$posterior.有深静脉血栓
  auc_value1 <- append(auc_value1,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict1)))
  
}  

# LR不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre2 <- glm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                   data=fold_train, family ="binomial")  
  
  fold_predict2 <- predict(fold_pre2,type='response', newdata=fold_test)  
  
  
  auc_value2 <- append(auc_value2,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict2)))
  
} 


# cart不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre3 <- rpart(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data=fold_train, control = rpart.control(cp="0.001"))  
  
  fold_predict3 <- predict(fold_pre3,type='class', newdata=fold_test)  
  
  auc_value3 <- append(auc_value3,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict3)))
  
} 

###### 换一种方式可视化分类树模型   ######

# install.packages("visNetwork")
# install.packages("sparkline")
library(visNetwork)
library(sparkline)
win.graph(width=4.875, height=2.5,pointsize=8)

fold_pre3_1 <- rpart(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC, data=fold_train, control = rpart.control(cp="0.001"))  
plotcp(fold_pre3_1)
visTree(fold_pre3_1,main = "classification tree model without D-Dimer",height = "400px")

fold_pre15_1 <- rpart(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC+Plaster+DDimer, data=fold_train, control = rpart.control(cp="0.001"))  
plotcp(fold_pre15_1)
visTree(fold_pre15_1,main = "classification tree model with D-Dimer",height = "400px")

# RF 不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre4 <- randomForest(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                            data=fold_train, mtry=9, importance=TRUE, ntree=1000)  
  
  fold_predict4 <- predict(fold_pre4,type='response', newdata=fold_test)  
  
  auc_value4 <- append(auc_value4,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict4)))
} 

# SVM 不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre5 <- ksvm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                    data=fold_train, sigma=0.025, C=5, prob.model = TRUE)  
  
  fold_predict5 <- predict(fold_pre5,type='response', newdata=fold_test)  
  
  auc_value5 <- append(auc_value5,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict5)))
} 


# Khorana
# 数据集分十折

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  auc_value6 <- append(auc_value6,auc(as.numeric(fold_test$DVT),as.numeric(fold_test$K_Score)))
} 


# 确定最优模型是第几折
num1<-which.max(auc_value1)
num2<-which.max(auc_value2)
num3<-which.max(auc_value3)
num4<-which.max(auc_value4)
num5<-which.max(auc_value5)
num5.new <-which.max(auc_value5.new)
num6 <-which.max(auc_value6)
print(auc_value1)
print(auc_value2)
print(auc_value3)
print(auc_value4)
print(auc_value5)
print(auc_value5.new)
print(auc_value6)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 不联合D二聚体

fold_test <- dataset[folds[[num1]],]   

fold_train <- dataset[-folds[[num1]],]

fold_pre7 <- lda(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                 data=fold_train) 

fold_predict7 <- predict(fold_pre7, newdata=fold_test, type="response")
fold_predict7 <- as.data.frame(fold_predict7)$posterior.有深静脉血栓

roc7 <- roc(as.numeric(fold_test$DVT),fold_predict7)
plot(roc7, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_7 <- ifelse(fold_predict7 >= 0.291,2,1)
table(pred_7, fold_test$DVT)
(51+9)/72   ###  0.833  ###

plot(fold_pre7, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre7$scaling ,file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 3/LDA回归系数无DD.csv")

# LR 不联合D二聚体
fold_test <- dataset2[folds[[num2]],]   

fold_train <- dataset2[-folds[[num2]],]

fold_pre8 <- glm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                 data=fold_train, family ="binomial")  

fold_predict8 <- predict(fold_pre8,type='response', newdata=fold_test)  

roc8 <- roc(as.numeric(fold_test$DVT),fold_predict8)
plot(roc8, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_7 <- ifelse(fold_predict7 >= 0.288,2,1)
table(pred_7, fold_test$DVT)
(50+10)/72   ###  0.833  ###

############ 绘制交互式列线图   ############

dataset2$Hb <- dataset2$Hb_1
dataset2$PLT <- dataset2$PLT_1
dataset2$WBC <- dataset2$WBC_1
dataset2$D_Dimer <- 2^dataset2$log2DDimer
dataset2$CCI <- 2^dataset2$log2CCI
dataset2$LOS <- 2^dataset2$log2LOS

fold_test <- dataset2[folds[[num2]],] 
fold_train <- dataset2[-folds[[num2]],]
fold_pre8 <- glm(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC,
                 data=fold_train, family ="binomial")  

library(regplot)
nomogram_for_none_DDimer_LR_model <- fold_pre8
regplot(nomogram_for_none_DDimer_LR_model,observation=fold_test[1,]) 


fold_test <- dataset2[folds[[num.2]],] 
fold_train <- dataset2[-folds[[num.2]],]
fold_pre14 <- glm(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC+Plaster+D_Dimer,
                  data=fold_train, family ="binomial")  

library(regplot)
nomogram_for_DDimer_LR_model <- fold_pre14
regplot(nomogram_for_DDimer_LR_model,observation=fold_test[1,]) 


############ 绘制网页计算器   ############

library(glmnet)
library(MASS)
library(survival)

library(rms)
library(magrittr)
library(DynNom) # 加载失败
library(packrat)
library(rsconnect)

install.packages("colorspace", depend=TRUE)   # 补充这个包也无效

# 重启R后成功加载

DynNom(nomogram_for_DDimer_LR_model, covariate = "numeric", DNtitle = "A Dynamic nomogram of Cancer-associated DVT")

DynNom(nomogram_for_DDimer_LR_model, covariate = "numeric", DNtitle = "A Dynamic nomogram of Cancer-associated DVT with DDimer", DNxlab = "Probability of Cancer-associated DVT")

# 做出动态列线图以后上传到网页

DNbuilder(nomogram_for_DDimer_LR_model, covariate = "numeric", DNtitle = "A Dynamic nomogram of Cancer-associated DVT with DDimer", DNxlab = "Probability of Cancer-associated DVT")

DNbuilder(nomogram_for_DDimer_LR_model, DNtitle = "A Dynamic nomogram of Cancer-associated DVT with DDimer", DNxlab = "Probability of Cancer-associated DVT")
#### use your own name, token, and secret get from https://www.shinyapps.io/
rsconnect::setAccountInfo(name='webcalculatorofcancerassociateddvt',
                          token='33242B65A76F007EC0D1CF462F7996D6',
                          secret='Z+mUU1IhR6/EZUUcpVQsTJKaHh6erGpnyP1ZLZgr')
rsconnect::deployApp('DynNomapp')

# CART 不联合D二聚体
fold_test <- dataset[folds[[num3]],]   

fold_train <- dataset[-folds[[num3]],]

fold_pre9 <- rpart(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                   data=fold_train, control = rpart.control(cp="0.001"))  

fold_predict9 <- predict(fold_pre9,type='class', newdata=fold_test)  

roc9 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict9))
plot(roc9, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict9, fold_test$DVT)
(48+6)/72   ###  0.750  ###

# CART 不联合D二聚体，无log2
library(foreign)
dataset_train <- read.spss("训练集和测试集.sav")
dataset_train <- as.data.frame(dataset_train)
dataset_validation <- read.spss("验证集.sav")
dataset_validation <- as.data.frame(dataset_validation)

dataset_train$LOS <- round(exp(dataset_train$log2LOS),0)
dataset_train$CCI <- round(exp(dataset_train$log2CCI),0)
dataset_train$WBC <- dataset_train$WBC_1
dataset_train$PLT <- dataset_train$PLT_1
dataset_train$Hb <- dataset_train$Hb_1
dataset_train$DDimer <- round(dataset_train$DDimer_1,0)

dataset_validation$LOS <- round(exp(dataset_validation$log2LOS),0)
dataset_validation$CCI <- round(exp(dataset_validation$log2CCI),0)
dataset_validation$WBC <- round(dataset_validation$WBC_1,2)
dataset_validation$PLT <- round(dataset_validation$PLT_1,0)
dataset_validation$Hb  <- round(dataset_validation$Hb_1,0)
dataset_validation$DDimer <-  round(dataset_validation$DDimer_1,0)

fold_test <- dataset_train[folds[[num3]],]   

fold_train <- dataset_train[-folds[[num3]],]

fold_pre9 <- rpart(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC,
                   data=fold_train, control = rpart.control(cp="0.001"))  
# Error in eval(predvars, data, env) : object 'WBC' not found
fold_predict9 <- predict(fold_pre9,type='class', newdata=fold_test)  

roc9 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict9))
plot(roc9, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

### AUC 0.616 ###

table(fold_predict9, fold_test$DVT)
(48+6)/72   ###  0.750  ###

plot(fold_pre9)
text(fold_pre9, use.n = F)

# RF 不联合D二聚体

fold_test <- dataset[folds[[num4]],]   

fold_train <- dataset[-folds[[num4]],]

fold_pre10 <- randomForest(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                           data=fold_train, mtry=9, importance=TRUE, ntree=1000)  

fold_predict10 <- predict(fold_pre10,type='response', newdata=fold_test)  

roc10 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict10))
plot(roc10, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict10, fold_test$DVT)
(51+5)/72   ###  0.778  ###

# RF 不联合D二聚体，无log2

fold_test <- dataset_train[folds[[num4]],]   

fold_train <- dataset_train[-folds[[num4]],]

fold_pre10 <- randomForest(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC,
                           data=fold_train, mtry=9, importance=TRUE, ntree=1000)  

fold_predict10 <- predict(fold_pre10,type='response', newdata=fold_test)  

roc10 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict10))
plot(roc10, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")
# 0.674

table(fold_predict10, fold_test$DVT)
(51+5)/72   ###  0.778  ###

varImpPlot(fold_pre10)

# SVM 不联合D二聚体

fold_test <- dataset[folds[[num5]],]   

fold_train <- dataset[-folds[[num5]],]

fold_pre11 <- ksvm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                   data=fold_train, sigma=0.025, C=5, prob.model = TRUE)  

fold_predict11 <- predict(fold_pre11,type='response', newdata=fold_test)  

roc11 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict11))
plot(roc11, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict11, fold_test$DVT)
(53+6)/72   ###  0.819  ###

###### 用主成分方法可视化SVM ######
library(kernlab);library(e1071);library(ggplot2)
fold_test <- dataset[folds[[num5]],]   
fold_train <- dataset[-folds[[num5]],]
fold_train$Age <- as.numeric(fold_train$Age)
fold_train$Chemotherapy <- as.numeric(fold_train$Chemotherapy)
fold_train$Port_cath <- as.numeric(fold_train$Port_cath)
fold_train$NSAID <- as.numeric(fold_train$NSAID)
fold_train$Bed <- as.numeric(fold_train$Bed)
fold_train$VTEHistory <- as.numeric(fold_train$VTEHistory)
fold_train$Plaster <- as.numeric(fold_train$Plaster)

pcaseed1 <- princomp(~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC,data=fold_train, cor = TRUE)
pcaseed2 <- princomp(~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+WBC+Plaster+DDimer_1,data=fold_train, cor = TRUE)

seed_score1 <- as.data.frame(pcaseed1$scores[,1:2])
seed_score1$DVT <- fold_train$DVT
seed_score2 <- as.data.frame(pcaseed2$scores[,1:2])
seed_score2$DVT <- fold_train$DVT

ggplot(seed_score1,aes(x=Comp.1,y= Comp.2,colour = DVT,shape = DVT))+
  geom_point()+theme(legend.position = "right")+
  labs(x = "主成分得分1", y = "主成分得分2", title = "主成分降维散点图" )
ggplot(seed_score2,aes(x=Comp.1,y= Comp.2,colour = DVT,shape = DVT))+
  geom_point()+theme(legend.position = "right")+
  labs(x = "主成分得分1", y = "主成分得分2", title = "主成分降维散点图" )

svm_tune <- tune.svm(DVT~Comp.1+Comp.2, data = seed_score1,kernel ="radial",
                     gamma = seq(0.1,1,0.1),cost = seq(0.1,5,0.5))
## 使用ggplot2包热力图可视化参数搜索结果
plotdata1 <- svm_tune$performances
head(plotdata1)

library(RColorBrewer)
ggplot(plotdata1,aes(x = cost, y = gamma))+
  geom_tile(aes(fill = error))+
  scale_fill_gradientn(colours=brewer.pal(10,"OrRd"))+
  ggtitle("Performance of SVM")

svm_tune2 <- tune.svm(DVT~Comp.1+Comp.2, data = seed_score2,kernel ="radial",
                      gamma = seq(0.1,1,0.1),cost = seq(0.1,5,0.5))
## 使用ggplot2包热力图可视化参数搜索结果
plotdata2 <- svm_tune2$performances
head(plotdata2)

library(RColorBrewer)
ggplot(plotdata2,aes(x = cost, y = gamma))+
  geom_tile(aes(fill = error))+
  scale_fill_gradientn(colours=brewer.pal(10,"OrRd"))+
  ggtitle("Performance of SVM")


set.seed(1) # radial核SVM分类器
seedsvm1 <- svm(DVT~Comp.1+Comp.2, data = seed_score1,gamma=0.6, C=1)

win.graph(width=4.875, height=2.5,pointsize=8)
plot(seedsvm1, data = seed_score1,xpd=TRUE)
plot(seedsvm1, data = seed_score1, Comp.1~Comp.2)

plot(cmdscale(dist(x)),
     col=c("red","blue")[as.integer(y)],
     pch=c("o","+")[1:100%in%seedsvm1$index+1], mai=c(0.2,0.2,0.2,0.2))
legend(1,1,c("Normal","CIN"), col=c("red","blue"),lty=1,cex=0.5)


set.seed(1) # radial核SVM分类器
seedsvm2 <- svm(DVT~Comp.1+Comp.2, data = seed_score2,gamma=0.6, C=2)
plot(seedsvm2, data = seed_score2)

# Khorana 不联合D二聚体

fold_test <- dataset[folds[[num6]],]   #取folds[[i]]作为测试集  
roc12 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_test$K_Score))
plot(roc12, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_12 <- ifelse(fold_test$K_Score >= 3,2,1)
table(pred_12, fold_test$DVT)
(48+8)/72   ###  0.778  ###

# 在验证集评价模型

# LDA 不联合D二聚体

fold_predict25 <- predict(fold_pre7, newdata=validation, type="response")
fold_predict25 <- as.data.frame(fold_predict25)$posterior.有深静脉血栓

roc25 <- roc(as.numeric(validation$DVT),fold_predict25)
plot(roc25, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_25 <- ifelse(fold_predict25 >= 0.241,2,1)
table(pred_25, validation$DVT)
(168+49)/310   ###  0.700  ###

# LR 不联合D二聚体

fold_predict26 <- predict(fold_pre8,type='response', newdata=validation)  

roc26 <- roc(as.numeric(validation$DVT),fold_predict26)
plot(roc26, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_26 <- ifelse(fold_predict26 >= 0.257,2,1)
table(pred_26, validation$DVT)
(172+48)/310   ###  0.710  ###

# CART 不联合D二聚体

fold_predict27 <- predict(fold_pre9,type='class', newdata=validation)  

roc27 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict27))
plot(roc27, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict27, validation$DVT)
(211+23)/310   ###  0.755  ###

# RF 不联合D二聚体

fold_predict28 <- predict(fold_pre10,type='response', newdata=validation)  

roc28 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict28))
plot(roc28, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict28, validation$DVT)
(220+26)/310   ###  0.794  ###

# SVM 不联合D二聚体

fold_predict29 <- predict(fold_pre11,type='response', newdata=validation)  

roc29 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict29))
plot(roc29, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict29, validation$DVT)
(219+18)/310   ###  0.765  ###

# Khorana 不联合D二聚体

roc30 <- roc(as.numeric(validation$DVT),as.numeric(validation$K_Score))
plot(roc30, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_30 <- ifelse(validation$K_Score >= 3,2,1)
table(pred_30, validation$DVT)
(184+27)/310   ###  0.681 ###

############   绘制校准曲线   ############

LDA <- fold_predict25
LR <- fold_predict26
CART <- as.numeric(fold_predict27)
RF <- as.numeric(fold_predict28)
SVM <-as.numeric(fold_predict29)
Khorana <- validation$K_Score

library(rms)
trellis.par.set(caretTheme())
cal_obj <- calibration(validation$DVT ~ LDA+LR+CART+RF+SVM+Khorana,
                       data = validation,
                       cuts = 13)
plot(cal_obj, type = "l", auto.key = list(columns = 3,
                                          lines = TRUE,
                                          points = FALSE))

library("plotROC")
ggplot(cal_obj)+geom_line()+geom_abline(intercept=100,slope=-1)+
  labs(title = "Fig.2. Calibration Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))


# 6.2 联合D二聚体算法测评

# DVT~Age+ADL+log2LOS+log2CCI+Chemotherapy+Port_cath+HSA+Antibiotic+NSAID+Opium+Bed+Plaster+VTEHistory+log2WBC+log2Hb+log2DDimer+Immunotherapy+log2PLT+TumorStage+Drinking+EPO+TargetedTherapy

# 十折交叉验证的方法

#加载包
library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

# 数据集分十折
set.seed(7)
folds <- createFolds(y=dataset[,59],k=10)

# 给定参数

num7=0; num8=0; num9=0;
num10=0; num11=0; num12=0
auc_value.1 <-as.numeric()
auc_value.2 <-as.numeric()
auc_value.3 <-as.numeric()
auc_value.4 <-as.numeric()
auc_value.5 <-as.numeric()
auc_value.5.new <-as.numeric()
auc_value.6 <-as.numeric()
auc_value7 <-as.numeric()
auc_value8 <-as.numeric()
auc_value9 <-as.numeric()
auc_value10 <-as.numeric()
auc_value11 <-as.numeric()
auc_value11.new <-as.numeric()
auc_value12 <-as.numeric()

# 训练集建模和测试集调优
# LDA 联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.1 <- lda(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                    data=fold_train)
  fold_predict.1 <- predict(fold_pre.1, newdata=fold_test, type="response") 
  fold_predict.1 <- as.data.frame(fold_predict.1)$posterior.有深静脉血栓
  auc_value.1 <- append(auc_value.1,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.1)))
  
}  

plot(fold_pre.1 , panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

# LR联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.2 <- glm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                    data=fold_train, family ="binomial")  
  
  fold_predict.2 <- predict(fold_pre.2,type='response', newdata=fold_test)  
  
  
  auc_value.2 <- append(auc_value.2,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.2)))
  
} 

# cart联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.3 <- rpart(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                      data=fold_train, control = rpart.control(cp="0.001"))  
  
  fold_predict.3 <- predict(fold_pre.3,type='class', newdata=fold_test)  
  
  auc_value.3 <- append(auc_value.3,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.3)))
  
} 

# RF 联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre.4 <- randomForest(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                             data=fold_train, mtry=11, importance=TRUE, ntree=1000)  
  
  fold_predict.4 <- predict(fold_pre.4,type='response', newdata=fold_test)  
  
  auc_value.4 <- append(auc_value.4,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.4)))
} 

# SVM联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre.5 <- ksvm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                     data=fold_train, sigma=0.025, C=10, prob.model = TRUE)  
  
  fold_predict.5 <- predict(fold_pre.5,type='response', newdata=fold_test)  
  
  auc_value.5 <- append(auc_value.5,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.5)))
}

# Khorana联合D二聚体
# 数据集分十折

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],] #取folds[[i]]作为测试集  
  
  fold_test$K_Score_DD <- as.numeric(fold_test$K_Score)+as.numeric(fold_test$D二聚体)
  
  auc_value.6 <- append(auc_value.6,auc(as.numeric(fold_test$DVT),as.numeric(fold_test$K_Score_DD)))
} 


# 确定最优模型是第几折
num.1<-which.max(auc_value.1)
num.2<-which.max(auc_value.2)
num.3<-which.max(auc_value.3)
num.4<-which.max(auc_value.4)
num.5<-which.max(auc_value.5)
num.5.new <-which.max(auc_value.5.new)
num.6 <-which.max(auc_value.6)
print(auc_value.1)
print(auc_value.2)
print(auc_value.3)
print(auc_value.4)
print(auc_value.5)
print(auc_value.5.new)
print(auc_value.6)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 联合D二聚体
fold_test <- dataset[folds[[num.1]],]   

fold_train <- dataset[-folds[[num.1]],]

fold_pre13 <- lda(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                  data=fold_train) 

fold_predict13 <- predict(fold_pre13, newdata=fold_test, type="response")
fold_predict13 <- as.data.frame(fold_predict13)$posterior.有深静脉血栓

roc13 <- roc(as.numeric(fold_test[,59]),fold_predict13)
plot(roc13, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_13 <- ifelse(fold_predict13 >= 0.238,2,1)
table(pred_13, fold_test$DVT)
(47+13)/73   ###  0.822  ###

plot(fold_pre13, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre13$scaling, file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 3/LDA回归系数有DD.csv")


# LR 联合D二聚体
fold_test <- dataset[folds[[num.2]],]   

fold_train <- dataset[-folds[[num.2]],]

fold_pre14 <- glm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                  data=fold_train, family ="binomial")  

fold_predict14 <- predict(fold_pre14,type='response', newdata=fold_test)  

roc14 <- roc(as.numeric(fold_test[,59]),fold_predict14)
plot(roc14, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_14 <- ifelse(fold_predict14 >= 0.263,2,1)
table(pred_14, fold_test$DVT)
(49+13)/73   ###  0.849  ###

brier <- mean((fold_predict14-(as.numeric(fold_train$DVT)-1))^2)
brier   # 0.1507073
############ 绘制交互式列线图   ############

library(regplot)
fold_train <- dataset[-folds[[num.2]],]
regplot(fold_pre14,observation=fold_train[2,],points=TRUE) 
regplot(fold_pre14,observation=fold_train[2,]) 

# CART 联合D二聚体
fold_test <- dataset[folds[[num.3]],]   

fold_train <- dataset[-folds[[num.3]],]

fold_pre15 <- rpart(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                    data=fold_train, control = rpart.control(cp="0.001"))  

fold_predict15 <- predict(fold_pre15,type='class', newdata=fold_test)  

roc15 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict15))
plot(roc15, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict15, fold_test$DVT)
(53+8)/72   ###  0.847  ###

# RF 联合D二聚体
fold_test <- dataset[folds[[num.4]],]   

fold_train <- dataset[-folds[[num.4]],]

fold_pre16 <- randomForest(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                           data=fold_train, mtry=11, importance=TRUE, ntree=1000)  

fold_predict16 <- predict(fold_pre16,type='response', newdata=fold_test)  

roc16 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict16))
plot(roc16, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict16, fold_test$DVT)
(52+9)/72   ###  0.847  ###

# RF 联合D二聚体
fold_test <- dataset_train[folds[[num.4]],]   

fold_train <- dataset_train[-folds[[num.4]],]

fold_pre16 <- randomForest(DVT~Age+LOS+CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+WBC+DDimer,
                           data=fold_train, mtry=11, importance=TRUE, ntree=1000)  

fold_predict16 <- predict(fold_pre16,type='response', newdata=fold_test)  


roc16 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict16))
plot(roc16, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")
# 0.683
table(fold_predict16, fold_test$DVT)
(52+7)/72   ###  0.819  ###

varImpPlot(fold_pre16)

# SVM 联合D二聚体

fold_test <- dataset[folds[[num.5]],]   

fold_train <- dataset[-folds[[num.5]],]

fold_pre17 <- ksvm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                   data=fold_train, sigma=0.025, C=10, prob.model = TRUE)  

fold_predict17 <- predict(fold_pre17,type='response', newdata=fold_test)  

roc17 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict17))
plot(roc17, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict17, fold_test$DVT)
(52+7)/73   ###  0.808  ###

# Khorana 联合D二聚体

fold_test <- dataset[folds[[num.6]],]   #取folds[[i]]作为测试集  
fold_test$K_Score_DD <- as.numeric(fold_test$K_Score)+as.numeric(fold_test$D二聚体)
roc18 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_test$K_Score_DD))
plot(roc18, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_18 <- ifelse(fold_test$K_Score_DD >= 4,2,1)
table(pred_18, fold_test$DVT)
(39+12)/72   ###  0.778  ###

# 在验证集评价模型

# LDA 联合D二聚体

fold_predict31 <- predict(fold_pre13, newdata=validation, type="response")
fold_predict31 <- as.data.frame(fold_predict31)$posterior.有深静脉血栓

roc31 <- roc(as.numeric(validation[,60]),fold_predict31)
plot(roc31, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_31 <- ifelse(fold_predict31 >= 0.227,2,1)
table(pred_31, validation$DVT)
(159+52)/310   ###  0.681  ###

# LR 联合D二聚体

fold_predict32 <- predict(fold_pre14,type='response', newdata=validation)  

roc32 <- roc(as.numeric(validation[,60]),fold_predict32)
plot(roc32, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_32 <- ifelse(fold_predict32 >= 0.238,2,1)
table(pred_32, validation$DVT)
(159+53)/310   ###  0.732  ###

# CART 联合D二聚体

fold_predict33 <- predict(fold_pre15,type='class', newdata=validation)  

roc33 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict33))
plot(roc33, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict33, validation$DVT)
(210+28)/310   ###  0.768  ###

# RF 联合D二聚体

fold_predict34 <- predict(fold_pre16,type='response', newdata=validation)  

roc34 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict34))
plot(roc34, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict34, validation$DVT)
(224+26)/310   ###  0.813  ###

# SVM 联合D二聚体
set.seed(7)
fold_predict35 <- predict(fold_pre17,type='response', newdata=validation)  

roc35 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict35))
plot(roc35, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict35, validation$DVT)
(222+29)/310   ###  0.810  ###

# Khorana 联合D二聚体
validation$K_Score_DD <- as.numeric(validation$K_Score)+as.numeric(validation$D二聚体)
roc36 <- roc(as.numeric(validation$DVT),as.numeric(validation$K_Score_DD))
plot(roc36, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_36 <- ifelse(validation$K_Score_DD >= 5,2,1)
table(pred_36, validation$DVT)
(198+25)/310   ###  0.719 ###


############   绘制校准曲线   ############

LDA <- fold_predict25
LR <- fold_predict26
CART <- as.numeric(fold_predict27)-1
RF <- as.numeric(fold_predict28)-1
SVM <-as.numeric(fold_predict29)-1
Khorana <- validation$K_Score

library(caret)
library(rms)
trellis.par.set(caretTheme())
cal_obj1 <- calibration(validation$DVT ~ LDA+LR+CART+RF+SVM+Khorana,
                        data = validation,
                        cuts = 10,class = "有深静脉血栓")
plot(cal_obj1, auto.key = list(columns = 3,
                               lines = TRUE,
                               points = FALSE))

library("plotROC")
a <- ggplot(cal_obj1)+geom_line()+geom_abline(intercept=100,slope=-1)+
  labs(title = "Fig.2. Calibration Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))

a <- ggplot(cal_obj1)+geom_line()+
  labs(title = "Calibration Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))

LDA_DD <- fold_predict31
LR_DD <- fold_predict32
CART_DD <- as.numeric(fold_predict33)-1
RF_DD <- as.numeric(fold_predict34)-1
SVM_DD <-as.numeric(fold_predict35)-1
Khorana_DD <- validation$K_Score_DD

library(rms)
trellis.par.set(caretTheme())
cal_obj2 <- calibration(validation$DVT ~ LDA_DD+LR_DD+CART_DD+RF_DD+SVM_DD+Khorana_DD,
                        data = validation,
                        cuts = 10,class = "有深静脉血栓")
plot(cal_obj2, auto.key = list(columns = 3,
                               lines = TRUE,
                               points = FALSE))
library("plotROC")
b <- ggplot(cal_obj2)+geom_line()+geom_abline(intercept=100,slope=-1)+
  labs(title = "Fig.3. Calibration Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))

b <- ggplot(cal_obj2)+geom_line()+
  labs(title = "Calibration Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))

############   绘制ROC曲线   ############

library("plotROC")
validation$LDA <- LDA
validation$LR <- LR
validation$CT <- CART
validation$RF <- RF
validation$SVM <- SVM
validation$Khorana <- Khorana

longtest1 <- melt_roc(validation, "DVT", c("LDA", "LR", "CT", "RF", "SVM", "Khorana"))

ggplot(longtest1, aes(d = D, m = M, color = name)) + geom_roc(n.cuts = 0) + style_roc()

c <- ggplot(longtest1, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) +
  labs(title = "Fig.4. ROC Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))+geom_abline()


library("plotROC")
validation$LDA_DD <- LDA_DD
validation$LR_DD <- LR_DD
validation$CT_DD <- CART_DD
validation$RF_DD <- RF_DD
validation$SVM_DD <- SVM_DD
validation$Khorana_DD <- Khorana_DD

longtest2 <- melt_roc(validation, "DVT", c("LDA_DD", "LR_DD", "CT_DD", "RF_DD", "SVM_DD", "Khorana_DD"))

ggplot(longtest2, aes(d = D, m = M, color = name)) + geom_roc(n.cuts = 0) + style_roc()

d <- ggplot(longtest2, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) +
  labs(title = "Fig.5. ROC Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5))+geom_abline()

############   绘制变量重要性图表   ############

library(randomForest)
Fig_3_Relative_importance <- fold_pre10
e <- varImpPlot(Fig_3_Relative_importance, main = "Fig.3. Relative importance according to none-D-Dimer RF")

library(randomForest)
Fig_4_Relative_importance <- fold_pre.4
f <- varImpPlot(Fig_4__Relative_importance, main = "Fig.4. Relative importance according to D-Dimer RF")

######  绘制组图   ######

library(ggpubr)
a <- ggplot(cal_obj1)+geom_line()+
  labs(title = "Calibration Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))
b <- ggplot(cal_obj2)+geom_line()+
  labs(title = "Calibration Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))
c <- ggplot(longtest1, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) +
  labs(title = "ROC Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))+geom_abline()
d <- ggplot(longtest2, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) +
  labs(title = "ROC Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))+geom_abline()

ggarrange(a,b,c,d, labels=c("A","B","C","D"), ncol = 2, nrow = 2)

e <- varImpPlot(Fig_6__Relative_importance)

ggarrange(ggarrange(c,d, ncol = 2, labels = c("A","B")), a, nrow=2, labels="C")

#########    二、随机森林变量重要性排序   ######### 

randomForest <- randomForest(DVT~Sex+Smoking+Drinking+Bed+Plaster+TumorStage+Chemotherapy+
                               TargetedTherapy+Surgery+Radiotherapy+Immunotherapy+CVC+
                               PICC+Port_cath+IPCP+Transfusion+EPO+sFe+HSA+MGF+Antibiotic+
                               Cortisol+NSAID+Opium+Lymphadenopathy+VaricoseVeins+VTEHistory+
                               ICUCCU+Age+ADL+K_Cancerlevel+K_BMI+log2CCI+log2LOS+log2WBC+log2PLT+
                               log2Hb, data=dataset, mtry=6, importance=TRUE, ntree=1000) 

varImpPlot(randomForest)

randomForest_DD <- randomForest(DVT~Sex+Smoking+Drinking+Bed+Plaster+TumorStage+Chemotherapy+
                                  TargetedTherapy+Surgery+Radiotherapy+Immunotherapy+CVC+
                                  PICC+Port_cath+IPCP+Transfusion+EPO+sFe+HSA+MGF+Antibiotic+
                                  Cortisol+NSAID+Opium+Lymphadenopathy+VaricoseVeins+VTEHistory+
                                  ICUCCU+Age+ADL+K_Cancerlevel+K_BMI+log2CCI+log2LOS+log2WBC+log2PLT+
                                  log2Hb+log2DDimer, data=dataset, mtry=6, importance=TRUE, ntree=1000) 

varImpPlot(randomForest_DD)

##########  2.1 确定变量池   ##########

# DVT~VTEHistory+Chemotherapy+Age+Bed+log2WBC+log2CCI+TumorStage+log2PLT+Transfusion+Surgery
# DVT~log2WBC+log2PLT+log2LOS+log2Hb+log2CCI+TumorStage+Age+K_Cancerlevel+VTEHistory+Chemotherapy

# DVT~log2DDimer+VTEHistory+Age+log2CCI+Chemotherapy+TumorStage+Bed+log2WBC+log2PLT+log2Hb
# DVT~log2DDimer+log2WBC+log2PLT+log2LOS+log2Hb+log2CCI+TumorStage+Age+K_Cancerlevel+VTEHistory

# DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT

# DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb


library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

##########  2.2 模型参数调优   ##########

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

## tune cart 无DD，cp=0.001,AUC=0.597；
set.seed(7)
cartgrid <- expand.grid(.cp=seq(0.001, 0.150, by=0.004))

cart_1 <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                data=dataset, method="rpart", metric="ROC", tuneGrid=cartgrid, trControl=fitControl)
print(cart_1)
plot(cart_1)

## tune cart 有DD，cp=0.005,AUC=0.690；
set.seed(7)
cartgrid <- expand.grid(.cp=seq(0.001, 0.150, by=0.004))
cart_2 <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                data=dataset, method="rpart", metric="ROC", tuneGrid=cartgrid, trControl=fitControl)
print(cart_2)
plot(cart_2)

## tune RF 无DD，mtry = 2,AUC=0.689；
library(import)

set.seed(7)
rfgrid <- expand.grid(.mtry=c(1, 2, 3, 4, 11))
RF_1 <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
              data=dataset, method="parRF", metric="ROC", tuneGrid=rfgrid, trControl=fitControl)
print(RF_1)
plot(RF_1)

## tune RF 有DD，mtry = 2,AUC=0.744；
set.seed(7)
rfgrid <- expand.grid(.mtry=c(1, 2, 3, 4, 11))
RF_2 <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
              data=dataset, method="parRF", metric="ROC", tuneGrid=rfgrid, trControl=fitControl)
print(RF_2)
plot(RF_2)

## tune svm 无DD，sigma = 0.025,C= 7, AUC=0.629；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_1 <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT, 
               data=dataset, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_1)
plot(svm_1)

## tune svm 有DD，sigma = 0.025,C= 2, AUC=0.729；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_2 <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
               data=dataset, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_2)
plot(svm_2)

# tune gbm 无DD，interaction.depth = 1
# n.trees = 50, shrinkage =0.1 , n.minobsinnode = 20, 
# AUC=0.661；

library(gbm)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
set.seed(7)
gbmFit <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT, data=dataset, 
                method = "gbm", 
                trControl = fitControl, 
                verbose = FALSE, 
                tuneGrid = gbmGrid,
                ## Specify which metric to optimize
                metric = "ROC")
print(gbmFit)
plot(gbmFit)

# tune gbm 有DD，interaction.depth = , 
# n.trees = , shrinkage = , n.minobsinnode = , 
# AUC=0.；
set.seed(7)
gbmFit_DD <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                   data=dataset, 
                   method = "gbm", 
                   trControl = fitControl, 
                   verbose = FALSE, 
                   tuneGrid = gbmGrid,
                   ## Specify which metric to optimize
                   metric = "ROC")
print(gbmFit_DD)
plot(gbmFit_DD)

### 取交集

set.seed(7)
knngrid <- expand.grid(.k=seq(1,20,by=1))
fit.knn <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                 data=dataset, method="knn", metric="ROC", tuneGrid=knngrid, trControl=fitControl)
print(fit.knn)
plot(fit.knn)

set.seed(7)
knngrid <- expand.grid(.k=seq(1,20,by=1))
fit.knn_DD <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                    data=dataset, method="knn", metric="ROC", tuneGrid=knngrid, trControl=fitControl)
print(fit.knn_DD)
plot(fit.knn_DD)


# 数据集分十折
set.seed(7)
folds <- createFolds(y=dataset[,59],k=10)

# 给定参数
max1=0; max2=0; max3=0;
max4=0; max5=0; max6=0;
num1=0; num2=0; num3=0;
num4=0; num5=0; num6=0
auc_value1 <-as.numeric()
auc_value2 <-as.numeric()
auc_value3 <-as.numeric()
auc_value4 <-as.numeric()
auc_value5 <-as.numeric()
auc_value5.new <-as.numeric()
auc_value6 <-as.numeric()

# 训练集建模和测试集调优
# LDA 不联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre1 <- lda(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                   data=fold_train)
  fold_predict1 <- predict(fold_pre1, newdata=fold_test, type="response") 
  fold_predict1 <- as.data.frame(fold_predict1)$posterior.有深静脉血栓
  auc_value1 <- append(auc_value1,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict1)))
  
}  

# LR不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre2 <- glm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                   data=fold_train, family ="binomial")  
  
  fold_predict2 <- predict(fold_pre2,type='response', newdata=fold_test)  
  
  
  auc_value2 <- append(auc_value2,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict2)))
  
} 


# cart不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre3 <- rpart(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                     data=fold_train, control = rpart.control(cp="0.001"))  
  
  fold_predict3 <- predict(fold_pre3,type='class', newdata=fold_test)  
  
  auc_value3 <- append(auc_value3,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict3)))
  
} 

# RF 不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre4 <- randomForest(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                            data=fold_train, mtry=7, importance=TRUE, ntree=1000)  
  
  fold_predict4 <- predict(fold_pre4,type='response', newdata=fold_test)  
  
  auc_value4 <- append(auc_value4,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict4)))
} 

# SVM 不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre5 <- ksvm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                    data=fold_train, sigma=0.025, C=7, prob.model = TRUE)  
  
  fold_predict5 <- predict(fold_pre5,type='response', newdata=fold_test)  
  
  auc_value5 <- append(auc_value5,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict5)))
} 


# 确定最优模型是第几折
num1<-which.max(auc_value1)
num2<-which.max(auc_value2)
num3<-which.max(auc_value3)
num4<-which.max(auc_value4)
num5<-which.max(auc_value5)
num5.new <-which.max(auc_value5.new)
num6 <-which.max(auc_value6)
print(auc_value1)
print(auc_value2)
print(auc_value3)
print(auc_value4)
print(auc_value5)
print(auc_value5.new)
print(auc_value6)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 不联合D二聚体

fold_test <- dataset[folds[[num1]],]   

fold_train <- dataset[-folds[[num1]],]

fold_pre7 <- lda(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                 data=fold_train) 

fold_predict7 <- predict(fold_pre7, newdata=fold_test, type="response")
fold_predict7 <- as.data.frame(fold_predict7)$posterior.有深静脉血栓

roc7 <- roc(as.numeric(fold_test$DVT),fold_predict7)
plot(roc7, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_7 <- ifelse(fold_predict7 >= 0.204,2,1)
table(pred_7, fold_test$DVT)
(40+12)/72   ###  0.722  ###

plot(fold_pre7, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre7$scaling ,file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 3/LDA回归系数无DD.csv")

# LR 不联合D二聚体
fold_test <- dataset[folds[[num2]],]   

fold_train <- dataset[-folds[[num2]],]

fold_pre8 <- glm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                 data=fold_train, family ="binomial")  

fold_predict8 <- predict(fold_pre8,type='response', newdata=fold_test)  

roc8 <- roc(as.numeric(fold_test$DVT),fold_predict8)
plot(roc8, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_7 <- ifelse(fold_predict7 >= 0.212,2,1)
table(pred_7, fold_test$DVT)
(41+11)/72   ###  0.722  ###

# CART 不联合D二聚体
fold_test <- dataset[folds[[num3]],]   

fold_train <- dataset[-folds[[num3]],]

fold_pre9 <- rpart(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                   data=fold_train, control = rpart.control(cp="0.001"))  

fold_predict9 <- predict(fold_pre9,type='class', newdata=fold_test)  

roc9 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict9))
plot(roc9, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict9, fold_test$DVT)
(46+8)/72   ###  0.750  ###

# RF 不联合D二聚体
fold_test <- dataset[folds[[num4]],]   

fold_train <- dataset[-folds[[num4]],]

fold_pre10 <- randomForest(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                           data=fold_train, mtry=7, importance=TRUE, ntree=1000)  

fold_predict10 <- predict(fold_pre10,type='response', newdata=fold_test)  

roc10 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict10))
plot(roc10, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict10, fold_test$DVT)
(54+3)/72   ###  0.792 ###

# SVM 不联合D二聚体

fold_test <- dataset[folds[[num5]],]   

fold_train <- dataset[-folds[[num5]],]

fold_pre11 <- ksvm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI+TumorStage+log2PLT,
                   data=fold_train, sigma=0.025, C=7, prob.model = TRUE)  

fold_predict11 <- predict(fold_pre11,type='response', newdata=fold_test)  

roc11 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict11))
plot(roc11, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict11, fold_test$DVT)
(53+5)/73   ###  0.795  ###

# 在验证集评价模型

# LDA 不联合D二聚体

fold_predict25 <- predict(fold_pre7, newdata=validation, type="response")
fold_predict25 <- as.data.frame(fold_predict25)$posterior.有深静脉血栓

roc25 <- roc(as.numeric(validation$DVT),fold_predict25)
plot(roc25, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_25 <- ifelse(fold_predict25 >= 0.164,2,1)
table(pred_25, validation$DVT)
(126+62)/310   ###  0.606  ###

# LR 不联合D二聚体

fold_predict26 <- predict(fold_pre8,type='response', newdata=validation)  

roc26 <- roc(as.numeric(validation$DVT),fold_predict26)
plot(roc26, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_26 <- ifelse(fold_predict26 >= 0.180,2,1)
table(pred_26, validation$DVT)
(126+61)/310   ###  0.603  ###

# CART 不联合D二聚体

fold_predict27 <- predict(fold_pre9,type='class', newdata=validation)  

roc27 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict27))
plot(roc27, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict27, validation$DVT)
(207+26)/310   ###  0.752  ###

# RF 不联合D二聚体

fold_predict28 <- predict(fold_pre10,type='response', newdata=validation)  

roc28 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict28))
plot(roc28, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict28, validation$DVT)
(223+17)/310   ###  0.774  ###

# SVM 不联合D二聚体

fold_predict29 <- predict(fold_pre11,type='response', newdata=validation)  

roc29 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict29))
plot(roc29, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict29, validation$DVT)
(215+16)/310   ###  0.745  ###


############   2.3 绘制校准曲线   ############

LDA <- fold_predict25
LR <- fold_predict26
CART <- as.numeric(fold_predict27)
RF <- as.numeric(fold_predict28)
SVM <-as.numeric(fold_predict29)
Khorana <- validation$K_Score

library(rms)
trellis.par.set(caretTheme())
cal_obj <- calibration(validation$DVT ~ LDA+LR+CART+RF+SVM+Khorana,
                       data = validation,
                       cuts = 13)
plot(cal_obj, type = "l", auto.key = list(columns = 3,
                                          lines = TRUE,
                                          points = FALSE))

# 6.2 联合D二聚体算法测评

# 十折交叉验证的方法

#加载包
library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

# 数据集分十折
set.seed(7)
folds <- createFolds(y=dataset[,59],k=10)

# 给定参数

num7=0; num8=0; num9=0;
num10=0; num11=0; num12=0
auc_value.1 <-as.numeric()
auc_value.2 <-as.numeric()
auc_value.3 <-as.numeric()
auc_value.4 <-as.numeric()
auc_value.5 <-as.numeric()
auc_value.5.new <-as.numeric()
auc_value.6 <-as.numeric()
auc_value7 <-as.numeric()
auc_value8 <-as.numeric()
auc_value9 <-as.numeric()
auc_value10 <-as.numeric()
auc_value11 <-as.numeric()
auc_value11.new <-as.numeric()
auc_value12 <-as.numeric()

# 训练集建模和测试集调优
# LDA 联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.1 <- lda(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                    data=fold_train)
  fold_predict.1 <- predict(fold_pre.1, newdata=fold_test, type="response") 
  fold_predict.1 <- as.data.frame(fold_predict.1)$posterior.有深静脉血栓
  auc_value.1 <- append(auc_value.1,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.1)))
  
}  

plot(fold_pre.1 , panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

# LR联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.2 <- glm(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                    data=fold_train, family ="binomial")  
  
  fold_predict.2 <- predict(fold_pre.2,type='response', newdata=fold_test)  
  
  
  auc_value.2 <- append(auc_value.2,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.2)))
  
} 

# cart联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.3 <- rpart(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                      data=fold_train, control = rpart.control(cp="0.005"))  
  
  fold_predict.3 <- predict(fold_pre.3,type='class', newdata=fold_test)  
  
  auc_value.3 <- append(auc_value.3,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.3)))
  
} 

# RF 联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre.4 <- randomForest(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                             data=fold_train, mtry=8, importance=TRUE, ntree=1000)  
  
  fold_predict.4 <- predict(fold_pre.4,type='response', newdata=fold_test)  
  
  auc_value.4 <- append(auc_value.4,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.4)))
} 

# SVM联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre.5 <- ksvm(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                     data=fold_train, sigma=0.025, C=2, prob.model = TRUE)  
  
  fold_predict.5 <- predict(fold_pre.5,type='response', newdata=fold_test)  
  
  auc_value.5 <- append(auc_value.5,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.5)))
}


# 确定最优模型是第几折
num.1<-which.max(auc_value.1)
num.2<-which.max(auc_value.2)
num.3<-which.max(auc_value.3)
num.4<-which.max(auc_value.4)
num.5<-which.max(auc_value.5)
num.5.new <-which.max(auc_value.5.new)
num.6 <-which.max(auc_value.6)
print(auc_value.1)
print(auc_value.2)
print(auc_value.3)
print(auc_value.4)
print(auc_value.5)
print(auc_value.5.new)
print(auc_value.6)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 联合D二聚体
fold_test <- dataset[folds[[num.1]],]   

fold_train <- dataset[-folds[[num.1]],]

fold_pre13 <- lda(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                  data=fold_train) 

fold_predict13 <- predict(fold_pre13, newdata=fold_test, type="response")
fold_predict13 <- as.data.frame(fold_predict13)$posterior.有深静脉血栓

roc13 <- roc(as.numeric(fold_test[,59]),fold_predict13)
plot(roc13, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_13 <- ifelse(fold_predict13 >= 0.166,2,1)
table(pred_13, fold_test$DVT)
(37+15)/73  ###  0.712  ###

plot(fold_pre13, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre13$scaling, file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 3/LDA回归系数有DD.csv")


# LR 联合D二聚体
fold_test <- dataset[folds[[num.2]],]   

fold_train <- dataset[-folds[[num.2]],]

fold_pre14 <- glm(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                  data=fold_train, family ="binomial")  

fold_predict14 <- predict(fold_pre14,type='response', newdata=fold_test)  

roc14 <- roc(as.numeric(fold_test[,59]),fold_predict14)
plot(roc14, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_14 <- ifelse(fold_predict14 >= 0.146,2,1)
table(pred_14, fold_test$DVT)
(31+16)/73   ###  0.644  ###

# CART 联合D二聚体
fold_test <- dataset[folds[[num.3]],]   

fold_train <- dataset[-folds[[num.3]],]

fold_pre15 <- rpart(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                    data=fold_train, control = rpart.control(cp="0.005"))  

fold_predict15 <- predict(fold_pre15,type='class', newdata=fold_test)  

roc15 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict15))
plot(roc15, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict15, fold_test$DVT)
(51+8)/73   ###  0.808  ###

# RF 联合D二聚体
fold_test <- dataset[folds[[num.4]],]   

fold_train <- dataset[-folds[[num.4]],]

fold_pre16 <- randomForest(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                           data=fold_train, mtry=8, importance=TRUE, ntree=1000)  

fold_predict16 <- predict(fold_pre16,type='response', newdata=fold_test)  

roc16 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict16))
plot(roc16, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict16, fold_test$DVT)
(54+8)/73   ###  0.847  ###

# SVM 联合D二聚体

fold_test <- dataset[folds[[num.5]],]   

fold_train <- dataset[-folds[[num.5]],]

fold_pre17 <- ksvm(DVT~log2DDimer+VTEHistory+Age+log2CCI+TumorStage+log2WBC+log2PLT+log2Hb,
                   data=fold_train, sigma=0.025, C=2, prob.model = TRUE)  

fold_predict17 <- predict(fold_pre17,type='response', newdata=fold_test)  

roc17 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict17))
plot(roc17, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict17, fold_test$DVT)
(55+5)/73   ###  0.822  ###


# 在验证集评价模型

# LDA 联合D二聚体

fold_predict31 <- predict(fold_pre13, newdata=validation, type="response")
fold_predict31 <- as.data.frame(fold_predict31)$posterior.有深静脉血栓

roc31 <- roc(as.numeric(validation[,60]),fold_predict31)
plot(roc31, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_31 <- ifelse(fold_predict31 >= 0.357,2,1)
table(pred_31, validation$DVT)
(213+38)/310   ###  0.681  ###

# LR 联合D二聚体

fold_predict32 <- predict(fold_pre14,type='response', newdata=validation)  

roc32 <- roc(as.numeric(validation[,60]),fold_predict32)
plot(roc32, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_32 <- ifelse(fold_predict32 >= 0.322,2,1)
table(pred_32, validation$DVT)
(196+41)/310   ###  0.765  ###

# CART 联合D二聚体

fold_predict33 <- predict(fold_pre15,type='class', newdata=validation)  

roc33 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict33))
plot(roc33, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict33, validation$DVT)
(209+30)/310   ###  0.771  ###

# RF 联合D二聚体

fold_predict34 <- predict(fold_pre16,type='response', newdata=validation)  

roc34 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict34))
plot(roc34, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict34, validation$DVT)
# RF 联合D二聚体

fold_predict34 <- predict(fold_pre16,type='response', newdata=validation)  

roc34 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict34))
plot(roc34, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict34, validation$DVT)
(227+29)/310    ###  0.826  ###

# SVM 联合D二聚体

fold_predict35 <- predict(fold_pre17,type='response', newdata=validation)  

roc35 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict35))
plot(roc35, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict35, validation$DVT)
(233+23)/310   ###  0.826  ###

#########   三、三种方法,取交集   ######### 

##########  3.1 确定变量池   ##########

# DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI

# DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC


library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

##########  3.2 模型参数调优   ##########

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

## tune cart 无DD，cp=0.001,AUC=0.608；
set.seed(7)
cartgrid <- expand.grid(.cp=seq(0.001, 0.150, by=0.004))

cart_1 <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                data=dataset, method="rpart", metric="ROC", tuneGrid=cartgrid, trControl=fitControl)
print(cart_1)
plot(cart_1)

## tune cart 有DD，cp=0.005,AUC=0.705；
set.seed(7)
cartgrid <- expand.grid(.cp=seq(0.001, 0.150, by=0.004))
cart_2 <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                data=dataset, method="rpart", metric="ROC", tuneGrid=cartgrid, trControl=fitControl)
print(cart_2)
plot(cart_2)

## tune RF 无DD，mtry = 1,AUC=0.669；
library(import)

set.seed(7)
rfgrid <- expand.grid(.mtry=c(1, 2, 3, 4, 11))
RF_1 <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
              data=dataset, method="parRF", metric="ROC", tuneGrid=rfgrid, trControl=fitControl)
print(RF_1)
plot(RF_1)

## tune RF 有DD，mtry = 1,AUC=0.741；
set.seed(7)
rfgrid <- expand.grid(.mtry=c(1, 2, 3, 4, 11))
RF_2 <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
              data=dataset, method="parRF", metric="ROC", tuneGrid=rfgrid, trControl=fitControl)
print(RF_2)
plot(RF_2)

## tune svm 无DD，sigma = 0.025,C= 2, AUC=0.650；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_1 <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI, 
               data=dataset, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_1)
plot(svm_1)

## tune svm 有DD，sigma = 0.025,C= 1, AUC=0.735；
set.seed(7)
svmgrid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
svm_2 <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
               data=dataset, method="svmRadial", metric="ROC", tuneGrid=svmgrid, trControl=fitControl)
print(svm_2)
plot(svm_2)

# tune gbm 无DD，interaction.depth = 1
# n.trees = 50, shrinkage =0.1 , n.minobsinnode = 20, 
# AUC=0.661；

library(gbm)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
set.seed(7)
gbmFit <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI, data=dataset, 
                method = "gbm", 
                trControl = fitControl, 
                verbose = FALSE, 
                tuneGrid = gbmGrid,
                ## Specify which metric to optimize
                metric = "ROC")
print(gbmFit)
plot(gbmFit)

# tune gbm 有DD，interaction.depth = 1, 
# n.trees = 100, shrinkage = 0.1, n.minobsinnode = 20, 
# AUC=0.729；
set.seed(7)
gbmFit_DD <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                   data=dataset, 
                   method = "gbm", 
                   trControl = fitControl, 
                   verbose = FALSE, 
                   tuneGrid = gbmGrid,
                   ## Specify which metric to optimize
                   metric = "ROC")
print(gbmFit_DD)
plot(gbmFit_DD)

### 取交集

set.seed(7)
knngrid <- expand.grid(.k=seq(1,20,by=1))
fit.knn <- train(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                 data=dataset, method="knn", metric="ROC", tuneGrid=knngrid, trControl=fitControl)
print(fit.knn)
plot(fit.knn)

set.seed(7)
knngrid <- expand.grid(.k=seq(1,20,by=1))
fit.knn_DD <- train(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                    data=dataset, method="knn", metric="ROC", tuneGrid=knngrid, trControl=fitControl)
print(fit.knn_DD)
plot(fit.knn_DD)


# 数据集分十折
set.seed(7)
folds <- createFolds(y=dataset[,59],k=10)

# 给定参数
max1=0; max2=0; max3=0;
max4=0; max5=0; max6=0;
num1=0; num2=0; num3=0;
num4=0; num5=0; num6=0
auc_value1 <-as.numeric()
auc_value2 <-as.numeric()
auc_value3 <-as.numeric()
auc_value4 <-as.numeric()
auc_value5 <-as.numeric()
auc_value5.new <-as.numeric()
auc_value6 <-as.numeric()

# 训练集建模和测试集调优
# LDA 不联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre1 <- lda(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                   data=fold_train)
  fold_predict1 <- predict(fold_pre1, newdata=fold_test, type="response") 
  fold_predict1 <- as.data.frame(fold_predict1)$posterior.有深静脉血栓
  auc_value1 <- append(auc_value1,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict1)))
  
}  

# LR不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre2 <- glm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                   data=fold_train, family ="binomial")  
  
  fold_predict2 <- predict(fold_pre2,type='response', newdata=fold_test)  
  
  
  auc_value2 <- append(auc_value2,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict2)))
  
} 


# cart不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre3 <- rpart(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                     data=fold_train, control = rpart.control(cp="0.001"))  
  
  fold_predict3 <- predict(fold_pre3,type='class', newdata=fold_test)  
  
  auc_value3 <- append(auc_value3,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict3)))
  
} 

# RF 不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre4 <- randomForest(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                            data=fold_train, mtry=5, importance=TRUE, ntree=1000)  
  
  fold_predict4 <- predict(fold_pre4,type='response', newdata=fold_test)  
  
  auc_value4 <- append(auc_value4,auc(as.numeric(fold_test$DVT),as.numeric(fold_predict4)))
} 

# SVM 不联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre5 <- ksvm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                    data=fold_train, sigma=0.025, C=5, prob.model = TRUE)  
  
  fold_predict5 <- predict(fold_pre5,type='response', newdata=fold_test)  
  
  auc_value5 <- append(auc_value5,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict5)))
} 


# 确定最优模型是第几折
num1<-which.max(auc_value1)
num2<-which.max(auc_value2)
num3<-which.max(auc_value3)
num4<-which.max(auc_value4)
num5<-which.max(auc_value5)
num5.new <-which.max(auc_value5.new)
num6 <-which.max(auc_value6)
print(auc_value1)
print(auc_value2)
print(auc_value3)
print(auc_value4)
print(auc_value5)
print(auc_value5.new)
print(auc_value6)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 不联合D二聚体

fold_test <- dataset[folds[[num1]],]   

fold_train <- dataset[-folds[[num1]],]

fold_pre7 <- lda(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                 data=fold_train) 

fold_predict7 <- predict(fold_pre7, newdata=fold_test, type="response")
fold_predict7 <- as.data.frame(fold_predict7)$posterior.有深静脉血栓

roc7 <- roc(as.numeric(fold_test$DVT),fold_predict7)
plot(roc7, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_7 <- ifelse(fold_predict7 >= 0.148,2,1)
table(pred_7, fold_test$DVT)
(33+13)/72   ###  0.639  ###

plot(fold_pre7, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre7$scaling ,file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 3/LDA回归系数无DD.csv")

# LR 不联合D二聚体
fold_test <- dataset[folds[[num2]],]   

fold_train <- dataset[-folds[[num2]],]

fold_pre8 <- glm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                 data=fold_train, family ="binomial")  

fold_predict8 <- predict(fold_pre8,type='response', newdata=fold_test)  

roc8 <- roc(as.numeric(fold_test$DVT),fold_predict8)
plot(roc8, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_7 <- ifelse(fold_predict7 >= 0.157,2,1)
table(pred_7, fold_test$DVT)
(34+12)/72   ###  0.639  ###

# CART 不联合D二聚体
fold_test <- dataset[folds[[num3]],]   

fold_train <- dataset[-folds[[num3]],]

fold_pre9 <- rpart(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                   data=fold_train, control = rpart.control(cp="0.001"))  

fold_predict9 <- predict(fold_pre9,type='class', newdata=fold_test)  

roc9 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict9))
plot(roc9, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict9, fold_test$DVT)
(47+8)/72   ###  0.764  ###

# RF 不联合D二聚体
fold_test <- dataset[folds[[num4]],]   

fold_train <- dataset[-folds[[num4]],]

fold_pre10 <- randomForest(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                           data=fold_train, mtry=5, importance=TRUE, ntree=1000)  

fold_predict10 <- predict(fold_pre10,type='response', newdata=fold_test)  

roc10 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict10))
plot(roc10, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict10, fold_test$DVT)
(49+6)/73   ###  0.753 ###

# SVM 不联合D二聚体

fold_test <- dataset[folds[[num5]],]   

fold_train <- dataset[-folds[[num5]],]

fold_pre11 <- ksvm(DVT~VTEHistory+Chemotherapy+Age+log2WBC+log2CCI,
                   data=fold_train, sigma=0.025, C=5, prob.model = TRUE)  

fold_predict11 <- predict(fold_pre11,type='response', newdata=fold_test)  

roc11 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict11))
plot(roc11, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict11, fold_test$DVT)
(55+2)/73   ###  0.781  ###

# 在验证集评价模型

# LDA 不联合D二聚体

fold_predict25 <- predict(fold_pre7, newdata=validation, type="response")
fold_predict25 <- as.data.frame(fold_predict25)$posterior.有深静脉血栓

roc25 <- roc(as.numeric(validation$DVT),fold_predict25)
plot(roc25, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_25 <- ifelse(fold_predict25 >= 0.186,2,1)
table(pred_25, validation$DVT)
(146+59)/310   ###  0.661  ###

# LR 不联合D二聚体

fold_predict26 <- predict(fold_pre8,type='response', newdata=validation)  

roc26 <- roc(as.numeric(validation$DVT),fold_predict26)
plot(roc26, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_26 <- ifelse(fold_predict26 >= 0.199,2,1)
table(pred_26, validation$DVT)
(147+59)/310   ###  0.603  ###

# CART 不联合D二聚体

fold_predict27 <- predict(fold_pre9,type='class', newdata=validation)  

roc27 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict27))
plot(roc27, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict27, validation$DVT)
(210+22)/310   ###  0.748  ###

# RF 不联合D二聚体

fold_predict28 <- predict(fold_pre10,type='response', newdata=validation)  

roc28 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict28))
plot(roc28, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict28, validation$DVT)
(205+20)/310   ###  0.726  ###

# SVM 不联合D二聚体

fold_predict29 <- predict(fold_pre11,type='response', newdata=validation)  

roc29 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict29))
plot(roc29, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict29, validation$DVT)
# SVM 不联合D二聚体

fold_predict29 <- predict(fold_pre11,type='response', newdata=validation)  

roc29 <- roc(as.numeric(validation$DVT),as.numeric(fold_predict29))
plot(roc29, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict29, validation$DVT)  
(233+9)/310  ###  0.781  ###


############   3.3 绘制校准曲线   ############

LDA <- fold_predict25
LR <- fold_predict26
CART <- as.numeric(fold_predict27)
RF <- as.numeric(fold_predict28)
SVM <-as.numeric(fold_predict29)
Khorana <- validation$K_Score

library(rms)
trellis.par.set(caretTheme())
cal_obj <- calibration(validation$DVT ~ LDA+LR+CART+RF+SVM+Khorana,
                       data = validation,
                       cuts = 13)
plot(cal_obj, type = "l", auto.key = list(columns = 3,
                                          lines = TRUE,
                                          points = FALSE))

# 6.2 联合D二聚体算法测评

# 十折交叉验证的方法

#加载包
library(MASS) # LDA
library(rms) # LR
library(rpart) # CART
library(kernlab) # SVM
library(randomForest) # RF
library(pROC) # ROC
library(caret)

# 数据集分十折
set.seed(7)
folds <- createFolds(y=dataset[,59],k=10)

# 给定参数

num7=0; num8=0; num9=0;
num10=0; num11=0; num12=0
auc_value.1 <-as.numeric()
auc_value.2 <-as.numeric()
auc_value.3 <-as.numeric()
auc_value.4 <-as.numeric()
auc_value.5 <-as.numeric()
auc_value.5.new <-as.numeric()
auc_value.6 <-as.numeric()
auc_value7 <-as.numeric()
auc_value8 <-as.numeric()
auc_value9 <-as.numeric()
auc_value10 <-as.numeric()
auc_value11 <-as.numeric()
auc_value11.new <-as.numeric()
auc_value12 <-as.numeric()

# 训练集建模和测试集调优
# LDA 联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.1 <- lda(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                    data=fold_train)
  fold_predict.1 <- predict(fold_pre.1, newdata=fold_test, type="response") 
  fold_predict.1 <- as.data.frame(fold_predict.1)$posterior.有深静脉血栓
  auc_value.1 <- append(auc_value.1,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.1)))
  
}  

plot(fold_pre.1 , panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

# LR联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.2 <- glm(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                    data=fold_train, family ="binomial")  
  
  fold_predict.2 <- predict(fold_pre.2,type='response', newdata=fold_test)  
  
  
  auc_value.2 <- append(auc_value.2,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.2)))
  
} 

# cart联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  fold_pre.3 <- rpart(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                      data=fold_train, control = rpart.control(cp="0.005"))  
  
  fold_predict.3 <- predict(fold_pre.3,type='class', newdata=fold_test)  
  
  auc_value.3 <- append(auc_value.3,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.3)))
  
} 

# RF 联合D二聚体
for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre.4 <- randomForest(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                             data=fold_train, mtry=5, importance=TRUE, ntree=1000)  
  
  fold_predict.4 <- predict(fold_pre.4,type='response', newdata=fold_test)  
  
  auc_value.4 <- append(auc_value.4,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.4)))
} 

# SVM联合D二聚体

for(i in 1:10){  
  
  fold_test <- dataset[folds[[i]],]   #取folds[[i]]作为测试集  
  
  fold_train <- dataset[-folds[[i]],]   # 剩下的数据作为训练集    
  
  
  fold_pre.5 <- ksvm(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                     data=fold_train, sigma=0.025, C=5, prob.model = TRUE)  
  
  fold_predict.5 <- predict(fold_pre.5,type='response', newdata=fold_test)  
  
  auc_value.5 <- append(auc_value.5,auc(as.numeric(fold_test[,59]),as.numeric(fold_predict.5)))
}


# 确定最优模型是第几折
num.1<-which.max(auc_value.1)
num.2<-which.max(auc_value.2)
num.3<-which.max(auc_value.3)
num.4<-which.max(auc_value.4)
num.5<-which.max(auc_value.5)
num.5.new <-which.max(auc_value.5.new)
num.6 <-which.max(auc_value.6)
print(auc_value.1)
print(auc_value.2)
print(auc_value.3)
print(auc_value.4)
print(auc_value.5)
print(auc_value.5.new)
print(auc_value.6)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 联合D二聚体
fold_test <- dataset[folds[[num.1]],]   

fold_train <- dataset[-folds[[num.1]],]

fold_pre13 <- lda(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                  data=fold_train) 

fold_predict13 <- predict(fold_pre13, newdata=fold_test, type="response")
fold_predict13 <- as.data.frame(fold_predict13)$posterior.有深静脉血栓

roc13 <- roc(as.numeric(fold_test[,59]),fold_predict13)
plot(roc13, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_13 <- ifelse(fold_predict13 >= 0.160,2,1)
table(pred_13, fold_test$DVT)
(38+14)/73  ###  0.712  ###

plot(fold_pre13, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre13$scaling, file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 3/LDA回归系数有DD.csv")


# LR 联合D二聚体
fold_test <- dataset[folds[[num.2]],]   

fold_train <- dataset[-folds[[num.2]],]

fold_pre14 <- glm(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                  data=fold_train, family ="binomial")  

fold_predict14 <- predict(fold_pre14,type='response', newdata=fold_test)  

roc14 <- roc(as.numeric(fold_test[,59]),fold_predict14)
plot(roc14, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_14 <- ifelse(fold_predict14 >= 0.191,2,1)
table(pred_14, fold_test$DVT)
(41+13)/73   ###  0.740  ###

# CART 联合D二聚体
fold_test <- dataset[folds[[num.3]],]   

fold_train <- dataset[-folds[[num.3]],]

fold_pre15 <- rpart(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                    data=fold_train, control = rpart.control(cp="0.005"))  

fold_predict15 <- predict(fold_pre15,type='class', newdata=fold_test)  

roc15 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict15))
plot(roc15, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict15, fold_test$DVT)
(51+8)/72   ###  0.808  ###

# RF 联合D二聚体
fold_test <- dataset[folds[[num.4]],]   

fold_train <- dataset[-folds[[num.4]],]

fold_pre16 <- randomForest(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                           data=fold_train, mtry=5, importance=TRUE, ntree=1000)  

fold_predict16 <- predict(fold_pre16,type='response', newdata=fold_test)  

roc16 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict16))
plot(roc16, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict16, fold_test$DVT)
(49+8)/73   ###  0.781  ###

# SVM 联合D二聚体

fold_test <- dataset[folds[[num.5]],]   

fold_train <- dataset[-folds[[num.5]],]

fold_pre17 <- ksvm(DVT~log2DDimer+VTEHistory+Age+log2CCI++log2WBC,
                   data=fold_train, sigma=0.025, C=2, prob.model = TRUE)  

fold_predict17 <- predict(fold_pre17,type='response', newdata=fold_test)  

roc17 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict17))
plot(roc17, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict17, fold_test$DVT)
(55+7)/73   ###  0.849  ###


# 在验证集评价模型

# LDA 联合D二聚体

fold_predict31 <- predict(fold_pre13, newdata=validation, type="response")
fold_predict31 <- as.data.frame(fold_predict31)$posterior.有深静脉血栓

roc31 <- roc(as.numeric(validation[,60]),fold_predict31)
plot(roc31, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_31 <- ifelse(fold_predict31 >= 0.367,2,1)
table(pred_31, validation$DVT)
(213+37)/310   ###  0.806  ###

# LR 联合D二聚体

fold_predict32 <- predict(fold_pre14,type='response', newdata=validation)  

roc32 <- roc(as.numeric(validation[,60]),fold_predict32)
plot(roc32, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_32 <- ifelse(fold_predict32 >= 0.366,2,1)
table(pred_32, validation$DVT)
(211+37)/310   ###  0.800  ###

# CART 联合D二聚体

fold_predict33 <- predict(fold_pre15,type='class', newdata=validation)  

roc33 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict33))
plot(roc33, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict33, validation$DVT)
(211+34)/310   ###  0.790 ###

# RF 联合D二聚体

fold_predict34 <- predict(fold_pre16,type='response', newdata=validation)  

roc34 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict34))
plot(roc34, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict34, validation$DVT)
(221+35)/310    ###  0.826  ###

# SVM 联合D二聚体

fold_predict35 <- predict(fold_pre17,type='response', newdata=validation)  

roc35 <- roc(as.numeric(validation[,60]),as.numeric(fold_predict35))
plot(roc35, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict35, validation$DVT)
(229+27)/310   ###  0.826  ###


#########  绘制校准曲线的尝试    ######
library(caret)
library(rms)

set.seed(7)
ctrl <- trainControl(method = "none", classProbs = TRUE, # 输出预测概率 #
                     summaryFunction = twoClassSummary)

### 不联合D二聚体建立不同算法预测模型
#LDA模型
set.seed(7)
lda_lift <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data = dataset[-folds[[num1]],],
                  method = "lda", metric = "ROC",
                  trControl = ctrl)
#LR模型 
set.seed(7)
lr_lift <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data = dataset[-folds[[num2]],],
                 method = "glm", metric = "ROC",
                 trControl = ctrl)
#CART模型 
library(rpart)
set.seed(7)
cart_lift <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data = dataset[-folds[[num3]],],
                   method = "rpart2", metric = "ROC",
                   trControl = ctrl)
#RF模型 
library(randomForest)
set.seed(7)
rf_lift <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data = dataset[-folds[[num4]],],
                 method = "parRF", metric = "ROC",
                 trControl = ctrl)
#SVM模型 
library(kernlab)
set.seed(7)
svm_lift <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC, data = dataset[-folds[[num5]],],
                  method = "svmRadial", metric = "ROC",
                  trControl = ctrl)
#Khorana模型
set.seed(7)
khorana_lift <- train(DVT~K_Cancerlevel+K_WBC+K_PLT+K_HbEPO+K_BMI, data = dataset[-folds[[num6]],],
                      method = "glm", metric = "ROC",
                      trControl = ctrl)


library("foreign")    
validation.new <- read.spss("validation.sav")  
validation.new <- as.data.frame(validation.new)

## 提取阳性结局的预测概率
lift_results <- data.frame(DVT = validation$DVT)
lift_results$LDA <- predict(lda_lift, validation, type = "prob")[,"有深静脉血栓"]
lift_results$LR <- predict(lr_lift, validation, type = "prob")[,"有深静脉血栓"]
lift_results$CT <- predict(cart_lift, validation, type = "prob")[,"有深静脉血栓"]
lift_results$RF <- predict(rf_lift, validation, type = "prob")[,"有深静脉血栓"]
lift_results$SVM <- predict(svm_lift, validation, type = "prob")[,"有深静脉血栓"]
lift_results$Khorana <- predict(khorana_lift, validation, type = "prob")[,"有深静脉血栓"]

###  绘制校准曲线
library(rms)
trellis.par.set(caretTheme())
cal_new1 <- calibration(DVT ~ LDA+LR+CT+RF+SVM,
                        data = lift_results,
                        cuts = 5, class = "1")
plot(cal_new1, type = "l", auto.key = list(columns = 3,
                                           lines = TRUE,
                                           points = FALSE))

### 联合D二聚体

set.seed(7)
lda_lift_DD <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer, data = dataset[-folds[[num.1]],],
                     method = "lda", metric = "ROC",
                     trControl = ctrl)
set.seed(7)
lr_lift_DD <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer, data = dataset[-folds[[num.2]],],
                    method = "glm", metric = "ROC",
                    trControl = ctrl)

library(rpart)
set.seed(7)
cart_lift_DD <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer, data = dataset[-folds[[num.3]],],
                      method = "rpart2", metric = "ROC",
                      trControl = ctrl)
library(randomForest)
set.seed(7)
rf_lift_DD <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer, data = dataset[-folds[[num.4]],],
                    method = "parRF", metric = "ROC",
                    trControl = ctrl)
library(kernlab)
set.seed(7)
svm_lift_DD <- train(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer, data = dataset[-folds[[num5]],],
                     method = "svmRadial", metric = "ROC",
                     trControl = ctrl)
set.seed(7)
khorana_lift_DD <- train(DVT~K_Cancerlevel+K_WBC+K_PLT+K_HbEPO+K_BMI+D二聚体, data = dataset[-folds[[num.6]],],
                         method = "glm", metric = "ROC",
                         trControl = ctrl)


lift_results <- data.frame(DVT = validation$DVT)
lift_results$LDA_DD <- predict(lda_lift_DD, validation, type = "prob")[,"有深静脉血栓"]
lift_results$LR_DD <- predict(lr_lift_DD, validation, type = "prob")[,"有深静脉血栓"]
lift_results$CT_DD <- predict(cart_lift_DD, validation, type = "prob")[,"有深静脉血栓"]
lift_results$RF_DD <- predict(rf_lift_DD, validation, type = "prob")[,"有深静脉血栓"]
lift_results$SVM_DD <- predict(svm_lift_DD, validation, type = "prob")[,"有深静脉血栓"]
lift_results$Khorana_DD <- predict(khorana_lift_DD, validation, type = "prob")[,"有深静脉血栓"]

library(rms)
trellis.par.set(caretTheme())
cal_new2 <- calibration(DVT ~ LDA_DD+LR_DD+CT_DD+RF_DD+SVM_DD,
                        data = lift_results,
                        cuts = 5, class = "1")
plot(cal_new2, type = "l", auto.key = list(columns = 3,
                                           lines = TRUE,
                                           points = FALSE))


library(ggpubr)
library(plotROC)
m <- ggplot(cal_new1)+geom_line()+
  labs(title = "Calibration Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))
n <- ggplot(cal_new2)+geom_line()+
  labs(title = "Calibration Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))

ggarrange(m,n, labels=c("A","B"), ncol = 2, nrow = 1)

library(plotROC)
c <- ggplot(longtest1, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) +
  labs(title = "ROC Curves without D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))+geom_abline()
d <- ggplot(longtest2, aes(d = D, m = M, color = name)) + 
  geom_roc(n.cuts = 0) +
  labs(title = "ROC Curves with D-Dimer in the Validation set")+theme(plot.title = element_text(hjust = 0.5, size = 8))+geom_abline()

ggarrange(m,n,c,d, labels=c("A","B","C","D"), ncol = 2, nrow = 2)

########  绘制模型结果图，基于开发集

library("foreign")    
dataset2 <- read.spss("dataset1.sav")  
dataset2 <- as.data.frame(dataset2)


# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 不联合D二聚体
library(MASS)

fold_test <- dataset2[folds[[num1]],]   

fold_train <- dataset2[-folds[[num1]],]

fold_pre7 <- lda(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                 data=fold_train) 

fold_predict7 <- predict(fold_pre7, newdata=fold_test, type="response")
fold_predict7 <- as.data.frame(fold_predict7)$posterior.yes

roc7 <- roc(as.numeric(fold_test$DVT),fold_predict7)
plot(roc7, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_7 <- ifelse(fold_predict7 >= 0.291,2,1)
table(pred_7, fold_test$DVT)
(51+9)/72   ###  0.833  ###

plot(fold_pre7, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre7$scaling ,file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 终 英文变量和赋值 绘制结果图/LDA回归系数无DD.csv")

# LR 不联合D二聚体
fold_test <- dataset2[folds[[num2]],]   

fold_train <- dataset2[-folds[[num2]],]

fold_pre8 <- glm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                 data=fold_train, family ="binomial")  

fold_predict8 <- predict(fold_pre8,type='response', newdata=fold_test)  

roc8 <- roc(as.numeric(fold_test$DVT),fold_predict8)
plot(roc8, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_7 <- ifelse(fold_predict7 >= 0.288,2,1)
table(pred_7, fold_test$DVT)
(50+10)/72   ###  0.833  ###

library(rms)

attach(fold_train)
dd<-datadist(fold_train)
options(datadist='dd')

nomofit <- lrm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC
               , data=fold_train, x=T, y=T)

nomogram <- nomogram(nomofit, fun=plogis,fun.at=c(.001, .01, .05, seq(.1,.9, by=.1), .95, .99, .999),lp=F, funlabel="DVT pro")
plot(nomogram)

# CART 不联合D二聚体
fold_test <- dataset2[folds[[num3]],]   

fold_train <- dataset2[-folds[[num3]],]

fold_pre9 <- rpart(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                   data=fold_train, control = rpart.control(cp="0.001"))  

fold_predict9 <- predict(fold_pre9,type='class', newdata=fold_test)  

roc9 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict9))
plot(roc9, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict9, fold_test$DVT)
(48+6)/72   ###  0.750  ###

plot(fold_pre9, margin=0.05)
text(fold_pre9,  cex=0.8, pretty = 1)

# RF 不联合D二聚体

fold_test <- dataset2[folds[[num4]],]   

fold_train <- dataset2[-folds[[num4]],]

fold_pre10 <- randomForest(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+VTEHistory+log2WBC,
                           data=fold_train, mtry=9, importance=TRUE, ntree=1000)  

fold_predict10 <- predict(fold_pre10,type='response', newdata=fold_test)  

roc10 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict10))
plot(roc10, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict10, fold_test$DVT)
(51+5)/72   ###  0.778  ###

none_D_Dimer_RF <- fold_pre10
plot(none_D_Dimer_RF)

# 训练对应的最优折(numi)模型&在测试集评价模型

# LDA 联合D二聚体
fold_test <- dataset2[folds[[num.1]],]   

fold_train <- dataset2[-folds[[num.1]],]

fold_pre13 <- lda(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                  data=fold_train) 

fold_predict13 <- predict(fold_pre13, newdata=fold_test, type="response")
fold_predict13 <- as.data.frame(fold_predict13)$posterior.yes

roc13 <- roc(as.numeric(fold_test$DVT),fold_predict13)
plot(roc13, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")

pred_13 <- ifelse(fold_predict13 >= 0.238,2,1)
table(pred_13, fold_test$DVT)
(47+13)/73   ###  0.822  ###

plot(fold_pre13, panel = panel.lda, 
     abbrev = FALSE, xlab = "LD1", ylab = "LD2")

write.csv(fold_pre13$scaling, file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 终 英文变量和赋值 绘制结果图/LDA回归系数有DD.csv")


# LR 联合D二聚体
fold_test <- dataset2[folds[[num.2]],]   

fold_train <- dataset2[-folds[[num.2]],]

fold_pre14 <- glm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                  data=fold_train, family ="binomial")  

fold_predict14 <- predict(fold_pre14,type='response', newdata=fold_test)  

roc14 <- roc(as.numeric(fold_test$DVT),fold_predict14)
plot(roc14, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

pred_14 <- ifelse(fold_predict14 >= 0.263,2,1)
table(pred_14, fold_test$DVT)
(49+13)/73   ###  0.849  ###

library(rms)

attach(fold_train)
dd<-datadist(fold_train)
options(datadist='dd')

nomofit <- lrm(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer
               , data=fold_train, x=T, y=T)

nomogram <- nomogram(nomofit, fun=plogis,fun.at=c(.001, .01, .05, seq(.1,.9, by=.1), .95, .99, .999),lp=F, funlabel="DVT pro")
plot(nomogram)

# CART 联合D二聚体
fold_test <- dataset2[folds[[num.3]],]   

fold_train <- dataset2[-folds[[num.3]],]

fold_pre15 <- rpart(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                    data=fold_train, control = rpart.control(cp="0.001"))  

fold_predict15 <- predict(fold_pre15,type='class', newdata=fold_test)  

library(pROC)
roc15 <- roc(as.numeric(fold_test$DVT),as.numeric(fold_predict15))
plot(roc15, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict15, fold_test$DVT)
(53+8)/72   ###  0.847  ###

plot(fold_pre15, margin=0.05)
text(fold_pre15,  cex=0.8, pretty = 1)

# RF 联合D二聚体
fold_test <- dataset2[folds[[num.4]],]   

fold_train <- dataset2[-folds[[num.4]],]

fold_pre16 <- randomForest(DVT~Age+log2LOS+log2CCI+Chemotherapy+Port_cath+NSAID+Bed+Plaster+VTEHistory+log2WBC+log2DDimer,
                           data=fold_train, mtry=11, importance=TRUE, ntree=1000)  

fold_predict16 <- predict(fold_pre16,type='response', newdata=fold_test)  

roc16 <- roc(as.numeric(fold_test[,59]),as.numeric(fold_predict16))
plot(roc16, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")

table(fold_predict16, fold_test$DVT)
(52+9)/72   ###  0.847  ###

D_Dimer_RF <- fold_pre16
plot(D_Dimer_RF)

write.csv(validation ,file = "F:/004/肿瘤血栓数据库及分析结果/2100例 分析结果 - 终 英文变量和赋值 绘制结果图/validation.csv")

##################  计算Brier Score   ##################

LDA_brier <- mean((lift_results$LDA-as.numeric(lift_results$DVT)+1)^2)
LDA_brier  # 0.1521626
LR_brier <- mean((lift_results$LR-as.numeric(lift_results$DVT)+1)^2)
LR_brier   # 0.1507073
CART_brier <- mean((lift_results$CART-as.numeric(lift_results$DVT)+1)^2)
CART_brier  # 0.1641148
RF_brier <- mean((lift_results$RF-as.numeric(lift_results$DVT)+1)^2)
RF_brier  # 0.150635
SVM_brier <- mean((lift_results$SVM-as.numeric(lift_results$DVT)+1)^2)
SVM_brier  # 0.1628477


LDA_DD_brier <- mean((lift_results$LDA_DD-as.numeric(lift_results$DVT)+1)^2)
LDA_DD_brier  #  0.1485817
LR_DD_brier <- mean((lift_results$LR_DD-as.numeric(lift_results$DVT)+1)^2)
LR_DD_brier   # 0.1460861
CART_DD_brier <- mean((lift_results$CART_DD-as.numeric(lift_results$DVT)+1)^2)
CART_DD_brier  # 0.1524953
RF_DD_brier <- mean((lift_results$RF_DD-as.numeric(lift_results$DVT)+1)^2)
RF_DD_brier  # 0.1390886
SVM_DD_brier <- mean((lift_results$SVM_DD-as.numeric(lift_results$DVT)+1)^2)
SVM_DD_brier  # 0.1553008

library(pROC)   #0.756
roc.LDA <- roc(as.numeric(lift_results$DVT), lift_results$LDA)
plot(roc.LDA, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA")
library(pROC)    #0.752
roc.LR <- roc(as.numeric(lift_results$DVT), lift_results$LR)
plot(roc.LR, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR")
library(pROC)     #0.626
roc.CART <- roc(as.numeric(lift_results$DVT), lift_results$CART)
plot(roc.CART, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of CART")
library(pROC)    #0.733
roc.RF <- roc(as.numeric(lift_results$DVT), lift_results$RF)
plot(roc.RF, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of RF")
library(pROC)    #0.631
roc.SVM <- roc(as.numeric(lift_results$DVT), lift_results$SVM)
plot(roc.SVM, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of SVM")

library(pROC)  #0.773
roc.LDA_DD <- roc(as.numeric(lift_results$DVT), lift_results$LDA_DD)
plot(roc.LDA_DD, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LDA_DD")
library(pROC)  #0.772
roc.LR_DD <- roc(as.numeric(lift_results$DVT), lift_results$LR_DD)
plot(roc.LR_DD , print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of LR_DD")
library(pROC)  #0.690
roc.CART_DD <- roc(as.numeric(lift_results$DVT), lift_results$CART_DD)
plot(roc.CART_DD, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of CART_DD")
library(pROC)  #0.733
roc.RF_DD <- roc(as.numeric(lift_results$DVT), lift_results$RF_DD)
plot(roc.RF, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of RF_DD")
library(pROC)  #0.715
roc.SVM_DD <- roc(as.numeric(lift_results$DVT), lift_results$SVM_DD)
plot(roc.SVM_DD, print.auc = TRUE, auc.polygon = TRUE, legacy.axes = TRUE, 
     grid = c(0.1, 0.2), grid.col = c("green", "red"), max.auc.polygon = TRUE,  
     auc.polygon.col = "skyblue", print.thres = TRUE, xlab = "1-特异度", ylab = "灵敏度",
     main = "ROC of SVM_DD")

######  输出AUC值的P值
library(pROC)
levels(lift_results$DVT)<-c('0','1')
roc.area(as.numeric(as.vector(lift_results$DVT)),roc.SVM_DD$predictor)
