# ensure the results are repeatable
require(xgboost)
set.seed(7)
# load the library
library(mlbench)
library(caret)
library(data.table)
library(mlr)
library(h2o)
library(readr)
library(MLmetrics)
library(ROCR)
library(readr)
rf.GIM_Dataset <- read_csv("D:/Suman/PGDBA/Competitions/TVS_Crypto/GIM_Dataset.csv")
names(rf.GIM_Dataset)


rf.GIM_Dataset$V6 <- floor(rf.GIM_Dataset$V6/10000)
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6==11 | rf.GIM_Dataset$V6==11]="DL"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6==12 | rf.GIM_Dataset$V6==13]="Ha"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=14 & rf.GIM_Dataset$V6<=16]="PU"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6==17]="HI"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=18 & rf.GIM_Dataset$V6<=19]="JK"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=20 & rf.GIM_Dataset$V6<=28]="UP"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=30 & rf.GIM_Dataset$V6<=34]="RJ"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=36 & rf.GIM_Dataset$V6<=39]="GJ"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=40 & rf.GIM_Dataset$V6<=44]="MH"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=45 & rf.GIM_Dataset$V6<=49]="MPC"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=50 & rf.GIM_Dataset$V6<=53]="APT"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=56 & rf.GIM_Dataset$V6<=59]="KT"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=60 & rf.GIM_Dataset$V6<=64]="TN"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=67 & rf.GIM_Dataset$V6<=69]="KL"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=70 & rf.GIM_Dataset$V6<=74]="WB"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=75 & rf.GIM_Dataset$V6<=77]="OR"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=78 & rf.GIM_Dataset$V6<=78]="AS"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=79 & rf.GIM_Dataset$V6<=79]="NE"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=80 & rf.GIM_Dataset$V6<=85]="BJ"
rf.GIM_Dataset$V6[rf.GIM_Dataset$V6>=90 & rf.GIM_Dataset$V6<=99]="JK"

#####feature2######

rf.GIM_Dataset$V13[rf.GIM_Dataset$V13=="Y" | rf.GIM_Dataset$V23=="Y"]="Y"
rf.GIM_Dataset$V23 <- NULL

########


rf.GIM_Dataset$V18 <- as.character(rf.GIM_Dataset$V18)
rf.GIM_Dataset$V18 <- factor(rf.GIM_Dataset$V18,levels= c("OTHERS","12TH", "SSC", "UNDER GRADUATE", "GRADUATE", "POST-GRADUATE" ,"PROFESSIONAL"),labels =1:7,ordered=T)
rf.GIM_Dataset$V17 <- as.factor(rf.GIM_Dataset$V17)
rf.GIM_Dataset$V17 <- as.numeric(rf.GIM_Dataset$V17)
rf.GIM_Dataset$V18 <- as.numeric(rf.GIM_Dataset$V18)

rf.GIM_Dataset1 <- rf.GIM_Dataset[,-1]

rf.GIM_Dataset1 <- fastDummies::dummy_cols(rf.GIM_Dataset[,-1])
colnames(rf.GIM_Dataset1) <- gsub(" ","_",colnames(rf.GIM_Dataset1))
colnames(rf.GIM_Dataset1) <- gsub("/","_",colnames(rf.GIM_Dataset1))
colnames(rf.GIM_Dataset1) <- gsub("-","_",colnames(rf.GIM_Dataset1))

#write_csv(rf.GIM_Dataset1,"rf.GIM_Dataset1.csv")
rf.GIM_Dataset1 <- rf.GIM_Dataset1[,sapply(rf.GIM_Dataset1, is.numeric)]

#rf.GIM_Dataset1$V27 <- as.factor(rf.GIM_Dataset1$V27)
#rf.GIM_Dataset1$V27 <- as.numeric(rf.GIM_Dataset1$V27)

#feature 3##

rf.GIM_Dataset1$V16 <- log(rf.GIM_Dataset1$V16)
#feature 4##
rf.GIM_Dataset1$V25 <- rf.GIM_Dataset1$V25 + rf.GIM_Dataset1$V26
rf.GIM_Dataset1$V26 <- NULL
rf.GIM_Dataset1$V13_N <- NULL
rf.GIM_Dataset1$V9_NonTechnical <- NULL
rf.GIM_Dataset1$V24_RENT <- NULL

labels <- as.matrix(rf.GIM_Dataset1["V27"])


df_train <- rf.GIM_Dataset1[-grep('V27',colnames(rf.GIM_Dataset1))]

X <- df_train[,c(importance_matrix[1:20]$Feature)]
y <- labels

dtrain <- xgb.DMatrix(data.matrix(X), label = labels)
cv <- xgb.cv(data = dtrain, nrounds = 50, nthread = 6, nfold = 10, metrics = list("error@0.4","auc"),
             max_depth = 15, eta = 0.05, objective = "binary:logistic",lambda= 0.5)
print(cv)
print(cv, verbose=TRUE)

xgb <- xgboost(data = data.matrix(X), 
               label = y, 
               eta =  0.05,
               max_depth =15, 
               nround= 50, 
               eval_metric = "logloss",
               objective = "binary:logistic",
               nthread = 6,
               lambda= 0.5
)

y_pred <- predict(xgb,data.matrix(X))
y_pred

pred <- prediction(y_pred , y) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

y_pred[y_pred>=.35]=1
y_pred[y_pred<.35]=0

table(y,y_pred)

AUC(y_pred,y)



names <- dimnames(X)[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:20])
xgb.plot.tree(feature_names = names, model = xgb, n_first_tree = 2)
