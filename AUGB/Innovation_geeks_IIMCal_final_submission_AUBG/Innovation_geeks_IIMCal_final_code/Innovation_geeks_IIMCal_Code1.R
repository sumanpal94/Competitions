library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(openxlsx)
library(readxl)
library(readxl)
library(zoo)
library(MLmetrics)
library(Deducer)
library(pROC)
library(broom)
library(ROCR)

require(xgboost)
set.seed(100)

AUBG_Dataset <- read_csv("Desktop/PGDBA/1st Semester/AUGB/5dc975b0250f1_Training_Dataset.csv")

####### FeatureEngg 1####

AUBG_Dataset$Winner <- as.factor(AUBG_Dataset$Winner)
AUBG_Dataset$Winner <- as.numeric(AUBG_Dataset$Winner)-1

####### FeatureEngg 2####

AUBG_Dataset$weight_class <- as.factor(AUBG_Dataset$weight_class)
AUBG_Dataset$weight_class <- as.numeric(AUBG_Dataset$weight_class)

####### FeatureEngg 3####

AUBG_Dataset[c("R_fighter","B_fighter","Referee","date","location","X1","title_bout")] <- NULL

#### defining labels###

labels=as.matrix(AUBG_Dataset['Winner'])

#### 


AUBG_Datasets <- AUBG_Dataset[-grep('Winner',colnames(AUBG_Dataset))]

AUBG_Datasets1 <- fastDummies::dummy_cols(AUBG_Datasets)

colnames(AUBG_Datasets1) <- gsub(" ","",colnames(AUBG_Datasets1))

AUBG_Datasets1 <- AUBG_Datasets1[,sapply(AUBG_Datasets1, is.numeric)]

for (i in (3:68)) {
  
  AUBG_Datasets1[i]=AUBG_Datasets1[i]-AUBG_Datasets1[i+66]
  #AUBG_Datasets1[i+66] <- NULL
}

  AUBG_Datasets1[69:137] <- NULL
  AUBG_Datasets1["B_draw"] <- NULL


#output_vector = Acumen_Attrition_Data_Dummyfied3[,Attrition_Flag] == 1


#sparse_matrix <- SparseM::model.matrix(Attrition_Flag ~ ., data = AUBG_Datasets1)
  AUBG_Datasets11 <- AUBG_Datasets1[ , grepl( "att" , names( AUBG_Datasets1 ) ) ]
  AUBG_Datasets12<-AUBG_Datasets1  
  AUBG_Datasets12[names(AUBG_Datasets11 )] <- NULL
  df_train_AUBG=AUBG_Datasets12
  
#write_csv(AUBG_Datasets12,"data.csv")
#write_csv(data.frame(AUBG_Dataset$Winner),"winner.csv")
#df_train_AUBG=AUBG_Datasets1
#df_train=df_train[,c(importance_matrix[1:30]$Feature)]
X <- df_train_AUBG
y <- labels


### CV##

dtrain <- xgb.DMatrix(data.matrix(X), label = labels)
cv <- xgb.cv(data = dtrain, nrounds = 500, num_class = 3,booster = "gbtree",nthread = 4, nfold = 10, metrics = list("merror"),
             max_depth =3, eta = 0.007, lambda=.0001,alpha=.0003,objective = "multi:softmax",, showsd = TRUE, stratified = TRUE, print_every_n = 5, early_stop_round = 20, maximize = FALSE, prediction = TRUE)
print(cv)
print(cv, verbose=TRUE)

#### XGboost###

xgb <- xgboost(data = data.matrix(X), 
               label = y, 
               eta = 0.007,
               max_depth = 3, 
               nround=180, 
               colsample_bytree = 0.5,
               seed = 1,
               booster = "gbtree",
               eval_metric = c("merror"),
               objective = "multi:softprob",
               nthread = 4,
               num_class = 3,
               show=T
)

y_pred2 <- predict(xgb,data.matrix(X_t),reshape=T)
y_pred2 = as.data.frame(y_pred2)

colnames(y_pred2) = c(0,1,2)


y_pred2$prediction = apply(y_pred2,1,function(x) colnames(y_pred2)[which.max(x)])
#y_pred2$label = levels(y)[y+1]

mean(y==y_pred2$prediction)

table(y,y_pred2$prediction)

names <- dimnames(X)[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:45])
xgb.plot.tree(feature_names = names, model = xgb, trees = 2)

output_AUBG <- y_pred2$prediction

