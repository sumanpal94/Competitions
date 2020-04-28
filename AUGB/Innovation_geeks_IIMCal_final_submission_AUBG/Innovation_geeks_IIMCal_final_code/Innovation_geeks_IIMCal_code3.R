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

AUBG_Dataset_test <- read_csv("Desktop/PGDBA/1st Semester/AUGB/5dc97593aef65_Testing_Dataset_final.csv")

####### FeatureEngg 1####

#AUBG_Dataset_test$Winner <- as.factor(AUBG_Dataset_test$Winner)
#AUBG_Dataset_test$Winner <- as.numeric(AUBG_Dataset_test$Winner)-1

####### FeatureEngg 2####

AUBG_Dataset_test$weight_class <- as.factor(AUBG_Dataset_test$weight_class)
AUBG_Dataset_test$weight_class <- as.numeric(AUBG_Dataset_test$weight_class)

####### FeatureEngg 3####

AUBG_Dataset_test[c("R_fighter","B_fighter","Referee","date","location","X1","title_bout")] <- NULL

#### defining labels###

#labels=as.matrix(AUBG_Dataset_test['Winner'])

#### 


AUBG_Dataset_tests <- AUBG_Dataset_test[colnames(AUBG_Dataset_test)]

AUBG_Dataset_tests1 <- fastDummies::dummy_cols(AUBG_Dataset_tests)

colnames(AUBG_Dataset_tests1) <- gsub(" ","",colnames(AUBG_Dataset_tests1))

AUBG_Dataset_tests1 <- AUBG_Dataset_tests1[,sapply(AUBG_Dataset_tests1, is.numeric)]

for (i in (3:68)) {
  
  AUBG_Dataset_tests1[i]=AUBG_Dataset_tests1[i]-AUBG_Dataset_tests1[i+66]
  #AUBG_Dataset_tests1[i+66] <- NULL
}

AUBG_Dataset_tests1[69:137] <- NULL
AUBG_Dataset_tests1["B_draw"] <- NULL


#output_vector = Acumen_Attrition_Data_Dummyfied3[,Attrition_Flag] == 1


#sparse_matrix <- SparseM::model.matrix(Attrition_Flag ~ ., data = AUBG_Dataset_tests1)
AUBG_Dataset_tests11 <- AUBG_Dataset_tests1[ , grepl( "att" , names( AUBG_Dataset_tests1 ) ) ]
AUBG_Dataset_tests12<-AUBG_Dataset_tests1  
AUBG_Dataset_tests12[names(AUBG_Dataset_tests11 )] <- NULL
df_test_AUBG=AUBG_Dataset_tests12

#write_csv(AUBG_Dataset_tests12,"data.csv")
#write_csv(data.frame(AUBG_Dataset_test$Winner),"winner.csv")
#df_train_AUBG=AUBG_Dataset_tests1

#df_test_AUBG=df_test_AUBG[,c(importance_matrix[1:30]$Feature)]
X_t <- df_test_AUBG[importance_matrix$Feature[1:45]]
#y <- labels
