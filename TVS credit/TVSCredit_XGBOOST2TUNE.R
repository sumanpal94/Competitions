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
library(tidyverse) # data manipulation
library(mlr)       # ML package (also some data manipulation)
library(knitr)     # just using this for kable() to make pretty tables
library(xgboost)
library(readr)
rf.GIM_Dataset <- read_csv("Desktop/PGDBA/1st Semester/TVS credit/original dataset/5dbabd3a3e7e8_GIM_Dataset/GIM_Dataset.csv")
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

write_csv(rf.GIM_Dataset1,"rf.GIM_Dataset1.csv")
rf.GIM_Dataset1 <- rf.GIM_Dataset1[,sapply(rf.GIM_Dataset1, is.numeric)]

rf.GIM_Dataset1$V27 <- as.factor(rf.GIM_Dataset1$V27)
#rf.GIM_Dataset1$V27 <- as.numeric(rf.GIM_Dataset1$V27)

#income log##

rf.GIM_Dataset1$V16 <- log(rf.GIM_Dataset1$V16)


labels <- as.matrix(rf.GIM_Dataset1["V27"])
df_train <- rf.GIM_Dataset1[-grep('V27',colnames(rf.GIM_Dataset1))]

X <- df_train
y <- labels

traintask <- makeClassifTask(data = rf.GIM_Dataset1 ,target = "V27") 

set.seed(1)
# Create an xgboost learner that is classification based and outputs
# labels (as opposed to probabilities)
xgb_learner <- makeLearner(
  "classif.xgboost",
  predict.type = "response",
  par.vals = list(
    objective = "binary:logistic",
    eval_metric = "error",
    nrounds = 200,
    show.info = T
  )
)

# Create a model
xgb_model <- train(xgb_learner, task = traintask)

xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = .1, upper = .5),
  # L2 regularization - prevents overfitting
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
)

control <- makeTuneControlRandom(maxit = 10)

# Create a description of the resampling plan
resample_desc <- makeResampleDesc("CV", iters = 10)

tuned_params <- tuneParams(
  learner = xgb_learner,
  task = traintask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control
  ,show.info = T
)

xgb_tuned_learner <- setHyperPars(
  learner = xgb_learner,
  par.vals = tuned_params$x
)

# Re-train parameters using tuned hyperparameters (and full training set)
xgb_model <- train(xgb_tuned_learner, traintask)

# Make a new prediction
y_pred  <- predict(xgb_model, traintask)

pred <- prediction(y_pred$data$response , y_pred$data$truth) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

y_pred[y_pred>=.5]=1
y_pred[y_pred<.5]=0

table(y,y_pred)

AUC(y_pred$data$response , y_pred$data$truth)



names <- dimnames(X)[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:30])
xgb.plot.tree(feature_names = names, model = xgb, n_first_tree = 2)
