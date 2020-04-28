library(readr)
GIM_Dataset <- read_csv("~/Downloads/5dbabd3a3e7e8_GIM_Dataset/GIM_Dataset.csv")
names(GIM_Dataset)
GIM_Dataset1 <- GIM_Dataset[,-1]
# GIM_Dataset1 <- fastDummies::dummy_cols(GIM_Dataset[,-1])
# colnames(GIM_Dataset1) <- gsub(" ","_",colnames(GIM_Dataset1))
# colnames(GIM_Dataset1) <- gsub("/","_",colnames(GIM_Dataset1))
# colnames(GIM_Dataset1) <- gsub("-","_",colnames(GIM_Dataset1))

labels=as.matrix(GIM_Dataset1['V27'])
df_train=GIM_Dataset1[-grep('V27',colnames(GIM_Dataset1))]
#df_train=df_train[,c(importance_matrix$Feature[1:30])]
X <- df_train
y <- labels

dtrain <- xgb.DMatrix(data.matrix(X), label = labels)
cv <- xgb.cv(data = dtrain, nrounds = 30, nthread = 4, nfold = 10, metrics = list("rmse","auc"),
             max_depth = 40, eta = 0.05, objective = "binary:logistic")
print(cv)
print(cv, verbose=TRUE)

xgb <- xgboost(data = data.matrix(X), 
               label = y, 
               eta = 0.5,
               max_depth = 30, 
               nround=50, 
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "logloss",
               objective = "binary:logistic",
               nthread = 3
)

y_pred <- predict(xgb,data.matrix(X))
y_pred

pred <- prediction(y_pred , y) 
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

y_pred[y_pred>=.5]=1
y_pred[y_pred<.5]=0

table(y,y_pred)

AUC(y_pred,y)



names <- dimnames(X)[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:30])
xgb.plot.tree(feature_names = names, model = xgb, n_first_tree = 2)
