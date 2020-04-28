library(data.table)
library(mlr)
library(h2o)
library(readr)
library(MLmetrics)

library(readr)
rf.GIM_Dataset <- read_csv("~/Downloads/5dbabd3a3e7e8_GIM_Dataset/GIM_Dataset.csv")
names(rf.GIM_Dataset)
 
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


# rf.GIM_Dataset1 <- rf.GIM_Dataset1[,sapply(rf.GIM_Dataset1, is.numeric)]

rf.GIM_Dataset1$V27 <- as.factor(rf.GIM_Dataset1$V27)

traintask <- makeClassifTask(data = rf.GIM_Dataset1 ,target = "V27") 
rdesc <- makeResampleDesc("CV",iters=10L)
rf.lrn <- makeLearner("classif.randomForest")
rf.lrn$par.vals <- list(ntree = 100L, importance=TRUE)
r <- resample(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc), show.info = T)


getParamSet(rf.lrn)

#set parameter space
params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 10),makeIntegerParam("nodesize",lower = 10,upper = 50))

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=5L)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 5L)

#start tuning
tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)
#[Tune] Result: mtry=2 : nodesize=23 : acc.test.mean=0.858