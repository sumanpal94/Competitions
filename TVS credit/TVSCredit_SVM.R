library(data.table)
library(mlr)
library(h2o)
library(readr)
library(e1071)
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

classifier =svm(formula = V27~.,
                data = rf.GIM_Dataset1,
                type = 'C-classification',
                kernel='polynomial',
                show.info = T
                )

y_pred <- predict(classifier,new_data=rf.GIM_Dataset1[,-c("V27")])
table(rf.GIM_Dataset1$V27,y_pred )
AUC(rf.GIM_Dataset1$V27,y_pred)

