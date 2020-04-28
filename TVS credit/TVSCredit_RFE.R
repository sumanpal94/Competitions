# ensure the results are repeatable

set.seed(7)
# load the library
library(mlbench)
library(caret)
library(data.table)
library(mlr)
library(h2o)
library(readr)
library(MLmetrics)

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
rf.GIM_Dataset1$V27 <- as.numeric(rf.GIM_Dataset1$V27)

labels <- rf.GIM_Dataset1["V27"]
df_train <- rf.GIM_Dataset1[-grep('V27',colnames(rf.GIM_Dataset1))]

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(as.matrix(df_train), as.matrix(labels), sizes=c(1:45), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))