library(readr)

Data22 <- read_csv("~/Desktop/PGDBA/1st Semester/Brain-a-lytics/feature generation/Data22.csv")

names(Data22)

histogram(Data22$value_mean_cust_id)


value_mean_cust_id_hist <- hist(Data22$value_mean_cust_id,freq = FALSE)
value_mean_cust_idEXC0_hist  <- hist(Data22$value_mean_cust_id[Data22$value_mean_cust_id>0],freq =FALSE)

plot( value_mean_cust_id_hist, col=rgb(0,0,1,1/4), xlim=c(0,1))  # first histogram
plot( value_mean_cust_idEXC0_hist, col=rgb(1,0,0,1/4), xlim=c(0,1), add=T)  # second

pairs(Data22[,c()])

Data22_num<- Data22[,sapply(Data22, is.numeric)]

Data22_num_cor <- cor(Data22_num)

heatmap(Data22_num_cor(is.na(Data22_num_cor)))

