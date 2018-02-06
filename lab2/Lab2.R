setwd("/Users/Omar/Documents/Year4/machineLearning/lab2")

train.data <- read.table("Data_Train.txt")
test.data <- read.table("Data_Test.txt")

x <- train.data[,c(1,2)]
y <- train.data[,3]

xt <- test.data[,c(1,2)]
yt <- test.data[,3]

classification.rates <- c()

for (i in 1:20){
  
  classification.rates <- c(classification.rates,100-CompLab2.Classification(x,y,xt,yt,i))
  
}

plot(classification.rates)
