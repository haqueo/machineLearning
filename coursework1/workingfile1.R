setwd("/Users/Omar/Documents/Year4/machineLearning/coursework1")

faces.train.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Inputs.csv",head=FALSE)

faces.train.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Labels.csv",head=FALSE)

matrix1 <- matrix(as.numeric(faces.train.inputs[1,][c(1:10304)]), nrow = 112)



start_time <- Sys.time()
matrices <- lapply(apply(X=faces.train.inputs, MARGIN=1, function(x) list(matrix(as.numeric(x), nrow = 112))), "[[", 1)
end_time <- Sys.time()

print("apply takes")
print(end_time - start_time)

start_time <- Sys.time()
matrices2 <- alply(faces.train.inputs, 1, function(x) matrix(as.numeric(x), nrow = 112))
end_time <- Sys.time()

print("alply takes")
print(end_time - start_time)



image(matrix(as.numeric(faces.train.inputs[1,][c(1:10304)]), nrow = 112),useRaster=TRUE, axes=FALSE)