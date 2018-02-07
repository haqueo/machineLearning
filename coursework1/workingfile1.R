setwd("/Users/Omar/Documents/Year4/machineLearning/coursework1")

# I load the raw csv's
faces.train.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Inputs.csv",head=FALSE)
faces.train.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Labels.csv",head=FALSE)

# I turn the input values into a list of 320 matrices, each matrix a 112 x 92 value of pixels corresponding to each image
# .. I need to use lapply again on the result because apply gives the matrices in a weird form
faces.train.inputs.cleaned <- lapply(apply(X=faces.train.inputs, MARGIN=1, function(x) list(matrix(as.numeric(x), nrow = 112))), "[[", 1)

# Here I calculate the average face
avg.face <- Reduce('+', faces.train.inputs.cleaned) / length(faces.train.inputs.cleaned)
image(avg.face)


find.pca.basis <- function(M,X){
  
  n <- dim(X)[1] # The number of images
  
  X <- faces.train.inputs
  M <- 10
  # M - the PCA basis size
  # X - the matrix containing the training set
  
  # Turn the input data into a matrix and transpose it
  X.data.matrix <- data.matrix(t(X))
  
  
  
  # Centralise the data matrix
  means <- rowMeans(X.data.matrix) # calculate row means
  data.matrix.centralised <- X.data.matrix - means %*% t(rep(1,n)) # and subtract
  
  # Calculate the covariance matrix as defined in lectures
  covariance.matrix <- (data.matrix.centralised %*% t(data.matrix.centralised)) / n
  
  # Now I need to compute the first M eigenvectors/ eigenvalues using the R package rARPACK
  results <- eigs_sym(covariance.matrix,k=M,which="LM")
  
  return(results$vectors)
  
  # return the eigenbasis
  image(matrix(as.numeric(faces.train.inputs[1,][c(1:10304)]), nrow = 112),useRaster=TRUE, axes=FALSE)
  # Let's see the first eigenface
  
  
  dim(ev$vectors[1,] %*% data.matrix(X))
  
  
  
}

image(matrix(results$vectors[,1], nrow = 112),useRaster=TRUE, axes=FALSE)

