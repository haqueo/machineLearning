setwd("/Users/Omar/Documents/Year4/machineLearning/coursework1")


library(rARPACK)

# I load the raw csv's
faces.train.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Inputs.csv",head=FALSE)
faces.train.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Labels.csv",head=FALSE)

# I turn the input values into a list of 320 matrices, each matrix a 112 x 92 value of pixels corresponding to each image
# .. I need to use lapply again on the result because apply gives the matrices in a weird form
faces.train.inputs.cleaned <- lapply(apply(X=faces.train.inputs, MARGIN=1, function(x) list(matrix(as.numeric(x), nrow = 112))), "[[", 1)

# Here I calculate the average face
avg.face <- Reduce('+', faces.train.inputs.cleaned) / length(faces.train.inputs.cleaned)
image(avg.face)


find.pca.basis <- function(M,X, return.full.results = FALSE){

  n <- dim(X)[1] # The number of images
  
  # Turn the input data into a matrix and transpose it
  X.data.matrix <- data.matrix(t(X))
  
  # Centralise the data matrix
  means <- rowMeans(X.data.matrix) # calculate row means
  data.matrix.centralised <- X.data.matrix - means %*% t(rep(1,n)) # and subtract
  
  # Calculate the covariance matrix as defined in lectures
  covariance.matrix <- (data.matrix.centralised %*% t(data.matrix.centralised)) / n
  
  # Now I need to compute the first M eigenvectors/ eigenvalues using the R package rARPACK
  results <- eigs_sym(covariance.matrix,k=M,which="LM")
  
  if (return.full.results){
    return(results)
  } else{
    return(results$vectors)
  }
  
}

eigenbasis <- find.pca.basis(5,faces.train.inputs)

par(mfrow=c(2,3))
for (i in 1:5){
  
  image(matrix(eigenbasis[,i], nrow = 112),useRaster=TRUE, axes=FALSE)
}
par(mfrow=c(1,1))

# question 2 Choose a single face and project it into a PCA basis 
# for dimension M = 5, 10, 50, then plot the results.##

dimensions <- c(5,10,50)
single.face <- 1
means <- as.vector(avg.face)

for (i in dimensions){
  
  eigenbasis <- find.pca.basis(i,faces.train.inputs)
  projection.vals <- t(as.numeric(faces.train.inputs[single.face,]) - means) %*% eigenbasis
  projection.vector <- eigenbasis %*% as.numeric(as.list(projection.vals))
  
  
  image(matrix(projection.vector, nrow = 112),useRaster=TRUE, axes=FALSE)
}

image(matrix(as.numeric(faces.train.inputs[1,]), nrow = 112),useRaster=TRUE, axes=FALSE)


## Question 3
# Plot a graph of the mean squared error of each lower dimensional approximation of this 
# chosen face, with the dimensionality plotted along the x-axis. Is there a clear point 
# at which we can choose a good approximation? Discuss how we should choose the appropriate 
# dimensionality of the approximation.

# First I calculate the full pca basis (and all the assosciated eigenvalues)
full.results <- find.pca.basis(320,faces.train.inputs,return.full.results = TRUE)

# the mean square error is equal to the total variance minus the sum of the eigenvalues for 
# components not used (and itself)
mses <- (cumsum(full.results$values)[length(full.results$values)] - cumsum(full.results$values) )
plot(mses,xlab="Dimensionality of PCA",ylab="Mean Square Error")

## Question 4
# Write a function implementing a K-nearest neighbour classifier and investigate its use on 
# the face recognition dataset. Make some recommendations regarding how to best set up this 
# algorithm for this particular application.


k.nearest.neighbours <- function(training.data.matrix, training.data.labels ){
  
  
}







