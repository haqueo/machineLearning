setwd("/Users/Omar/Documents/Year4/machineLearning/coursework1")


library(rARPACK)

# I load the raw csv's
faces.train.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Inputs.csv",head=FALSE)
faces.train.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Train_Labels.csv",head=FALSE)
faces.test.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Test_Inputs.csv",head=FALSE)
faces.test.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/Faces_Test_Labels.csv",head=FALSE)


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

k.nearest.neighbours <- function(training.data.matrix, training.data.labels, testing.data.matrix,
                                 K = 10, distance.type = "euclidean"){
  
  # Make sure all the data is numeric
  training.data.matrix <- data.matrix(training.data.matrix)
  testing.data.matrix <- data.matrix(testing.data.matrix)
  training.data.labels <- as.numeric(training.data.labels)
  
  # Initialise the list which will take the classifications
  classifiers <- c()
  
  # iterate through every row of the testing matrix
  for (i in 1:dim(testing.data.matrix)[1]){
    
    # compute the distance of this row of the testing matrix to every other row in the training set
    all.distances <- apply(training.data.matrix, MARGIN=1, function(x) distance(rbind(testing.data.matrix[i,],x),method=distance.type))
    # sort these distances in increasing order.
    sorted.distances <- sort(all.distances,index.return=TRUE)
    # Look at the k closest rows in the training set to this testing row. Whichever classification comes
    # up the most - is the classification we will give this particular row.
    classification <- sort(tabulate(training.data.labels[sorted.distances$ix[1:K]]), index.return=TRUE,decreasing = TRUE)$ix[1]
    # add it to the list of classifiers
    classifiers <- c(classifiers,classification)

  }
  
  return(classifiers)
  
}


# making this work for this specific case
classes <- k.nearest.neighbours(training.data.matrix = faces.train.inputs,training.data.labels = faces.train.label, testing.data.matrix = faces.test.inputs,K=1)
classes.actual <- as.integer(faces.test.label)
accuracy <- length(which(classes == classes.actual)) / length(classes.actual)
## 75% accuracy with no preprocessing, default K= 4
## 91.25% accuracy with K = 2
## 93.75% accuracy with K = 3
## 95% accuracy with K = 1

## TRY pca preprocessing

eigenbasis <- find.pca.basis(50,faces.train.inputs)

faces.train.new.basis <- lapply(X = c(1:320),FUN=function(x) t(as.numeric(faces.train.inputs[x,]) - means) %*% eigenbasis)
faces.train.new.basis <- do.call("rbind",faces.train.new.basis)

faces.test.new.basis <- lapply(X = c(1:80),FUN=function(x) t(as.numeric(faces.test.inputs[x,]) - means) %*% eigenbasis)
faces.test.new.basis <- do.call("rbind",faces.test.new.basis)

classes <- k.nearest.neighbours(training.data.matrix = faces.train.new.basis,
                                training.data.labels = faces.train.label, 
                                testing.data.matrix = faces.test.new.basis,K=5)
classes.actual <- as.integer(faces.test.label)
accuracy <- length(which(classes == classes.actual)) / length(classes.actual)
print(accuracy)


## k = 1, 96.25%
## k = 2, 92.5%
## k = 3, 93.75%
## k = 4, 91.25%
## k = 5, 92.5%

## now basis of size 50
# k = 1, 95%
# k = 2, 95%
# k = 3, 93.75%
# k = 4, 93.75%
# k = 5, 91.25%
lapply(X = c(1:2),FUN=function(x) t(as.numeric(faces.train.inputs[x,]) - means) %*% eigenbasis)

projection.vector <- eigenbasis %*% as.numeric(as.list(projection.vals))

faces.train.inputs[c(1,2,3),]


##
















 ## testing and other stuff 



training.fake.matrix <- rbind(c(1,2,4,5),c(0,5,10,100),c(1000,2000,3000,4000),c(1,1,1,1),c(2,2,2,2))
training.fake.labels <- rbind(c(1,2,1,2,1))
testing.fake.matrix <- rbind(c(1,2,10,100),c(9,8,7,6))

test <- k.nearest.neighbours(training.fake.matrix,training.fake.labels,testing.fake.matrix,K=3,distance="euclid")






K = 3

training.data.matrix <- training.fake.matrix
training.data.labels <- training.fake.labels
testing.data.matrix <- testing.fake.matrix

all.distances <- apply(training.data.matrix, MARGIN=1, function(x) distance(rbind(testing.data.matrix[1,],x),method="euclidean"))

sorted.distances <- sort(all.distances,index.return=TRUE)
training.data.labels[sorted.distances$ix[1:K]]
classification <- sort(tabulate(training.data.labels[sorted.distances$ix[1:K]]), index.return=TRUE,decreasing = TRUE)$ix[1]
classifiers <- c(classifiers,classification)