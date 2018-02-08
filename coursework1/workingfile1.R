setwd("/Users/Omar/Documents/Year4/machineLearning/coursework1")


library(rARPACK)
library(philentropy)
# I load the raw csv's
faces.train.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/
                               Faces_Train_Inputs.csv",head=FALSE)
faces.train.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/
                              Faces_Train_Labels.csv",head=FALSE)
faces.test.inputs <- read.csv("./2018_ML_Assessed_Coursework_1_Data/
                              Faces_Test_Inputs.csv",head=FALSE)
faces.test.label <- read.csv("./2018_ML_Assessed_Coursework_1_Data/
                             Faces_Test_Labels.csv",head=FALSE)
# I turn the input values into a list of 320 matrices, each matrix a 112 x 92 value 
# of pixels corresponding to each image .. I need to use lapply again on the 
#result because apply gives the matrices in a weird form
faces.train.inputs.cleaned <- lapply(apply(X=faces.train.inputs, 
                                           MARGIN=1, 
                                           function(x) list(matrix(as.numeric(x), 
                                           nrow = 112))), "[[", 1)
# Here I calculate the average face
avg.face <- Reduce('+', faces.train.inputs.cleaned) / 
  length(faces.train.inputs.cleaned)
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
  # Now I need to compute the first M eigenvectors/ eigenvalues using the 
  # R package rARPACK
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
  # eigenbasis[,i] corresponds to the i'th eigenvector.
  image(matrix(eigenbasis[,i], nrow = 112),useRaster=TRUE, axes=FALSE,main=i)
}
par(mfrow=c(1,1))

# question 2 Choose a single face and project it into a PCA basis 
# for dimension M = 5, 10, 50, then plot the results.##

# initialise the dimensions and face used.
dimensions <- c(5,10,50)
single.face <- 1
means <- as.vector(avg.face) # convert the mean face back into a vector

# iterate through the dimensions considered
for (i in dimensions){
  
  # compute the eigenbasis using the function created
  eigenbasis <- find.pca.basis(i,faces.train.inputs)
  # calculate the projection values in this pca basis
  projection.vals <- t(as.numeric(faces.train.inputs[single.face,]) - 
                         means) %*% eigenbasis
  # calculate the actual face in this basis
  projection.vector <- eigenbasis %*% as.numeric(as.list(projection.vals))
  
  # carry out the plot
  image(matrix(projection.vector, nrow = 112),useRaster=TRUE, 
        axes=FALSE,main=paste("dimension",i,sep=" "))
}

# and here's the original image
image(matrix(as.numeric(faces.train.inputs[single.face,]), nrow = 112), 
      useRaster=TRUE, axes=FALSE, main="Original Image")


## Question 3
# Plot a graph of the mean squared error of each lower dimensional approximation of this 
# chosen face, with the dimensionality plotted along the x-axis. Is there a clear point 
# at which we can choose a good approximation? Discuss how we should choose the appropriate 
# dimensionality of the approximation.

# First I calculate the full pca basis (and all the assosciated eigenvalues)
full.results <- find.pca.basis(320,faces.train.inputs,return.full.results = TRUE)

# the mean square error is equal to the total variance minus the sum of the 
# eigenvalues for components not used 
mses <- (cumsum(full.results$values)[length(full.results$values)] - 
           cumsum(full.results$values) )
plot(mses,xlab="Dimensionality of PCA",ylab="Mean Square Error")

## Question 4
# Write a function implementing a K-nearest neighbour classifier and investigate its use on 
# the face recognition dataset. Make some recommendations regarding how to best set up this 
# algorithm for this particular application.

k.nearest.neighbours <- function(training.data.matrix, training.data.labels, 
                                 testing.data.matrix,K = 4, 
                                 distance.type = "squared_euclidean"){
  # Make sure all the data is numeric
  training.data.matrix <- data.matrix(training.data.matrix)
  testing.data.matrix <- data.matrix(testing.data.matrix)
  training.data.labels <- as.numeric(training.data.labels)
  # Initialise the list which will take the classifications
  classifiers <- c()
  # iterate through every row of the testing matrix
  for (i in 1:dim(testing.data.matrix)[1]){
    # compute the distance of this row of the testing matrix to every other 
    # row in the training set
    all.distances <- apply(training.data.matrix, MARGIN=1, 
                           function(x) distance(rbind(testing.data.matrix[i,],x), 
                                                method=distance.type))
    # sort these distances in increasing order.
    sorted.distances <- sort(all.distances,index.return=TRUE)
    
    # Look at the k closest rows in the training set to this testing row. 
    # Whichever classification comes up the most - is the classification we 
    # will give this particular row.
    all.counts <- tabulate(training.data.labels[sorted.distances$ix[1:K]])
    sorted.counts <- sort(all.counts, index.return=TRUE, decreasing=TRUE)
    ## these are all the voters which share the maximum score
    max.votes <- which(sorted.counts$x == sorted.counts$x[1])
    if (length(max.votes) > 1){
      # if it's a split vote, randomly select one.
      voter <- sample(1:length(max.votes),size=1)
    } else {
      # else, there's only one.
      voter <- 1
    }
    # set the classification.
    classification <- sorted.counts$ix[voter]
    classifiers <- c(classifiers,classification)

  }
  
  return(classifiers)
  
}


# making this work for this specific case
classes <- k.nearest.neighbours(training.data.matrix = faces.train.inputs, 
                                training.data.labels = faces.train.label, 
                                testing.data.matrix = faces.test.inputs,K=5)
classes.actual <- as.integer(faces.test.label)
accuracy <- length(which(classes == classes.actual)) / length(classes.actual)
print(accuracy)
## 91.25% accuracy with no preprocessing, default K= 4
## 91.25% accuracy with K = 2
## 93.75% accuracy with K = 3
## 95% accuracy with K = 1





## TRY pca preprocessing
## compute the eigenbasis then move all of the data into this PCA space
eigenbasis <- find.pca.basis(150,faces.train.inputs)
faces.train.new.basis <- lapply(X = c(1:320),
                                FUN=function(x) t(as.numeric(faces.train.inputs[x,]) 
                                                  - means) %*% eigenbasis)
faces.train.new.basis <- do.call("rbind",faces.train.new.basis)
faces.test.new.basis <- lapply(X = c(1:80),
                               FUN=function(x) t(as.numeric(faces.test.inputs[x,]) -
                                                   means) %*% eigenbasis)
faces.test.new.basis <- do.call("rbind",faces.test.new.basis)
# initialise the list of accuracies
accuracy.sor.list <- c()
accuracy.man.list <- c()
accuracy.sq.list <- c()
for (i in 1:8){ # these are the values of k we will use
  # calculate the classes for each distance type
  classes.sor <- k.nearest.neighbours(training.data.matrix = faces.train.new.basis , 
                                      training.data.labels = faces.train.label, 
                                      testing.data.matrix = faces.test.new.basis,K=i, 
                                      distance.type = "sorensen")
  classes.man <- k.nearest.neighbours(training.data.matrix = faces.train.new.basis, 
                                      training.data.labels = faces.train.label, 
                                      testing.data.matrix = faces.test.new.basis,K=i,
                                      distance.type = "manhattan")
  classes.sq <- k.nearest.neighbours(training.data.matrix = faces.train.new.basis, 
                                       training.data.labels = faces.train.label, 
                                      testing.data.matrix = faces.test.new.basis,K=i, 
                                       distance.type = "squared_euclidean")
  classes.actual <- as.integer(faces.test.label)
  ## add the accuracy to the list of accuracies..
  accuracy.sor <- 100*length(which(classes.sor == classes.actual)) / 
    length(classes.actual)
  accuracy.sor.list <- c(accuracy.sor.list,accuracy.sor)
  accuracy.man <- 100*length(which(classes.man == classes.actual)) / 
    length(classes.actual)
  accuracy.man.list <- c(accuracy.man.list,accuracy.man)
  accuracy.sq <- 100*length(which(classes.sq == classes.actual)) / 
    length(classes.actual)
  accuracy.sq.list <- c(accuracy.sq.list,accuracy.sq)
}

## now check and plot them

# plot all three on same graph
fullmat <- cbind(sorensen=accuracy.sor.list,manhattan=accuracy.man.list,euclidean=accuracy.sq.list)
matplot(fullmat, type = c("b"),pch=1,col = 1:3,ylab="accuracy",xlab="k") #plot
legend("right", legend = colnames(fullmat), col=1:3, pch=1) 







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











## KNN not for submission ##

faces.train.with.labels <- cbind(faces.train.inputs,actual=as.numeric(faces.train.label))

classes2 <- knn(faces.train.inputs,faces.test.inputs,cl=as.numeric(faces.train.label),k =5,l=3)

all(classes[!is.na(classes2)] == classes2[!is.na(classes2)])

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