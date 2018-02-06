CompLab2.Classification <- function(x, y, xt, yt, PolyOrder) {

# x and y are column vectors containing the training inputs and targets respectively
# xt and yt are column vectors containing the testing inputs and targets respectively
# PolyOrder is the order of polynomial model used for fitting
 
NumOfDataPairs_Train <- nrow(x)
NumOfDataPairs_Test  <- nrow(xt)

# First construct design matrix of given order
X       <- rep(1, NumOfDataPairs_Train)
dim(X)  <- c(NumOfDataPairs_Train, 1)
Xt      <- rep(1, NumOfDataPairs_Test)
dim(Xt) <- c(NumOfDataPairs_Test, 1)

for (n in 1:PolyOrder){
    X  = cbind(X, x^n)
    Xt = cbind(Xt, xt^n)
}

NumOfParas = ncol(X)

# Set the prior precision
alpha = 10

# Newton routine to find MAP values of Theta
# Fix number of steps to 10 and initial estimate to Theta = 0
N_Steps = 10;
Theta   = matrix(0, nrow=NumOfParas, ncol=1);

for (n in 1:N_Steps){    
    # Newton Step
    P     = 1./(1 + exp(-X %*% Theta))
    A     = diag(P[,1]*(1-P[,1]))
    H     = t(X) %*% A %*% X + diag(NumOfParas)/alpha
    Theta = solve(H, t(X)%*%(A%*%X%*%Theta + y - P) )
        
    # Compute new likelihood and unormalised posterior values for training data
    f             = X%*%Theta # train
    logPrior      = sum( dnorm(Theta, rep(0, NumOfParas), rep(alpha, NumOfParas), log=TRUE) )
    logLikelihood = t(f)%*%y - sum(log(1+exp(f))) # training likelihood
    logJointProb  = logLikelihood + logPrior

    cat("Iteration ", n, ": Log joint probability is ", logJointProb, "\n")
}

cat("\nFinal parameters are: ", Theta, "\n")


# Compute new likelihood and unormalised posterior values for testing data
ft                 = Xt%*%Theta # test
logPrior_Test      = sum( dnorm(Theta, rep(0, NumOfParas), rep(alpha, NumOfParas), log=TRUE) )
logLikelihood_Test = t(ft)%*%yt - sum(log(1+exp(ft))) # testing likelihood
logJointProb_Test  = logLikelihood_Test + logPrior_Test

# Calculate percentage of training errors
Train_Error = 100 - 100*sum( (1/(1+exp(-X%*%Theta)) > 0.5) == y)/NumOfDataPairs_Train

# Calculate percentage of test errors
Test_Error = 100 - 100*sum( (1/(1+exp(-Xt%*%Theta)) > 0.5) == yt)/NumOfDataPairs_Test

cat("Percentage training error: ", Train_Error, "\n")
cat("Percentage testing error: ", Test_Error, "\n")

return(Test_Error)

}