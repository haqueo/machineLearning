data <- CompLab1.Generate_Data()

y <- data[,2]
x <- data[,1]
PolyOrder <- 10

computeMSE <- function(x, y, PolyOrder) {
  
  # x and y are vectors containing the inputs and targets respectively
  # PolyOrder is the order of polynomial model used for fitting
  
  
  
  NumOfDataPairs <- length(x)  
  
  # First construct design matrix of given order
  X      <- rep(1, NumOfDataPairs)
  dim(X) <- c(NumOfDataPairs, 1)
  
  # Initialise MSE
  mses <- rep(1,PolyOrder)
  
  rss.vals <- rep(1,PolyOrder)
  esquaredvals <- c()
  
  
  for (n in 1:PolyOrder){
    X = cbind(X, x^n)
    mses[n] <- 1/n
    current.e <- y - X[,1:(n+1)] %*% solve( t(X[,1:(n+1)]) %*% X[,1:(n+1)] , t(X[,1:(n+1)]) %*% y)
    esquaredvals <- c(esquaredvals, t(current.e) %*% current.e)
  }
  
  
  mses <- esquaredvals * mses
  
  plot(mses,xlab="Degree of polynomial",ylab="Mean Squared Error",main="Effect of increasing function complexity on mse")
  
  
  
  # leave one out cross validation 
  
  for (n in 1:NumOfDataPairs){
    
    # Create training design matrix and target data, leaving one out each time
    Train_X <- X[-n, ]
    Train_y <- y[-n]
    
    # Create testing design matrix and target data
    Test_X <- X[n, ]
    Test_y <- y[n]
    
    # Learn the optimal paramerers using MSE loss
    
    # Initialise MSE
    mses <- rep(1,PolyOrder)
    
    rss.vals <- rep(1,PolyOrder)
    esquaredvals <- c()
    
    
    for (n in 1:PolyOrder){
      X = cbind(X, x^n)
      mses[n] <- 1/n
      current.e <- y - X[,1:(n+1)] %*% solve( t(X[,1:(n+1)]) %*% X[,1:(n+1)] , t(X[,1:(n+1)]) %*% y)
      esquaredvals <- c(esquaredvals, t(current.e) %*% current.e)
    }
    
    
    
    
    
    
    
    
    Paras_hat <- solve( t(Train_X) %*% Train_X , t(Train_X) %*% Train_y)
    Pred_y    <- Test_X %*% Paras_hat;
    
    # Calculate the MSE of prediction using training data
    CV[n]     <- (Pred_y - Test_y)^2
    
  }
  
  
  
  
}