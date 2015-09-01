traingrouplasso=function(y,x,kernels=rep("l",m),parameters=rep(1,m),penalty=1) {
  ##group lasso generalization
  ##y should be (1&-1) rather than (1&0)
  ##p means norm, penalty is the parameter of loosen variable in svm
  ##for kernel types, "p"means polynomial, "g" means Gaussian
  ##default kernel:linear kernel   default parameter of Gausssian kernel:1
  ##now no code for Gaussian kernel, iteration condition can be changed
  
  library(kernlab)
  
  #initialize kernels and parameters
  n=dim(y)[1] #sample size
  m=length(x) #number of kernels
  gamma=rep(1/m,m)
  temp=rep(0,m)
  epsilon=rep(1,m)
  f=rep(0,m)
  iteration=0
  kk=matrix(rep(0,n*n),nrow=n,ncol=n)
  maxa=0
  
  #construct the dataset
  # x=list()
  k=mapply(function(x,y,z){
    if (z=="l") {
      return (kernelMatrix(vanilladot(),as.matrix(x)))
    } else if (z=="p") {
      return (kernelMatrix(polydot(degree = y),as.matrix(x)))
    } else if (z=="g") {
      return(kernelMatrix(rbfdot(sigma = y),as.matrix(x)))
    } else if (z=="t") {
      return(kernelMatrix(tanhdot(scale=1,offset=1),as.matrix(x)))
    }
  },x,parameters,kernels,SIMPLIFY = FALSE)
  
  yy=y%*%t(y)
  
  #group-lasso
  while (max(epsilon)>0.0001) { 
    iteration=iteration+1
    kk=Reduce('+',mapply("*", k, gamma,SIMPLIFY = FALSE) )
    h=kk*yy
    model=ipop(rep(-1,n),h,t(y),0,rep(0,n),rep(penalty,n),0)
    a=primal(model)
    f=unlist(mapply(function(r,q){
      sqrt(r^2*t(a*y)%*%q%*%(a*y))
    },gamma,k,SIMPLIFY = FALSE))
    temp=gamma
    sumf=sum(f) #^(2*p/(1+p))
    gamma=sapply(f,function(x) x/sumf)
    epsilon=abs(temp-gamma)
  }
  gamma=gamma*((gamma>=0.0001)+1-1)

  j=match(a[(a>0)&(a<penalty)][1],a)
  b=y[j]-sum(a*y*kk[,j])
  model=list("alpha"=a,"y"=y,"x"=x,"b"=b,"gamma"=gamma,"iteration"=iteration,"parameters"=parameters,"kernels"=kernels,"epsilon"=max(epsilon))
  return(model)
}
