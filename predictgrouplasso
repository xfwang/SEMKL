predictgrouplasso=function(xnew,model) {
N=dim(xnew[[1]])[1]
n=dim(model[["y"]])[1]
m=length(model[["x"]])
product=list()
fushion=matrix(,n,N) #rep(0,n)
yhat=rep(0,N)
predict=rep(0,N)

product=mapply(function(x,y,z,w,v) {
  if (v=="l") {
    return(kernelMatrix(vanilladot(),as.matrix(x),as.matrix(y))*z)
  } else if (v=="p") {
    return(kernelMatrix(polydot(degree = w),as.matrix(x),as.matrix(y))*z)
  } else if (v=="g") {
    return(kernelMatrix(rbfdot(sigma = w),as.matrix(x),as.matrix(y))*z)
  } else if (v=="t") {
    return(kernelMatrix(rbfdot(scale=1,offset=1),as.matrix(x),as.matrix(y))*z)
  }
},
model[["x"]],xnew,model[["gamma"]],model[["parameters"]],model[["kernels"]],
SIMPLIFY = FALSE)

fushion=Reduce('+',product)
yhat=t(fushion)%*%(model[["alpha"]]*model[["y"]])+model[["b"]]
predict=sign(yhat)

result=list("yhat"=yhat,"predict"=predict)
return(result)
}
