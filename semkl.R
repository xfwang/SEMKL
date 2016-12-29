library(RcppArmadillo)
library(Rcpp)
library(kernlab)
sourceCpp('spicyfam.cpp')

## for the functions logib1n (logistic loss+lasso),
## logien (logistic loss+elastic nets), lmumb1n(lmum loss+lasso) and lmumen,
## ytr are the response variable with {-1,+1}, ktr is a list, every entry of the
## list is a m*m kernel matrix. The user should calculate the kernel matrix 
## by themself!!!

model_logi <- logib1n(ytr,ktr,alpha,b,rho,c,500,500,cri1,cri2)
predict_logi <- predict(model_logi$alpha,model_logi$b,kva)

model_logien <- logien(ytr,ktr,alpha,b,rho,c,.5,500,500,cri1,cri2)
predict_logien <- predict(model_logien$alpha,model_logien$b,kva)

model_lmum <- lmumb1n(ytr,ktr,alpha,b,rho,c,500,500,cri1,cri2)
predict_lmum <- predict(model_lmum$alpha,model_lmum$b,kva)

model_lmumen <- lmumen(ytr,ktr,alpha,b,rho,c,.5,500,500,cri1,cri2)
predict_lmumen <- predict(model_lmumen$alpha,model_lmumen$b,kva)
