library(ggplot2)
library(reshape2)
library(ggpubr)
library(caTools)
library(corrplot)
library(MVA)
library(Boruta)
library(caret)
library(MASS)
library(rpart)
library(rpart.plot)
library(randomForest)

# Load Parkinson Data Set from UCI Directory
parkinsons_data = read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",sep = ",", header = TRUE, stringsAsFactors = FALSE)
#parkinsons_data <- read.table("C:/Users/DPHARIKA/Documents/Wayne/My Project/parkinsons_updrs.data")
str(parkinsons_data)
summary(parkinsons_data)

#park_data <- parkinsons_data



#Exploratory Data analysis
#Analysis to understand the data and its distribution 

par(mfrow = c(2,6))
hist(parkinsons_data$age, xlab="age",main="Histogram : Age", col = bluered(14))
hist(parkinsons_data$sex,xlab= "Sex",main="Histogram : Sex", col = bluered(14))
hist(parkinsons_data$test_time, xlab="test_time",main="Histogram : Test_Time", col = bluered(14))
hist(parkinsons_data$total_UPDRS, xlab="total_UPDRS",main="Histogram of total_UPDRS", col = bluered(14))
hist(parkinsons_data$motor_UPDRS, xlab="motor_UPDRS",main="Histogram : motor_UPDRS", col = bluered(14))
hist(parkinsons_data$Jitter..., xlab="JitterPerc",main="Histogram : JitterPerc", col = bluered(14))
hist(parkinsons_data$Shimmer, xlab="Shimmer",main="Histogram : Shimmer", col = bluered(14))
hist(parkinsons_data$NHR, xlab="NHR",main="Histogram : NHR", col = bluered(14))
hist(parkinsons_data$parkinsons_data$RPDE, xlab="RPDE",main="Histogram : RPDE", col = bluered(14))
hist(parkinsons_data$DFA, xlab="DFA",main="Histogram : DFA", col = bluered(14))
hist(parkinsons_data$PPE, xlab="PPE",main="Histogram : PPE", col = bluered(14))






#To check if the data is normally distributed:  Chi-Square plot:
------------------------------------------------------------------------
library(mvnormtest)
data <- parkinsons_data
cm <- colMeans(data)
S <- cov(data)
d <- apply(data, 1, function(data) t(data - cm) %*% solve(S) %*% (data - cm))

# Chi-Square plot:
plot(qchisq((1:nrow(data) - 1/2) / nrow(data), df = ncol(data)), 
sort(d),
xlab = expression(paste(chi[22]^2, " Quantile")), 
ylab = "Ordered distances")



#Data Cleaning:

#Anomaly detection and treatment:
na = colSums(is.na(parkinsons_data))
na


#### Collinear Distribution

library(corrplot)
corrplot(cor(parkinsons_data), type="full", method ="color", 
         title = "Parkinson Correlation Distribution",
         mar=c(0,0,1,0), tl.cex= 0.8, outline= T, tl.col="blue")
#Outlier Detection & Removal:
#Scattered plot to look into data distribution

par(mfrow = c(1,3))
windows()
plot(jitter(total_UPDRS)~., parkinsons_data)


#To Draw boxplots for understanding the outliers of the data

boxplot(parkinsons_data)

#To Draw Bivariate Boxplot against total_UPDRS
library(MVA)
par(mfrow = c(1,3))
bvbox(parkinsons_data[,6:7],     xlab = "total_UPDRS", ylab = "Jitter")
bvbox(parkinsons_data[,c(6,12)], xlab = "total_UPDRS", ylab = "Shimmer")
bvbox(parkinsons_data[,c(6,18)], xlab = "total_UPDRS", ylab = "NHR")
bvbox(parkinsons_data[,c(6,20)], xlab = "total_UPDRS", ylab = "RPDE")
bvbox(parkinsons_data[,c(6,21)], xlab = "total_UPDRS", ylab = "DFA")
bvbox(parkinsons_data[,c(6,22)], xlab = "total_UPDRS", ylab = "PPE")

#Convex hull method

hull1 <- chull(parkinsons_data[,6:7])
parkhull <- match(lab <- rownames(parkinsons_data[hull1,]),rownames(parkinsons_data))
plot(parkinsons_data[,6:7], xlab = "total_UPDRS", ylab = "Jitter")
polygon(parkinsons_data$Jitter...[hull1]~parkinsons_data$total_UPDRS[hull1])
text(parkinsons_data[parkhull,6:7], labels = lab, pch=".", cex = 0.9)

#Outlier Reduction

outlier <- parkinsons_data[-hull1,]
dim(outlier)
dim(parkinsons_data)

hull2 <- chull(outlier[,c(6,12)])
parkinsons_data <- outlier[-hull2,]

hull3 <- chull(parkinsons_data[,c(6,18)])
outlier <- parkinsons_data[-hull3,]

hull4 <- chull(outlier[,c(6,20)])
parkinsons_data <- outlier[-hull4,]

hull5 <- chull(parkinsons_data[,c(6,21)])
outlier <- parkinsons_data[-hull5,]

hull6 <- chull(outlier[,c(6,22)])
parkinsons_data <- outlier[-hull6,]

dim(parkinsons_data)
summary(parkinsons_data)

#FEATURE SELECTION:
# Preprocessing: Using Boruta to identify important Attributes

library(Boruta)
Borutatrain <- Boruta(age ~ ., data = parkinsons_data, doTrace = 2)
print(Borutatrain)

# Plot Important Attributes " No unimpotant attribute detected"

plot(Borutatrain, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(Borutatrain$ImpHistory),function(i)
  Borutatrain$ImpHistory[is.finite(Borutatrain$ImpHistory[,i]),i])
names(lz) <- colnames(Borutatrain$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(Borutatrain$ImpHistory), cex.axis = 0.7)




#Dimensionality Reduction: PCA
pca_parkinsons_data<-park_data
pca_parkinsons_data[["subject."]]= NULL
pca_parkinsons_data[["motor_UPDRS"]]=NULL
pca_parkinsons_data[["total_UPDRS"]]=NULL


pc=prcomp(pca_parkinsons_data,center = TRUE,scale. = TRUE)
plot(pc,type="l",main = "PCA:SCREE PLOT")
component1=pc$x[,1]
component2=pc$x[,2]
pc_data =cbind(component1,component2)
plot(component1,component2,main = "SCATTER PLOT AFTER PCA",col = "red")

#K-MEANS CLUSTERING
Parkinsons_kmeans = kmeans(pc_data,2)
plot(pc_data, col = Parkinsons_kmeans$cluster, main = " K-MEANS CLUSTERING WITH PCA DATA")
points(Parkinsons_kmeans$centers, col = c("yellow"), pch = 17, cex= 2)
rect.hclust(pk_den,k=2,border = "blue")

#HIERARCHIAL CLUSTERING 
set.seed(666)
library (cluster)
p = sample(length(pc_data),400) 
pk_den = hclust(dist(p),method = "complete")
cut_clust=cutree(pk_den,k=2)
windows()
plot(pk_den,main = "DENDROGRAM WITH COMPLETE LINKAGE")
rect.hclust(pk_den,k=2,border = "blue")





#CLASSIFICATION: RANDOM FOREST

# remove motor UPDRS to make the data single variate regression
p<-data.frame(parkinsons_data[,-5])
str(p)

#Training a model on the data
##Set up trainning and test data sets:
set.seed(350)
indx = sample(1:nrow(p), as.integer(0.9*nrow(p)))
indx[1:10]

p_train = p[indx,]
p_test = p[-indx,]

library(randomForest)
rf<- randomForest(total_UPDRS~ ., data =p_train )
rf

#Check importance of each predictor:

library(randomForest)
library(ggplot2)
importance(rf)

#Plot of Important Variables
varImpPlot(rf)


#Evaluating Model Performance
pred<-predict(rf,p_test,type='response')
head(pred)
summary(pred)
summary(p$total_UPDRS)

cor(pred,p_test$total_UPDRS)

MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}
MAE(pred, p_test$total_UPDRS)


#To Improve Model Performance
library(randomForest)
rf1<- randomForest(total_UPDRS~ ., data =p_train,mtry=10)
rf1


importance(rf1)
varImpPlot(rf1)
pred1<-predict(rf1,p_test,type='response')
head(pred1)
head(p_test$total_UPDRS)
summary(pred1)
summary(p$total_UPDRS)
#Compare the correlation between predicted and actual total UPDRS.

cor(pred1,p_test$total_UPDRS)

#Mean absolute error between predicted and actual values:
  
MAE(pred1, p_test$total_UPDRS)



#CLASSIFICATION: SVM


library(e1071)
#linear
tune.out=tune(svm,total_UPDRS~.,data=p_train,kernel="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10),
                          epsilon=c(0.1,0.5,1)))
summary(tune.out) 
bestmod=tune.out$best.model
summary(bestmod) 
predict2 <- predict(bestmod,p_test, type= 'response')
svmRMSE2 <- sqrt(mean((predict2-p_test$total_UPDRS)^2))
svmRMSE2

plot(p_test$total_UPDRS,predict2, xlab = "Observed",ylab = "Predicted")

#radial
tune.out=tune(svm,total_UPDRS~.,data=p_train,kernel="radial",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10),
                          gamma=c(0.1,0.3,0.4,0.5,1,2)))
summary(tune.out) 
bestmod=tune.out$best.model
summary(bestmod)
predict2 <- predict(bestmod,p_test, type= 'response')
svmRMSE2 <- sqrt(mean((predict2-p_test$total_UPDRS)^2))
svmRMSE2

plot(p_test$total_UPDRS,predict2, xlab = "Observed",ylab = "Predicted")

#poly
tune.out=tune(svm,total_UPDRS~.,data=p_train,kernel="polynomial",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10),
                          degree=c(2,3,4)))
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod)
predict2 <- predict(bestmod,p_test, type= 'response')
svmRMSE2 <- sqrt(mean((predict2-p_test$total_UPDRS)^2))
svmRMSE2 
plot(p_test$total_UPDRS,predict2, xlab = "Observed",ylab = "Predicted")
