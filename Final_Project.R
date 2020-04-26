# load Parkinson Data Set

# Loading the dataset from local
parkinsons<- read.csv("C:/Users/DPHARIKA/Documents/Wayne/My Project/parkinsons.data")
dummy_data = parkinsons
k= parkinsons
str(k)

#To check if Multivariate data normally distributed:

library(mvnormtest)
x <- subset(parkinsons, select = -name )
cm <- colMeans(x)
S <- cov(x)
d <- apply(x, 1, function(x) t(x - cm) %*% solve(S) %*% (x - cm))

#### Check correlations between the variables
x <- subset(k,select =-name)
library(corrplot)
corrplot(cor(x), type="full", method ="color", title = "Parkinson Correlation", mar=c(0,0,1,0), tl.cex= 0.8, outline= T, tl.col="indianred4")


#Doing some Exploratory analysis
park <-parkinsons
park.m=melt(park[,-1],id.vars = "status")
p <- ggplot(data = park.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=status))
p + facet_wrap( ~ variable, scales="free")



#Preprocessing: Chi-square plot

plot(qchisq((1:nrow(x) - 1/2) / nrow(x), df = ncol(x)), 
     sort(d),
     xlab = expression(paste(chi[22]^2, 
                             " Quantile")), 
     ylab = "Ordered distances")


# Data Cleaning and Outlier Removal:

#### Check null values

missing <- apply(parkinsons, 2, function(x) 
  round(100 * (length(which(is.na(x))))/length(x) , digits = 1))
as.data.frame(missing)

# 
# 
# ##
# 
# boxplot(x$MDVP.Fo.Hz)
# boxplot(x$MDVP.Fhi.Hz)
# boxplot(x$MDVP.Flo.Hz)
# boxplot(x$MDVP.Jitter...)
# boxplot(x$MDVP.Jitter.Abs.)
# boxplot(x$MDVP.RAP)
# boxplot(x$MDVP.PPQ)
# boxplot(x$Jitter.DDP)
# boxplot(x$MDVP.Shimmer)
# boxplot(x$MDVP.Shimmer.dB.)
# boxplot(x$Shimmer.APQ3)
# boxplot(x$Shimmer.APQ5)
# boxplot(x$MDVP.APQ)
# boxplot(x$Shimmer.DDA)
# boxplot(x$NHR)
# boxplot(x$HNR)
# boxplot(x$status)
# boxplot(x$RPDE)
# boxplot(x$DFA)
# boxplot(x$spread1)
# boxplot(x$spread2)
# boxplot(x$D2)
# boxplot(x$PPE)
# 
# 






# Preprocessing: Using Boruta to identify important Attributes

library(Boruta)
Borutatrain <- Boruta(MDVP.Fo.Hz. ~ ., data = x, doTrace = 8)
# Borutatrain <- Boruta(name ~ ., data = x, doTrace = 2)
print(Borutatrain)

# Plot Important Attributes " No unimportant attribute detected"

plot(Borutatrain, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(Borutatrain$ImpHistory),function(i)
  Borutatrain$ImpHistory[is.finite(Borutatrain$ImpHistory[,i]),i])
names(lz) <- colnames(Borutatrain$ImpHistory)
print(names(lz))
Labels <- sort(sapply(lz,median))
print(Labels)
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(Borutatrain$ImpHistory), cex.axis = 0.7)

# Apply PCA method for filtering attrbutes

library(ggfortify)
my_pca <- prcomp(pdata[,2:24], center=TRUE, scale=TRUE)
autoplot(my_pca, data = pdata)



# Apply KNN Classifier to the data set

library(class)
n_pdata <- pdata[-1]
normalize <- function(x) {  return ((x - min(x)) / (max(x) - min(x))) }
npdata <- as.data.frame(lapply(n_pdata[2:23], normalize))
npdata_train <- npdata[1:150,]
npdata_test <- npdata[151:195,]
npdata_train_labels <- pdata[1:150, 18]
npdata_test_labels <- pdata[151:195, 18]
npdata_test_pred <- knn(train = npdata_train, test = npdata_test,cl = npdata_train_labels, k=14)
table(npdata_test_labels, npdata_test_pred)



