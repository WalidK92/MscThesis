#The code below is to predict the undrained shear strength of cohesive soils.
#Prediction is accomplished using Machine learning algorithms such as Linear Regression, Random Forest
#and Gradient Boosting. An ensemble of all the aforementioned algorithms is also used for prediction. 
#For the purpose of clear and quick execution, please follow the instructions placed the length of the code.

#author Walid Khalid

#Ensure the following packages are installed before attempting to run the code. 1) Caret 2) Metrics

#--------*--------*BEGINNING OF SOURCE CODE*--------*--------*#
# read data from csv into list
library(readr)
library(Metrics)
library(caret)

#Please ensure that the Data is saved in a '.csv' format
#Copy the path to where your Data has been saved in your PC in place of the existing path.


cat('Loading Data==============>\n\n')
Trial <- read_csv("C:/Users/Walid/Desktop/Thesis_1/Codes/Dataset.csv")

#Data Cleaning, and Drawing new subset from the Orginal Data
NewData <- subset(Trial, Cu<200)
summary(NewData)

# Data Split into Training and Testing Data
cat('Splitting Data into 80 / 20 Train Test Sets==============>\n\n')
set.seed(122)
samp <- sample(nrow(NewData), 0.8 * nrow(NewData))
train <- NewData[samp, ]
test <- NewData[-samp, ]

#Building a Single Predictor Model
cat('Fitting a Single Predictor Linear Model==============>\n\n')
LmMod1 <- lm(Cu ~ N60, data = train)
predLm1 <- predict(LmMod1, newdata = test)

#Building a Multiple Predictor Model
cat('Fitting a Single Predictor Linear Model==============>\n\n')
LmMod2 <- lm(Cu ~ ., data = train)
predLm2 <- predict(LmMod2, newdata = test)

# Building and Predicting Using The Linear Model
cat('Fitting a Linear Model==============>\n\n')
LmMod <- lm(Cu ~N60+wn , data = train)
predLm <- predict(LmMod, newdata=test)

# Building, Tuning and Predicting Using Random Forest

# Tuning the Random Forest 
cat('Tuning Random Forest Model Parameters using CV==============>\n\n')
control <- trainControl(method="repeatedcv", number=10, repeats=5, search="grid")
tunegrid <- expand.grid(mtry=c(1:5)) 
rf_gridsearch <- train(Cu~., data=train, method="rf", 
                       metric='RMSE', tuneGrid=tunegrid, trControl=control)
bestMtry <- rf_gridsearch['bestTune'][[1]][,1]
print(rf_gridsearch)
plot(rf_gridsearch)

# Building and Predicting Using the Random Forest
cat('Training the Random Forest Model and Predicting on the Test Set==============>\n\n')
rfMod <- randomForest(Cu ~ . , data =train, mtry=bestMtry, importance=TRUE)
predRF <- predict(rfMod, newdata=test)

# Building, Tuning and Predicting Using Gradient Boosting

# Tuning the Gradient Boosting Model
cat('Tuning Gradient Boosting Model Parameters using CV==============>\n\n')
control <- trainControl(method="repeatedcv", number=10, repeats=5, search="grid")
tunegrid <- expand.grid(shrinkage=seq(0.0001,1,by=0.01), n.trees= 500, 
                        n.minobsinnode=10, interaction.depth=seq(1,5,1))
gbm_gridsearch <- train(Cu~ ., data=train, method="gbm", 
                        metric='RMSE', tuneGrid=tunegrid, trControl=control)

bestNTrees <- gbm_gridsearch['bestTune'][[1]][,1]
bestIntDepth <- gbm_gridsearch['bestTune'][[1]][,2]
bestShrinkage <- gbm_gridsearch['bestTune'][[1]][,3]

print(gbm_gridsearch)
plot(gbm_gridsearch)

# Building and Predicting Using Gradient Boosting
cat('Training the Gradient Boosting Model and Predicting on the Test Set==============>\n\n')
gbmMod <- gbm(formula = Cu ~ ., distribution = "gaussian", data = train, 
              n.trees = bestNTrees,
              shrinkage = bestShrinkage,
              interaction.depth = bestIntDepth,
              keep.data = TRUE )
predGBM <- predict(gbmMod, test, n.trees = 500)

# Print RMSE Indices of All The Models
RMSELm1 <- RMSE(test$Cu, predLm1)
RMSELm <- RMSE(test$Cu, predLm) 
RMSERF <- RMSE(test$Cu, predRF) 
RMSEGBM <- RMSE(test$Cu, predGBM)

# Combining All Models
# Checking Correlations of Predicted Values
cat('Checking Correlations between the Models==============>\n\n')
PredictedValues <- data.frame(predLm = rnorm(39), predRF = rnorm(39), predGBM = rnorm(39))
cor(PredictedValues)

# Building Stacked Model Using Weighted Averages
cat('Building a Stacked Layer==============>\n\n')

# Finding the weights for the Stacking layer
gb = abs(RMSEGBM - RMSELm) + 1
rf = abs(RMSERF - RMSELm) + 1
lm = abs(RMSELm -RMSELm) +1
gb_rf_lm <-sum(c(gb,rf,lm))

EnsembleMod <- predGBM*(gb / gb_rf_lm) + predRF*(rf / gb_rf_lm) + predLm*(lm/gb_rf_lm)
RMSE_Stacking <- RMSE(test$Cu, EnsembleMod)

#--------*--------*END OF SOURCE CODE*--------*--------*#

#--------*--------*BEGINNING OF PLOTTING FUNCTIONS*--------*--------*#

#Plotting RMSE Indices of All Models
H <- c(RMSELm, RMSERF, RMSEGBM, RMSE_Stacking)
M <- c('Linear Model', "Random Forest", "GBM", "Stacking")
Colors <- c("gray20", "gray45", "gray80", "White")
barplot(H,names.arg = M,xlab = "Models",ylab = "RMSE",col = Colors,
        main = "RMSE Plots",border = "black")

#Plotting Linear Model Predicted vs Measured Values
plot(LmMod, pch = 16, which =1, main = NULL)
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

#Plotting Random Forest Model Predicted vs Measured Values
x = predRF
y = test$Cu
mydf <- data.frame(x = x, y = y)
model <- lm(y ~ x, data = mydf)
plot(predRF ~ test$Cu, data = mydf, xlab="Measured Cu (kPa) ",ylab="Predicted Cu (kPa)",   xlim=c(0,200),ylim=c(0,200))
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

#Plotting Gradient Boosting Predicted vs Measured Values
x = predGBM
y = test$Cu
mydf <- data.frame(x = x, y = y)
model <- lm(y ~ x, data = mydf)
plot(predGBM ~ test$Cu, data = mydf, xlab="Measured Cu (kPa) ",ylab="Predicted Cu (kPa)",   xlim=c(0,200),ylim=c(0,200))
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

#Plotting Ensemnle Mode Predicted vs Measured Values
x = EnsembleMod
y = test$Cu
mydf <- data.frame(x = x, y = y)
model <- lm(y ~ x, data = mydf)
plot(EnsembleMod ~ test$Cu, data = mydf, xlab="Measured Cu (kPa) ",ylab="Predicted Cu (kPa)",   xlim=c(0,200),ylim=c(0,200))
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

#--------*--------*END OF PLOTTING FUNCTIONS*--------*--------*#

#--------*--------*BEGINNING OF USER BASED CODE*--------*--------*#

# Please Copy the Path of Desired Data to be Used for Predictions into "..." below.
UserData <- read_csv ("C:/Users/Walid/Desktop/Thesis_1/Codes/Dataset.csv")

#Predictions Using Different Models Developed in Source Code above.

#Using Linear Model
Predictions_LinearModel <- predict(LmMod, newdata = UserData)

#Using Random Forest
Predictions_RandomForest <- predict(rfMod, newdata = UserData)

#Using Gradient Boosting
Predictions_GradientBoosting <- predict(gbmMod, UserData, n.trees = 500)

#Using Stacked Model Model
Predictions_StackedModel <- Predictions_GradientBoosting*(gb / gb_rf_lm) + Predictions_RandomForest*(rf / gb_rf_lm) + Predictions_LinearModel*(lm/gb_rf_lm)

#Exporting Predicted Values Individually and Saving them in 'DOCUMENTS' Folder.
write.csv(data.frame(Predictions_LinearModel), "Predictions_LinearModel.csv")
write.csv(data.frame(Predictions_RandomForest), "Predictions_RandomForest.csv")
write.csv(data.frame(Predictions_GradientBoosting), "Predictions_GradientBoosting.csv")
write.csv(data.frame(Predictions_StackedModel), "Predictions_StackedModel.csv")

#--------*--------*END OF USER BASED CODE*--------*--------*#