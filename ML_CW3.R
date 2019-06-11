library(ggplot2)
library(nnet)
library(plyr)
library(data.table)
library(CORElearn)
library(caret)
library(dummies)
library (VIM)
library(Boruta)
library(doParallel)
library(corrplot)
library(DMwR)
library(e1071)
library(kernlab)
library(pROC)
library(AppliedPredictiveModeling)

na = c('?','NA', 'null','Null','NULL')

# setwd("/home/mcarn003/MACHINE_LEARNING_3")
# df = as.data.table(read.table(file.path('/home/mcarn003/MACHINE_LEARNING_3','allhyper.data.csv'),sep = ',', header = FALSE, stringsAsFactors = TRUE,na.strings = na))
# df_TEST = as.data.table(read.table(file.path('/home/mcarn003/MACHINE_LEARNING_3','allhyper.test.csv'), sep = ',', header = FALSE, stringsAsFactors = TRUE, na.strings = na))

setwd("/Users/Mazy/Desktop/ML_3/")
#http://archive.ics.uci.edu/ml/datasets/thyroid+disease

#Read data in data.table format (LOCAL PATH)
df = as.data.table(read.table(file.path('/Users/Mazy/Desktop/ML_3/', 'allhyper.data.csv'),
                              header=FALSE,
                              sep=',', 
                              stringsAsFactors=TRUE,
                              na.strings=na))

colnames(df) = c("age","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick",
                 "pregnant", "thyroid_surgery", "treatment","query_hypothyroid", "query_hyperthyroid",
                 "lithium", "goitre", "tumor","hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured",
                 "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured","TBG",
                 "referral_source","CLASS")
dim(df)
head(df)
# transform all "?" to 0s
df[is.na(df)] = 0
head(df)
str(df)
summary(df)
head(df$CLASS) #Only the two classes are needed for the classification. Everything after "." will be removed
df$CLASS = gsub("\\..*","",df$CLASS)
head(df$CLASS)
unique(df$CLASS)
df$CLASS = as.factor(df$CLASS)

levels(df$CLASS)
levels(df$CLASS)[4] = "T3_toxic"; levels(df$CLASS)
summary(df$CLASS)

#Check if any participants have no diagnosis 
sum(is.na(df$CLASS))

df = data.frame(df$CLASS,df[,-30])
names(df)[1] = "CLASS" ; names(df)

missing_values  = sapply(df, function(col) sum(is.na(col)))/ 2800 
hist(missing_values)

featureNames = names(df); featureNames
featureNames = featureNames[missing_values <= 0.2]
featureNames

head(df$TBG_measured) #* TBG_measured is a factor that has only one level.
levels(df$TBG_measured) = c(0,1) 
head(df$TBG_measured)

#Parallel Process
cl = makeCluster(detectCores()); cl
registerDoParallel(cl)

relief = attrEval(CLASS ~ ., 
                  data=df,
                  estimator="ReliefFbestK", #*
                  ReliefIterations=50)
df$TBG = NULL

perm = permuteRelief(x=df[,-1],
                     y=df[,1],
                     estimator="ReliefFbestK",
                     nperm=500)

set.seed(3291)
pred.90 = names(sort(abs(perm$standardized[which(abs(perm$standardized)>=1.65)]), decreasing=T))
print(pred.90)
#[1] "FTI"                "TT4"                "T3"                 "query_hyperthyroid" "tumor"             
#[6] "pregnant"           "sex"                "T4U_measured"       "FTI_measured"       "TT4_measured"      
#[11] "referral_source"    "T4U"                "T3_measured"
new_df = data.frame(df$CLASS,df[,pred.90])
names(new_df)[1] = "CLASS"
summary(new_df)

#save resulting dataset
write.table(new_df, file='selected_features.csv', sep=',', na='', row.names=F)

#````````````````````````````````````````````````````````````````````````````````````````````````````
#Using the relief takes a lot of time for feature selection hence the features selected beforehand are put in a csv file called 'selected_features.csv' to speed up the running of this project.
#Read data in data.table format (LOCAL PATH)
# new_df = as.data.table(read.table(file.path('/Users/Mazy/Desktop/ML_3/', 'selected_features.csv'),
#                                  sep=',',
#                                  header=TRUE,
#                                  stringsAsFactors=TRUE,
#                                  na.strings=na))
#````````````````````````````````````````````````````````````````````````````````````````````````````
featureNames = names(new_df); featureNames

featureNames_class = factor(sapply(new_df, class))
featureNames_class

numeric_featureNames = featureNames[featureNames_class != 'factor']
numeric_featureNames 

nominal_featureNames = featureNames[featureNames_class == 'factor']
nominal_featureNames 

numb_of_columns = ncol(new_df); numb_of_columns
numb_of_rows = nrow(new_df); numb_of_rows 

#.......................IMPUTATION OF MISSING VALUES.................................
sum(is.na(new_df))
# 110 missing values
rowSums(is.na(new_df)) 
#Lets double check if there are any NAs
summary(new_df)
#*The sex (M, F) column seems to have NA values. Lets get rid of these rows as it may intefere with our predictions.
#If we continue with the rows with missing values and impute them, it may influence our results.
new_df = new_df[complete.cases(new_df$sex), ]
summary(new_df$sex)

sum(is.na(new_df)) #No missing values in the whole dataframe

#------------------ Dummification -----------------------------
set.seed(3291)
str(new_df)
#* 1's indicate the encoding as per the label and the 0's are the other options.
dummifing_ = dummyVars("CLASS~.", data = new_df)
dummified_ = predict(dummifing_, newdata = new_df)
head(dummified_)
dim(dummified_)

dummified_df = data.frame(new_df$CLASS, dummified_)
head(dummified_df)
#Renaming new_df.CLASS to CLASS
names(dummified_df)[1] = "CLASS"
#CLASS is moved to the first column from the last column
head(dummified_df)

#................................CENTERING AND SCALING ALL THE COLUMNS.....................

# Preparing the Nominal columns
set.seed(3291)

nearZeroVar(dummified_df, names = TRUE) #We are not getting rid of these features as they seem important
dummified_df$referral_source.STMW = NULL
dummified_df$referral_source.SVHD = NULL

#Centering and scalling the features and knnImpute for Imputation just incase
df_cent_sca_preprocessing = preProcess(dummified_df, method = c("center","scale","knnImpute"))
df_cent_sca = predict(df_cent_sca_preprocessing, dummified_df)
head(df_cent_sca)
anyNA(df_cent_sca)
summary(df_cent_sca)

split_data = data.frame(df_cent_sca)

#.......................TRAINING AND VALIDATION DATASETS..............................
# Shuffle the data
split_data = split_data[sample(nrow(split_data)),]


#Splitting the data into training and testing
split = createDataPartition(split_data$CLASS, p = 0.70)[[1]]
df_TRAIN = split_data[split,]; dim(df_TRAIN)
VALIDATION = split_data[-split,]; dim(VALIDATION)

#.......................BALANCING THE TRAINING SET ONLY................................
# Balance the classes in the data 

prop.table(table(df_TRAIN$CLASS))  # returns %propotion 0f each class

#ploting the Upselling proportion
barplot(prop.table(table(df_TRAIN$CLASS)), col = rainbow(2), ylim = c(0,0.7), main = 'Class Distribution')  #plots the imbalance Upselling
length(df_TRAIN$CLASS)
dim(df_TRAIN) 

#using UPSAMPLING
#https://topepo.github.io/caret/subsampling-for-class-imbalances.html
#?upSample
df_trainBalanced = upSample(df_TRAIN, df_TRAIN$CLASS, list = FALSE) #upsamples Upselling
head(df_trainBalanced)
dim(df_trainBalanced) #7628   19
summary(df_trainBalanced)

prop.table(table(df_trainBalanced$Class))  # returns %propotion 0f each balanced class    
barplot(prop.table(table(df_trainBalanced$Class)), col = rainbow(2), ylim = c(0,0.7), main = 'Class Distribution') #plots the balanced labels
head(df_trainBalanced$Class)

#Removing CLASS
names(df_trainBalanced)
TrainSet = df_trainBalanced[-1]; names(TrainSet) #¢¢

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPARING THE TEST SET~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_TEST = as.data.table(read.table(file.path('/Users/Mazy/Desktop/ML_3/', 'allhyper.test.csv'),
                                   header=FALSE, 
                                   sep=',', 
                                   stringsAsFactors=TRUE,
                                   na.strings=na))

dim(df_TEST)
head(df_TEST)
# transform all "?" to 0s
df_TEST[is.na(df_TEST)] = 0
head(df_TEST)
str(df_TEST) 
summary(df_TEST)
head(df_TEST$V30) #Only the two classes are needed for the classification. Everything after "." will be removed
df_TEST$V30 = gsub("\\..*","",df_TEST$V30)
df_TEST$V30 = as.factor(df_TEST$V30)
head(df_TEST$V30)
unique(df_TEST$V30)

colnames(df_TEST) = c("age","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick",
                      "pregnant", "thyroid_surgery", "treatment","query_hypothyroid", "query_hyperthyroid",
                      "lithium", "goitre", "tumor","hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured",
                      "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured","TBG",
                      "referral_source","CLASS")

df_TEST = df_TEST[,c("CLASS","FTI","TT4","T3","query_hyperthyroid", "tumor","pregnant","sex","T4U_measured","FTI_measured","TT4_measured",      
                     "referral_source","T4U","T3_measured")]
levels(df_TEST$CLASS)
levels(df_TEST$CLASS)[5] = "T3_toxic"; levels(df_TEST$CLASS)
summary(df_TEST$CLASS)
sum(is.na(df_TEST$CLASS))

#Remove participants without secondary toxic as there was'nt a SECONDARY TOXIC THYROID DISEASE
df_TEST = subset(df_TEST, CLASS != "secondary toxic")
table(df_TEST$CLASS)
table(droplevels(df_TEST$CLASS))
df_TEST$CLASS = droplevels(df_TEST$CLASS)
featureNames = names(df_TEST); featureNames

featureNames_class = factor(sapply(df_TEST, class))
featureNames_class

numeric_featureNames = featureNames[featureNames_class != 'factor']
numeric_featureNames 

nominal_featureNames = featureNames[featureNames_class == 'factor']
nominal_featureNames 

numb_of_columns = ncol(df_TEST); numb_of_columns
numb_of_rows = nrow(df_TEST); numb_of_rows 

#Parallel PKappaess
cl = makeCluster(detectCores()); cl
registerDoParallel(cl)

#.......................IMPUTATION OF MISSING VALUES.................................
sum(is.na(df_TEST))
summary(df_TEST)
rowSums(is.na(df_TEST)) 
#Lets double check if there are any NAs
summary(df_TEST)
#*The sex (M, F) column seems to have NA values. Lets get rid of these rows as it may intefere with our predictions.
#If we continue with the rows with missing values and impute them, it may influence our results.
df_TEST = df_TEST[complete.cases(df_TEST$sex), ]
summary(df_TEST$sex)

sum(is.na(df_TEST)) #No missing values in the whole dataframe

#------------------ Dummification -----------------------------
set.seed(3291)
str(df_TEST)
#* 1's indicate the encoding as per the label and the 0's are the other options.
dummified_TEST = predict(dummifing_, newdata = df_TEST)
head(dummified_TEST)
dim(dummified_TEST)

dummified_df_TEST = data.frame(df_TEST$CLASS, dummified_TEST)
dummified_df_TEST$referral_source.STMW = NULL
dummified_df_TEST$referral_source.SVHD = NULL
head(dummified_df_TEST)
#Renaming df_TEST.CLASS to CLASS
names(dummified_df_TEST)[1] = "CLASS"
#CLASS is moved to the first column from the last column
names(dummified_df_TEST)

#................................CENTERING AND SCALING ALL THE COLUMNS.....................

# Preparing the Nominal columns
set.seed(3291)

#Centering and scalling the features and knnImpute for Imputation just incase
df_cent_sca_TEST = predict(df_cent_sca_preprocessing, dummified_df_TEST)
head(df_cent_sca_TEST)
anyNA(df_cent_sca_TEST)
summary(df_cent_sca_TEST)

levels(df_cent_sca_TEST$CLASS)
levels(split_data$CLASS)[4] = "T3_toxic"; levels(split_data$CLASS)

TestSet = data.frame(df_cent_sca_TEST); head(TestSet)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING AND VALIDATING MODELS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(3291)
require(compiler)

multiClassSummary = cmpfun(function (data, lev = NULL, model = NULL){
  
  #Load Libraries
  require(Metrics)
  require(caret)
  
  #Check data
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  
  #Calculate custom one-vs-all stats for each class
  prob_stats <- lapply(levels(data[, "pred"]), function(class){
    
    #Grab one-vs-all data for the class
    pred <- ifelse(data[, "pred"] == class, 1, 0)
    obs  <- ifelse(data[,  "obs"] == class, 1, 0)
    prob <- data[,class]
    
    #Calculate one-vs-all AUC and logLoss and return
    cap_prob <- pmin(pmax(prob, .000001), .999999)
    prob_stats <- c(auc(obs, prob), logLoss(obs, cap_prob))
    names(prob_stats) <- c('Kappa', 'logLoss')
    return(prob_stats) 
  })
  prob_stats <- do.call(rbind, prob_stats)
  rownames(prob_stats) <- paste('Class:', levels(data[, "pred"]))
  
  #Calculate confusion matrix-based statistics
  CM <- confusionMatrix(data[, "pred"], data[, "obs"])
  
  #Aggregate and average class-wise stats
  #Todo: add weights
  class_stats <- cbind(CM$byClass, prob_stats)
  class_stats <- colMeans(class_stats)
  
  #Aggregate overall stats
  overall_stats <- c(CM$overall)
  
  #Combine overall with class-wise stats and remove some stats we don't want 
  stats <- c(overall_stats, class_stats)
  stats <- stats[! names(stats) %in% c('AccuracyNull', 
                                       'Prevalence', 'Detection Prevalence')]
  
  #Clean names and return
  names(stats) <- gsub('[[:blank:]]+', '_', names(stats))
  return(stats)
  
})

#https://www.r-bloggers.com/error-metrics-for-multi-class-problems-in-r-beyond-accuracy-and-kappa/

levels(TrainSet$Class)
anyNA(TrainSet)


ctrl = trainControl(method='cv',                 #  Cross Validation
                    classProbs=TRUE,             # compute class probabilities
                    summaryFunction = multiClassSummary,
                    number = 10,             
                    verboseIter = TRUE,
                    allowParallel=TRUE,
                    savePredictions = TRUE)

table(TrainSet$Class)
table (VALIDATION$CLASS)
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''''
set.seed(3291)

nB_grid = data.frame(fL=c(0,0.5,1.0), usekernel = TRUE, adjust=c(0,0.5,1.0))

naiveBayes.model = train(Class ~.,
                         data = TrainSet,
                         method='nb',
                         metric='Kappa',
                         trControl=ctrl,
                         tuneGrid = nB_grid)
#bestTune
#fL usekernel adjust
#2 0.5      TRUE    0.5

head(naiveBayes.model)
head(naiveBayes.model$pred)
naiveBayes.model$results

#Confusion matrix
confusionMatrix(data = naiveBayes.model$pred$pred,
                reference = naiveBayes.model$pred$obs) 
#Kappa =  0.7158   Accuracy : 0.7868 

set.seed(3291)
naiveBayes.model_Prediction_Classes = predict(naiveBayes.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(naiveBayes.model_Prediction_Classes)
confusionMatrix(naiveBayes.model_Prediction_Classes,VALIDATION$CLASS) 
# Kappa = 0.0484   Accuracy : 0.5764   

naiveBayes.model_Prediction_Probs = predict(naiveBayes.model, newdata = VALIDATION, type = "prob")
multiclass.roc(VALIDATION$CLASS, naiveBayes.model_Prediction_Probs)$auc 
#AUC = Multi-class area under the curve:0.9234

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''2'''''''''''''''''''''''''''''''''''''''''''''''''''''
set.seed(3291)
rpart.tuneGrid <- expand.grid(cp = seq(0, 0.05, 0.005))

rpart.model = train(Class~., 
                    data=TrainSet, 
                    method="rpart", 
                    tuneLength = 30,
                    metric = 'Kappa',
                    trControl=ctrl,
                    tuneGrid = rpart.tuneGrid)
ggplot(rpart.model) + theme_bw()
print(rpart.model) 
head(rpart.model$pred)
#Confusion matrix
confusionMatrix(data = rpart.model$pred$pred, 
                reference = rpart.model$pred$obs)
#Kappa = 0.9657   Accuracy =  0.9743 

set.seed(3291)
rpart.model_Prediction_Classes = predict(rpart.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(rpart.model_Prediction_Classes)
confusionMatrix(rpart.model_Prediction_Classes,VALIDATION$CLASS) 
#Kappa -  0.7163 Accuracy - 0.9809

rpart.model_Prediction_Classes_prob = predict(rpart.model, newdata = VALIDATION, type="prob")
multiclass.roc(VALIDATION$CLASS, rpart.model_Prediction_Classes_prob)$auc 
#Multi-class area under the curve: 0.9886   

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''3'''''''''''''''''''''''''''''''''''''''''''''''''''''

set.seed(3291)
c50Grid = expand.grid(.trials = c(1:9),
                       .model = c("tree", "rules"),
                       .winnow = c(TRUE, FALSE))
C5.0.model = train(Class~., 
                   data=TrainSet, 
                   method="C5.0", 
                   tuneLength = 30,
                   metric = 'Kappa',
                   trControl=ctrl,
                   tuneGrid = c50Grid ) #Fitting trials = 7, model = rules, winnow = FALSE on full training set

ggplot(C5.0.model) + theme_bw()
print(C5.0.model) 
head(C5.0.model$pred)
#Confusion matrix
confusionMatrix(data = C5.0.model$pred$pred, 
                reference = C5.0.model$pred$obs)
#Kappa with relief  0.9927          Accuracy : 0.9945 

set.seed(3291)
C5.0.model_Prediction_Classes = predict(C5.0.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(C5.0.model_Prediction_Classes)
confusionMatrix(C5.0.model_Prediction_Classes,VALIDATION$CLASS) 
#Kappa with relief  0.6516       Accuracy : 0.9797

C5.0.model_Prediction_Classes_prob = predict(C5.0.model, newdata = VALIDATION, type="prob")
multiclass.roc(VALIDATION$CLASS, C5.0.model_Prediction_Classes_prob)$auc 
#Multi-class area under the curve:  0.9756   

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''4'''''''''''''''''''''''''''''''''''''''''''''''''''''
set.seed(3291)

Svmgrid = expand.grid(sigma = c(.01, .015, 0.2),
                      C = 2^(seq(-4,4)))
Svm.model = train(Class~., 
                  data=TrainSet, 
                  method="svmRadial", 
                  tuneGrid = Svmgrid,
                  metric = 'Kappa',
                  fit = FALSE,
                  trControl=ctrl)

ggplot(Svm.model) + theme_bw()
print(Svm.model) 
head(Svm.model$pred)
#Confusion matrix
confusionMatrix(data = Svm.model$pred$pred, 
                reference = Svm.model$pred$obs)
#Kappa with relief  0.8885       Accuracy : 0.9164 with sigma = 0.2 and C = 0.5

set.seed(3291)
Svm.model_Prediction_Classes = predict(Svm.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(Svm.model_Prediction_Classes)
confusionMatrix(Svm.model_Prediction_Classes,VALIDATION$CLASS) 
#Kappa with relief  0.3944       Accuracy : 0.9654

Svm.model_Prediction_Classes_prob = predict(Svm.model, newdata = VALIDATION, type="prob")
multiclass.roc(VALIDATION$CLASS, Svm.model_Prediction_Classes_prob)$auc 
#Multi-class area under the curve: 0.8733  

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''5'''''''''''''''''''''''''''''''''''''''''''''''''''
set.seed(3291)

gbmGrid = expand.grid(interaction.depth = c(4,6,8), 
                      n.trees = seq(1, 500, by = 50), 
                      shrinkage = c(0.1,0.01, 0.001, 0.0001),
                      n.minobsinnode = 20)

Gbm.model = train(Class ~ .,
                  data = TrainSet,
                  method='gbm',
                  trControl = ctrl,
                  verbose = FALSE,
                  tuneGrid = gbmGrid,
                  metric = 'Kappa')

ggplot(Gbm.model) + theme_bw()
#The final values used for the model were n.trees = 301, interaction.depth = 8, shrinkage= 0.1 and n.minobsinnode = 20.
print(Gbm.model) 
head(Gbm.model$pred)
#Confusion matrix
confusionMatrix(data = Gbm.model$pred$pred, 
                reference = Gbm.model$pred$obs)
#Kappa with relief   0.9885        Accuracy : 0.9914  

set.seed(3291)
Gbm.model_Prediction_Classes = predict(Gbm.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(Gbm.model_Prediction_Classes)
confusionMatrix(Gbm.model_Prediction_Classes,VALIDATION$CLASS)
#Kappa with relief 0.5815       Accuracy : 0.9785

Gbm.model_Prediction_Classes_prob = predict(Gbm.model, newdata = VALIDATION, type="prob")
multiclass.roc(VALIDATION$CLASS, Gbm.model_Prediction_Classes_prob)$auc 
#AUC = Multi-class area under the curve:  0.9407  

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''6''''''''''''''''''''''''''''''''''''''''''''''''''''''''

set.seed(3291)

nnetGrid = expand.grid(.size = c(1:10,50,70,100)
                       ,.decay = c(0.01,0.1,1.2))
maxSize = max(nnetGrid$.size)
numWts = 1*(maxSize*(length(df_TRAIN)+1 + maxSize +1))

nnet.model = train(Class ~.,
                   data = TrainSet,
                   method='nnet',
                   metric = "Kappa",
                   prePKappaess = "spatialSign",
                   tuneGrid = nnetGrid,
                   MaxNWts = numWts,
                   maxit = 500,
                   trControl=ctrl)

ggplot(nnet.model) + theme_bw()
head(nnet.model,5)
head(nnet.model$pred,5)
#Confusion matrix
confusionMatrix(data = nnet.model$pred$pred,    
                reference = nnet.model$pred$obs)
#Kappa  0.9643  Accuracy : 0.9732 

set.seed(3291)

nnet.model_Prediction_Classes = predict(nnet.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(nnet.model_Prediction_Classes)
confusionMatrix(nnet.model_Prediction_Classes,VALIDATION$CLASS) 
#Kappa 0.446      Accuracy : 0.9678

nnet.model_Prediction_Probs = predict(nnet.model, newdata = VALIDATION, type = "prob")
multiclass.roc(VALIDATION$CLASS, nnet.model_Prediction_Probs)$auc 
#AUC = Multi-class area under the curve: 0.8387

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''7'''''''''''''''''''''''''''''''''''''''''''''''''''''
set.seed(3291)

rf_grid = expand.grid(.mtry=c(5,10,25,75,100))
rf.model = train(Class ~.,
                 data = TrainSet,
                 method='rf',
                 metric='Kappa',
                 ntree=600,            # number of trees in the Random Forest
                 nodesize=100,       # minimum node size set small enough to allow for complex trees,
                 # but not so small as to require too large B to eliminate high variance
                 importance=TRUE,    # evaluate importance of predictors
                 keep.inbag=TRUE,
                 trControl=ctrl,
                 tuneGrid=rf_grid,
                 allowParallel=TRUE)

print(rf.model)
head(rf.model$pred)
#Confusion matrix
confusionMatrix(data = rf.model$pred$pred,    
                reference = rf.model$pred$obs)
# Kappa = 0.9899 Accuracy : 0.9924     Fitting mtry = 5 on full training set

set.seed(3291)

rf.model_Prediction_Classes = predict(rf.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(rf.model_Prediction_Classes)
confusionMatrix(rf.model_Prediction_Classes,VALIDATION$CLASS) 
#Kappa : 0.6075  Accuracy : 0.9749

rf.model_Prediction_Probs = predict(rf.model, newdata = VALIDATION, type = "prob")
multiclass.roc(VALIDATION$CLASS, rf.model_Prediction_Probs)$auc 
#AUC = Multi-class area under the curve: 0.9932

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''8'''''''''''''''''''''''''''''''''''''''''''''''''''
set.seed(3291)

knn_grid = expand.grid(k = seq(1, 111, by = 11))
knn.model = train(Class ~.,
                  data = TrainSet,
                  method='knn',
                  metric='Kappa',
                  trControl=ctrl,
                  tuneGrid = knn_grid)

ggplot(knn.model) + theme_bw()
head(knn.model,5)
head(knn.model$pred,5)
knn.model$results
#Confusion matrix
confusionMatrix(data = knn.model$pred$pred,
                reference = knn.model$pred$obs) 
#Kappa = 0.937     Accuracy : 0.9527      Fitting k = 1 on full training set

set.seed(3291)

knn.model_Prediction_Classes = predict(knn.model, newdata = VALIDATION)
table (VALIDATION$CLASS)
table(knn.model_Prediction_Classes)
confusionMatrix(knn.model_Prediction_Classes,VALIDATION$CLASS) 
# Kappa = 0.4621    Accuracy : 0.9785 

knn.model_Prediction_Probs = predict(knn.model, newdata = VALIDATION, type = "prob")
multiclass.roc(VALIDATION$CLASS, knn.model_Prediction_Probs)$auc 
#AUC = Multi-class area under the curve:0.5733

# collect resamples
results = resamples(list(NB = naiveBayes.model,
                         Rpart = rpart.model,
                         C5.0 = C5.0.model,
                         SVM = Svm.model,  
                         GBM = Gbm.model,
                         NNET = nnet.model,
                         RF = rf.model,
                         KNN = knn.model))
results$timings
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)

#...............................TESTING MODELS ON TEST SET..................................

set.seed(3291)
Test_rpart.model_Prediction_Classes = predict(rpart.model, newdata = TestSet)
table (TestSet$CLASS)
table(Test_rpart.model_Prediction_Classes)
confusionMatrix(Test_rpart.model_Prediction_Classes,TestSet$CLASS) 
#Kappa with relief  0.5902           Accuracy : 0.9742 

Test_rpart.model_Prediction_Classes_prob = predict(rpart.model, newdata = TestSet, type="prob")
multiclass.roc(TestSet$CLASS, Test_rpart.model_Prediction_Classes_prob)$auc 
#Multi-class area under the curve:  0.8826  

# -----------
set.seed(3291)
Test_C5.0.model_Prediction_Classes = predict(C5.0.model, newdata = TestSet)
table (TestSet$CLASS)
table(Test_C5.0.model_Prediction_Classes)
confusionMatrix(Test_C5.0.model_Prediction_Classes,TestSet$CLASS) 
#Kappa with relief  0.5506           Accuracy : 0.9764 

Test_C5.0.model_Prediction_Classes_prob = predict(C5.0.model, newdata = TestSet, type="prob")
multiclass.roc(TestSet$CLASS, Test_C5.0.model_Prediction_Classes_prob)$auc 
#Multi-class area under the curve:  0.8903   

#----------
set.seed(3291)
Test_Gbm.model_Prediction_Classes = predict(Gbm.model, newdata = TestSet)
table (TestSet$CLASS)
table(Test_Gbm.model_Prediction_Classes)
confusionMatrix(Test_Gbm.model_Prediction_Classes,TestSet$CLASS) #Kappa with relief    
#Kappa : 0.5497   Accuracy : 0.9796

Test_Gbm.model_Prediction_Classes_prob = predict(Gbm.model, newdata = TestSet, type="prob")
multiclass.roc(TestSet$CLASS, Test_Gbm.model_Prediction_Classes_prob)$auc 
#AUC = Multi-class area under the curve: 0.9546   

#----------
set.seed(3291)
Test_rf.model_Prediction_Classes = predict(rf.model, newdata = TestSet)
table (TestSet$CLASS)
table(Test_rf.model_Prediction_Classes)
confusionMatrix(Test_rf.model_Prediction_Classes,TestSet$CLASS) 
#Kappa : 0.5759    Accuracy : 0.9742 

Test_rf.model_Prediction_Probs = predict(rf.model, newdata = TestSet, type = "prob")
multiclass.roc(TestSet$CLASS, Test_rf.model_Prediction_Probs)$auc 
#AUC = Multi-class area under the curve: 0.9931 

#----------
set.seed(3291)
Test_knn.model_Prediction_Classes = predict(knn.model, newdata = TestSet)
table (TestSet$CLASS)
table(Test_knn.model_Prediction_Classes)
confusionMatrix(Test_knn.model_Prediction_Classes,TestSet$CLASS) 
# Kappa = 0.66    Accuracy : 0.9839  

Test_knn.model_Prediction_Probs = predict(knn.model, newdata = TestSet, type = "prob")
multiclass.roc(TestSet$CLASS, Test_knn.model_Prediction_Probs)$auc 
#AUC = Multi-class area under the curve:0.8529   

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx