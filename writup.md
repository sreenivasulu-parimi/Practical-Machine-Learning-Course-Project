Creating a prediction model
===========================

Overview
--------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement, a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.

The goal of this project is to build a machine learning algorithm to
predict activity quality (classe) from activity monitors.

#### Course Project Prediction Quiz Portion

Apply your machine learning algorithm to the 20 test cases available in
the test data above and submit your predictions in appropriate format to
the Course Project Prediction Quiz for automated grading.

Load Required Libraries
-----------------------

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(ggplot2)

Download the Data
-----------------

    train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    validation_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    # Download the data into current working directory:
    train_file <- "./data/pml-training.csv"
    validation_file  <- "./data/pml-testing.csv"

    if(!file.exists("./data")) {
          dir.create("./data")
    }

    if(!file.exists(train_file)) {
          download.file(train_url, destfile=train_file, method="curl")
    }

    if(!file.exists(validation_file)) {
          download.file(validation_url, destfile=validation_file, method="curl")
    }

Read the Data
-------------

    train <- read.csv("./data/pml-training.csv")
    validation <- read.csv("./data/pml-testing.csv")

Data Exploration and Preprocessing
----------------------------------

    dim(train)

    ## [1] 19622   160

    dim(validation)

    ## [1]  20 160

    str(train)

    ## 'data.frame':    19622 obs. of  160 variables:
    ##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
    ##   [list output truncated]

The training data has 19622 observations and 160 variables whereas has
20 observations and 160 variables.

From the above, we can observe that first few (7) variables called **x**
contains row numbers; **user\_name** contains user name who did the
test; **raw\_timestamp\_part\_1**, **raw\_timestamp\_part\_2**, and
**cvtd\_timestamp** are contain timestamps; **new\_window** and
**num\_window** are not related to sensor data. So, we can ignore these
variables since they don't add any value to the prediction outcome.

    train_NAs <- colMeans(is.na(train))
    table(train_NAs)

    ## train_NAs
    ##                 0 0.979308938946081 
    ##                93                67

    validation_NAs <- colMeans(is.na(validation))
    table(validation_NAs)

    ## validation_NAs
    ##   0   1 
    ##  60 100

From the above, we can notice that there are 67 variables in which
almost all values (97.93%) are missing. Therefore, we can remove these
variable as well since they don't add any additionval value to the
outcome class.

### Feature selection

Now we can tranform the data to only include the variables we will need
to build our model. We will remove variables with near zero variance,
variables with mostly missing data, and variables that are obviously not
useful as predictors.

**Remove the first 7 variables**:

    # remove variables that don't make intuitive sense for prediction
    train_tidy <- train[,-c(1:7)]
    validation_tidy <- validation[,-c(1:7)]
    dim(train_tidy)

    ## [1] 19622   153

    dim(validation_tidy)

    ## [1]  20 153

**Remove the NearZeroVariance variables**:

    # remove variables with nearly zero variance
    nzv <- nearZeroVar(train_tidy, saveMetrics=TRUE)
    train_tidy <- train_tidy[,nzv$nzv==FALSE]

    nzv<- nearZeroVar(validation_tidy,saveMetrics=TRUE)
    validation_tidy <- validation_tidy[,nzv$nzv==FALSE]

    dim(train_tidy)

    ## [1] 19622    94

    dim(validation_tidy)

    ## [1] 20 53

**Remove the variables those are all most NAs**:

    # remove variables that are almost NA
    # Here we get the indexes of the columns having at least 90% of NA or blank values on the training dataset
    train_NAs <- sapply(train_tidy, function(x) mean(is.na(x))) > 0.95
    validation_NAs <- sapply(validation_tidy, function(x) mean(is.na(x))) > 0.95
    train_tidy <- train_tidy[, train_NAs==FALSE]
    validation_tidy  <- validation_tidy[, validation_NAs==FALSE]
    validation_tidy  <- validation_tidy[, -length(validation_tidy)] #Remove problem_id

    dim(train_tidy)

    ## [1] 19622    53

    dim(validation_tidy)

    ## [1] 20 52

Build Model
-----------

In order to pick best fit model on the given data, we perform 3
different models **Classification Tree**, **Random Forest** and
**gradient boosting** in the following sections.

#### Split the Data to training and testing

Partioning the cleaned training set into train and test.

    set.seed(45)
    inTrain <- createDataPartition(train_tidy$classe, p=0.70, list=FALSE)
    train_data <- train_tidy[inTrain,]
    test_data <- train_tidy[-inTrain,]
    dim(train_data)

    ## [1] 13737    53

    dim(test_data)

    ## [1] 5885   53

#### Cross Validation

Cross validation method can be used to limit the effects of overfitting
and improve the efficiency of the models. Useally, this method uses 5 or
10 folds. But, 10 folds take more run time with no significant increase
of the accuracy. So, we can use 5 fold.

    # instruct train to use 5-fold CV to select optimal tuning parameters
    fitControl <- trainControl(method="cv", number=5)

### Classification Tree (Decision Trees)

    # Classification Tree Fit
    fit_cf<- train(classe ~ ., data=train_data, method="rpart", trControl=fitControl)

    # use model to predict classe in test_data
    cf_pred <- predict(fit_cf, newdata=test_data)

    # show confusion matrix to get estimate of out-of-sample error
    confusionMatrix(test_data$classe, cf_pred)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1521   14  134    0    5
    ##          B  478  363  298    0    0
    ##          C  468   31  527    0    0
    ##          D  412  183  369    0    0
    ##          E  150  135  295    0  502
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.495           
    ##                  95% CI : (0.4821, 0.5078)
    ##     No Information Rate : 0.5147          
    ##     P-Value [Acc > NIR] : 0.9988          
    ##                                           
    ##                   Kappa : 0.3405          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.5021  0.50000  0.32471       NA  0.99014
    ## Specificity            0.9464  0.84958  0.88292   0.8362  0.89215
    ## Pos Pred Value         0.9086  0.31870  0.51365       NA  0.46396
    ## Neg Pred Value         0.6419  0.92351  0.77444       NA  0.99896
    ## Prevalence             0.5147  0.12336  0.27579   0.0000  0.08615
    ## Detection Rate         0.2585  0.06168  0.08955   0.0000  0.08530
    ## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
    ## Balanced Accuracy      0.7243  0.67479  0.60381       NA  0.94115

    plot(fit_cf)

![](writup_files/figure-markdown_strict/unnamed-chunk-11-1.png)

### Random Forests

    # Random Forest Model Fit
    fit_rf<- train(classe ~ ., data=train_data, method="rf", trControl=fitControl)

    # use model to predict classe in in test_data
    rf_pred <- predict(fit_rf, newdata=test_data)

    # show confusion matrix to get estimate of out-of-sample error
    confusionMatrix(test_data$classe, rf_pred)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670    3    0    0    1
    ##          B    6 1131    2    0    0
    ##          C    0    8 1018    0    0
    ##          D    0    0   17  945    2
    ##          E    0    0    0    6 1076
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9924          
    ##                  95% CI : (0.9898, 0.9944)
    ##     No Information Rate : 0.2848          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9903          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9964   0.9904   0.9817   0.9937   0.9972
    ## Specificity            0.9990   0.9983   0.9983   0.9961   0.9988
    ## Pos Pred Value         0.9976   0.9930   0.9922   0.9803   0.9945
    ## Neg Pred Value         0.9986   0.9977   0.9961   0.9988   0.9994
    ## Prevalence             0.2848   0.1941   0.1762   0.1616   0.1833
    ## Detection Rate         0.2838   0.1922   0.1730   0.1606   0.1828
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9977   0.9943   0.9900   0.9949   0.9980

    plot(fit_rf,main="Accuracy of Random forest model by number of predictors")

![](writup_files/figure-markdown_strict/unnamed-chunk-12-1.png)

    plot(fit_rf$finalModel,main="Model error of Random forest model by number of trees")

![](writup_files/figure-markdown_strict/unnamed-chunk-12-2.png)

    # variable importance
    varImp(fit_rf)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                   Overall
    ## roll_belt          100.00
    ## yaw_belt            81.21
    ## magnet_dumbbell_z   73.80
    ## magnet_dumbbell_y   66.20
    ## pitch_belt          63.42
    ## pitch_forearm       62.66
    ## magnet_dumbbell_x   56.94
    ## roll_forearm        54.51
    ## accel_dumbbell_y    48.72
    ## accel_belt_z        47.73
    ## magnet_belt_z       46.69
    ## roll_dumbbell       45.88
    ## magnet_belt_y       43.42
    ## accel_dumbbell_z    40.22
    ## roll_arm            35.40
    ## accel_forearm_x     34.55
    ## gyros_belt_z        34.34
    ## yaw_dumbbell        31.42
    ## accel_dumbbell_x    30.60
    ## magnet_arm_x        30.03

### Gradient Boosting

    # Gradient Boosting Model Fit
    fit_gb <- train(classe ~ ., data=train_data, method="gbm", trControl=fitControl, verbose=FALSE)

    # use model to predict classe in in test_data
    gb_pred <- predict(fit_gb, newdata=test_data)

    # show confusion matrix to get estimate of out-of-sample error
    confusionMatrix(test_data$classe, gb_pred)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1649   17    8    0    0
    ##          B   31 1078   28    1    1
    ##          C    0   27  990    7    2
    ##          D    2    5   38  912    7
    ##          E    0   11    7   20 1044
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.964           
    ##                  95% CI : (0.9589, 0.9686)
    ##     No Information Rate : 0.2858          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9544          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9804   0.9473   0.9244   0.9702   0.9905
    ## Specificity            0.9941   0.9871   0.9925   0.9895   0.9921
    ## Pos Pred Value         0.9851   0.9464   0.9649   0.9461   0.9649
    ## Neg Pred Value         0.9922   0.9874   0.9833   0.9943   0.9979
    ## Prevalence             0.2858   0.1934   0.1820   0.1597   0.1791
    ## Detection Rate         0.2802   0.1832   0.1682   0.1550   0.1774
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9872   0.9672   0.9584   0.9798   0.9913

    plot(fit_gb)

![](writup_files/figure-markdown_strict/unnamed-chunk-13-1.png)

### Best Model

From the above 3 models, we can notice that Random Forest gives best
accuracy 99.24%. Acutally, the **train\_data** contains only 70% of
original training data. In order to produce the most accurate
predictions, we can train the best model i.e random forest on the full
training data **train\_tidy data**.

    # Random Forest Model Fit on full training data
    fit_best <- train(classe ~ ., data=train_tidy, method="rf", trControl=fitControl)

Submission
----------

Now we can use the best fitted model **fit\_best** to prict the
**classe** on the given actual test data.

    validation_pred <- predict(fit_best, newdata=validation_tidy)

The following function generates a file with predictions to submit as
answers for the assignment.

    pml_write_files = function(x) {
          n = length(x)
          for(i in 1:n) {
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
          }
    }
    pml_write_files(validation_pred)
    validation_pred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
