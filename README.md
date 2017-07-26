# Loan-Prediction
About Company Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Problem Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.
rm(list=ls())
train_Loan <- read_csv("~/Analytics/loanpredictor/train.csv")
test_Loan <- read_csv("~/Analytics/loanpredictor/test.csv")

require(plyr)
require(dplyr)
require(caret)
require(rpart)
require(ggplot2)
require(readr)
require(xgboost)
require(readr)
require(rpart)
#Look and feel of data
attach(train_Loan)
glimpse(train_Loan)

#train<-train_Loan[,-13]

test_Loan$Loan_Status=1
glimpse(test_Loan)

train_Loan$Loan_Status <- ifelse(train_Loan$Loan_Status=='Y',1,0)

combine_data=rbind(train_Loan,test_Loan)
summary(combine_data)

n_train=1:nrow(train_Loan)
#cat_chr<-names(train_Loan)[which(sapply(train_Loan,is.character))]
#cat_fac<- as.factor(cat_chr)
#train1<-c(cat_fac,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History )

combine_data$Gender=factor(combine_data$Gender)
combine_data$Married=factor(combine_data$Married)
combine_data$Dependents=factor(combine_data$Dependents)
combine_data$Education=factor(combine_data$Education)
combine_data$Self_Employed=factor(combine_data$Self_Employed)
combine_data$Property_Area=factor(combine_data$Property_Area)
combine_data$Credit_History=factor(combine_data$Credit_History)

combine_data1=combine_data[,-c(1)]

pairs(combine_data1)

#missing values
#complete.cases(combine_data1)
colSums(sapply(combine_data1, is.na))
## get the rows of missing values
new_DF <- subset(combine_data, is.na(combine_data$Married))
new_DF
#we do have missing values 

sum(complete.cases(combine_data1))
dim(combine_data1)
summary(combine_data1)

#Plot missing values
missmap(combine_data1[,1:11],
        main = "Missing values in Loan Dataset",
        col=c('grey', 'steelblue'),
        y.cex=0.5, x.cex=0.8)
sort(sapply(combine_data1, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#combine_data1$Married[which(is.na(combine_data1$Married))]= "Yes"

#combine_data1 <- dummyVars(~.,data=combine_data1,fullRank=T)

#levels(combine_data1)
                

summary(as.factor(combine_data1$Self_Employed))


table(combine_data1$Gender,combine_data1$Self_Employed)

col.pred<- c("Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome",
             "LoanAmount","Loan_Amount_Term","Credit_History", "Property_Area")
glimpse(combine_data1)
sort(sapply(combine_data1, function(x) { sum(is.na(x)) }), decreasing=TRUE)

# Predict Credit history
crdthis.rpart <- rpart(Credit_History ~ .,
                       data = combine_data1[!is.na(combine_data1$Credit_History),col.pred],
                       method = "class",
                       na.action=na.omit)
combine_data1$Credit_History[is.na(combine_data1$Credit_History)] <- as.character(predict(crdthis.rpart, combine_data1
                                                          [is.na(combine_data1$Credit_History),col.pred],type="class"))

# Predict Married
Married.rpart <- rpart(Married ~ .,
                      data = combine_data1[!is.na(combine_data1$Married),col.pred],
                      method = "class",
                      na.action=na.omit)
combine_data1$Married[is.na(combine_data1$Married)] <- as.character(predict(Married.rpart, combine_data1
                                                                          [is.na(combine_data1$Married),col.pred], 
                                                                          type="class"))
# Predict Gender
Gender.rpart <- rpart(Gender ~ .,
                    data = combine_data1[!is.na(combine_data1$Gender),col.pred],
                    method = "class",
                    na.action=na.omit)
combine_data1$Gender[is.na(combine_data1$Gender)] <- as.character(predict(Gender.rpart, combine_data1
                                                                          [is.na(combine_data1$Gender),col.pred], 
                                                                          type="class"))
# Predict Self_emp
SE.rpart <- rpart(Self_Employed ~ .,
                      data = combine_data1[!is.na(combine_data1$Self_Employed),col.pred],
                      method = "class",
                      na.action=na.omit)
combine_data1$Self_Employed[is.na(combine_data1$Self_Employed)] <- as.character(predict(SE.rpart, combine_data1
                                                                          [is.na(combine_data1$Self_Employed),col.pred], 
                                                                          type="class"))
# Predict Dependents
Depndt.rpart <- rpart(Dependents ~ .,
                  data = combine_data1[!is.na(combine_data1$Dependents),col.pred],
                  method = "class",
                  na.action=na.omit)
combine_data1$Dependents[is.na(combine_data1$Dependents)] <- as.character(predict(Depndt.rpart, combine_data1
                                                                                        [is.na(combine_data1$Dependents),col.pred],                                                                                     type="class"))
# Predict loan amt
Loanamt.rpart <- rpart(LoanAmount ~ .,
                       data = combine_data1[!is.na(combine_data1$LoanAmount),col.pred],
                       method = "anova",
                       na.action=na.omit)
combine_data1$LoanAmount[is.na(combine_data1$LoanAmount)] <- round(predict(Loanamt.rpart, combine_data1[is.na(combine_data1$LoanAmount),col.pred]))

#loan term
Loantrm.rpart <- rpart(Loan_Amount_Term ~ .,
                       data = combine_data1[!is.na(combine_data1$Loan_Amount_Term),col.pred],
                       method = "anova",
                       na.action=na.omit)
combine_data1$Loan_Amount_Term[is.na(combine_data1$Loan_Amount_Term)] <- round(predict(Loantrm.rpart, combine_data1[is.na(combine_data1$Loan_Amount_Term),col.pred]))


# Run cv
dumy <- dummyVars(" ~ .", data = combine_data1)
dtraintrsf=data.frame(predict(dumy,newdata = combine_data1))
dim(dtraintrsf)

#get the train and test data back
dtrain=dtraintrsf[1:614,]
#dtrain=cbind(train_Loan[,c(13)],dtrain)
#dtrain=dtrain[,c(1,3,4,5,6,7,8,9,10,11,12,13,2)]
dtest=dtraintrsf[615:981,]
#dtest=cbind(test_Loan[,1],dtest)


# what we're trying to predict Loan _status
outcomeName <- c('Loan_Status')
# list of features
predictors <- names(dtraintrsf)[!names(dtraintrsf) %in% outcomeName]

#predict

smallestError <- 20
for (depth in seq(1,10,1)) {
  for (rounds in seq(1,20,1)) {
    
    # train
    bst <- xgboost(data = as.matrix(dtrain[,predictors]),
                   label = dtrain[,outcomeName],
                   max.depth=depth, nround=rounds,
                   objective = "reg:linear", verbose=0)
    gc()
    
    # predict
    predictions <- predict(bst, as.matrix(dtest[,predictors]), outputmargin=TRUE)
    err <- rmse(as.numeric(dtest[,outcomeName]), as.numeric(predictions))
    
    if (err < smallestError) {
      smallestError = err
      print(paste(depth,rounds,err))
    }     
  }
}  

#pred <- predict(bst, as.matrix(dtest))
dtest$Loan_Status=ifelse(predictions >= .5,'Y','N')

submission=as.matrix(dtest[,c("Loan_Status")])
submission=cbind(test_Loan[,c(1)],submission)

write.csv(submission,file="sample_submission.csv",row.names=FALSE)

