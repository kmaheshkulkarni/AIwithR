# Install packages
library(keras)
# install_keras()

# Read data
data <- read.csv(file.choose(), header = T)
str(data)

# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Normalize
data[, 1:21] <- normalize(data[,1:21])
data[,22] <- as.numeric(data[,22]) -1
summary(data)

# Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:21]
test <- data[ind==2, 1:21]
trainingtarget <- data[ind==1, 22]
testtarget <- data[ind==2, 22]

# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
print(testLabels)

# Create sequential model
model <- keras_model_sequential()
model %>% layer_dense(units=8, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)

# Compile     

##################### categorical_crossentropy for one variable and binary_crossentropy loss function used for two category ##################

model %>%
         compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = 'accuracy')

# Adam optimaization algorithm is a popular algorithm in deep learning field 

# Fit model

#### Multilayer Perceptron Neural N/W for Multi-class Softmax Classification ####

history <- model %>%
         fit(training,
             trainLabels,
             epoch = 200, # epoch means executions no. of time
             batch_size = 32,
             validation_split = 0.2) # 20 % of Data
plot(history)

# Evaluate model with test data
model1 <- model %>% evaluate(test, testLabels)

# Prediction & confusion matrix - test data
prob <- model %>% predict_proba(test)

pred <- model %>% predict_classes(test)
table1 <- table(Predicted = pred, Actual = testtarget)

cbind(prob, pred, testtarget)

# Fine-tune model

table1
model1

model <- keras_model_sequential()
model %>% layer_dense(units=50, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)

model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

history <- model %>%
  fit(training,
      trainLabels,
      epoch = 200, # epoch means executions no. of time
      batch_size = 32,
      validation_split = 0.2) # 20 % of Data

model2 <- model %>% evaluate(test, testLabels)
pred2 <- model %>% predict_classes(test)
table2 <- table(Predicted = pred2, Actual = testtarget)


model %>% layer_dense(units=50, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

history <- model %>%
  fit(training,
      trainLabels,
      epoch = 200, # epoch means executions no. of time
      batch_size = 32,
      validation_split = 0.2) # 20 % of Data

model3 <- model %>% evaluate(test, testLabels)
pred3 <- model %>% predict_classes(test)
table3 <- table(Predicted = pred3, Actual = testtarget)
