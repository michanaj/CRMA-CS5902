1. Installation of Keras with tensorflow at the backend.
The steps to install Keras in RStudio is very simple. Just follow the below steps and you would be good to make your first Neural Network Model in R.

install.packages("devtools")

devtools::install_github("rstudio/keras")

The above step will load the keras library from the GitHub repository. Now it is time to load keras into R and install tensorflow.

library(keras)

By default RStudio loads the CPU version of tensorflow. Use the below command to download the CPU version of tensorflow.

install_tensorflow()

To install the tensorflow version with GPU support for a single user/desktop system, use the below command.

install_tensorflow(gpu=TRUE)

For multi-user installation, refer this installation guide.

Now that we have keras and tensorflow installed inside RStudio, let us start and build our first neural network in R to solve the MNIST dataset.

 

2. Different types of models that can be built in R using keras
Below is the list of models that can be built in R using Keras.

Multi-Layer Perceptrons
Convoluted Neural Networks
Recurrent Neural Networks
Skip-Gram Models
Use pre-trained models like VGG16, RESNET etc.
Fine-tune the pre-trained models.
Let us start with building a very simple MLP model using just a single hidden layer to try and classify handwritten digits.

 

3. Classifying MNIST handwritten digits using an MLP in R
#loading keras library
library(keras)

#loading the keras inbuilt mnist dataset
data<-dataset_mnist()

#separating train and test file
train_x<-data$train$x
train_y<-data$train$y
test_x<-data$test$x
test_y<-data$test$y

rm(data)

# converting a 2D array into a 1D array for feeding into the MLP and normalising the matrix
train_x <- array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x <- array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255

#converting the target variable to once hot encoded vectors using keras inbuilt function
train_y<-to_categorical(train_y,10)
test_y<-to_categorical(test_y,10)

#defining a keras sequential model
model <- keras_model_sequential()

#defining the model with 1 input layer[784 neurons], 1 hidden layer[784 neurons] with dropout rate 0.4 and 1 output layer[10 neurons]
#i.e number of digits from 0 to 9

model %>% 
layer_dense(units = 784, input_shape = 784) %>% 
layer_dropout(rate=0.4)%>%
layer_activation(activation = 'relu') %>% 
layer_dense(units = 10) %>% 
layer_activation(activation = 'softmax')

#compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
loss = 'categorical_crossentropy',
optimizer = 'adam',
metrics = c('accuracy')
)

#fitting the model on the training dataset
model %>% fit(train_x, train_y, epochs = 100, batch_size = 128)

#Evaluating model on the cross validation dataset
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)

 

The above code had a training accuracy of 99.14 and validation accuracy of 96.89. The code ran on my i5 processor and took around 13.5s for a single epoch whereas, on a TITANx GPU, the validation accuracy was 98.44 with an average epoch taking 2s.

 
