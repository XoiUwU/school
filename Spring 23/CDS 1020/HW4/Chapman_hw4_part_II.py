### Import libraries. ###
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

### Load data. ###

writer = SummaryWriter('runs/mnist_experiment_1')

#Resource: https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
#Load the training data from the MNIST dataset.

training_data = datasets.MNIST(
  root="data",
  train="True",
  download="True",
  transform=ToTensor()
)

#Load the testing data from the MNIST dataset.

test_data = datasets.MNIST(
  root="data",
  train="False",
  download="True",
  transform=ToTensor()
)

### Wrap the data as a dataloader object. ###
#Load the training data as a dataloader object. 

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
print(type(train_dataloader))

#Load the testing data as a dataloader object. 

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
print(type(test_dataloader))

###Part II.A. ###

### Set up model. ###
#Specify the dimensions of the input layer. 
#Specify the dimensions of the output layer. 
class Net(nn.Module):
  def __init__ (self, input_dim, output_dim):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_dim, 512) 
    self.fc2 = nn.Linear(512, output_dim)
  def forward(self, x):
    x = x.view(-1, 28*28)
    x = self.fc1(x)
    x = torch.relu(x)
    x = self.fc2(x)
    return x

input_dim = 28*28
output_dim = 10
model = Net(input_dim, output_dim)

#Define model.
class Net(nn.Module):
  def __init__(self,input_dim=input_dim,output_dim=output_dim):
    super(Net, self).__init__()

    #Flatten layer
    #Define a flattening layer called flatten.

    self.flatten = nn.Flatten()

    #Linear function
    #Define a linear layer.
    #Hint: As this is a logistic regression, there are no 
    #hidden layers. The two arguments will reflect current dimensions.
    self.linear1=nn.Linear(in_features= 28*28, out_features= 512) ### Placeholders contained within line. ###
    
    #Initialization of weights.
    nn.init.xavier_normal_(self.linear1.weight)

    #Non-linear function.
    #Define the tanh function.
    self.tanh = nn.Tanh()

    
   

  #Define the function for the forward pass through all the layers.
  #Hint: this will use the previous attributes in the order they are in.
  #You can pass x as the input and output for each layer as this
  #still represents the passage of data.
  def forward(self, x):
      x = x.view(-1, 28*28)
      x = self.linear1(x)
      x = torch.relu(x)
      return x
  

### Instantiate network. ###
#Instantiate an object from the class Net. The name to use is referenced downstream.

net = Net()

### Backpropagation. ###
#Define loss function.

loss_fn = nn.CrossEntropyLoss()

#Define optimization algorithm. 
#Set the learning rate to 0.001.

optimizer  = torch.optim.Adam(net.parameters(), lr=0.001)

### Train model. ###
def train_loop(train_dataloader,net,loss_fn,optimizer):
  num_batches = len(train_dataloader)
  train_loss = 0
  for batch, (X,y) in enumerate(train_dataloader):
    #Prediction and loss calculation.
    pred = net(X)

    #Call loss function, loss_fn, passing the required arguments. Store in object called loss
    loss = loss_fn(pred, y)

    train_loss = train_loss + loss

    #Backpropagation.
    optimizer.zero_grad() #zeros the gradient buffers due to gradient accumulation. 
    #zero_grad() empties the weights for the next iteration.
    loss.backward()
    optimizer.step() #performs the update of parameters

    if batch % 100 == 0: #print loss values
      loss, current = loss.item(), batch*len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{num_batches:>5d}]")

  return train_loss

### Test model. ###
def test_loop(test_dataloader, net, loss_fn):
  num_batches = len(test_dataloader)
  test_loss, correct, size = 0, 0, 0

  with torch.no_grad(): 
    for X, y in test_dataloader: 
      pred = net(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      size += len(y)

  test_loss /= num_batches
  accuracy = correct / size

  #Calculate the ratio of correct predictions to number of samples.
  #Store the quotient in an object called correct.

  predicted_labels = torch.argmax(pred)  
  num_correct = (predicted_labels == y).sum().item()
  correct = num_correct / len(y)

  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")

  return test_loss, accuracy
  
myNetwork = Net()
#Define number of epochs.
num_epochs = 25
train_lossesA = []
test_lossesA = []
accuraciesA = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n----------")
    train_loss = train_loop(train_dataloader,myNetwork,loss_fn,optimizer)
    train_lossesA.append(train_loss)
    test_loss, accuracy = test_loop(test_dataloader,myNetwork,loss_fn)
    writer.add_scalar("Loss/train", test_loss, epoch)
    test_lossesA.append(test_loss)
    accuraciesA.append(accuracy)

#Plotting the training and test losses.
train_losses_detached = [loss.detach().item() for loss in train_lossesA]
plt.plot(train_losses_detached,label='Training Loss')
plt.plot(test_lossesA,label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')  
plt.legend()
plt.show()

#Plotting the test accuracies.
plt.plot(accuraciesA)
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')  
plt.show()

writer.flush()
writer.close()
print("Done!")

################################################################################

###Part II.B. ###

#Resource: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

### Set up model. ###
#Specify the dimensions of the input layer. 
   
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512,512),
      nn.ReLU(),
      nn.Linear(512,10),
    )

#Specify the dimensions of the output layer. 

   ### Placeholder. ###

#Define model.
#The network will have multiple hidden layers.
class Net(nn.Module):
    def __init__(self, input_dim = input_dim, output_dim = output_dim):
        super(Net, self).__init__()

        #Conv2d layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)

        #Relu layer 1
        #Define a Relu layer called relu1.

        self.relu1 = nn.ReLU()
        
        #BatchNorm2d 
        self.batchnorm1 = nn.BatchNorm2d(3)

        #Conv2d layer
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)

        #Relu layer 2
        #Define a Relu layer called relu2.

        self.relu2 = nn.ReLU()
        
        #BatchNorm2d 
        self.batchnorm2 = nn.BatchNorm2d(3)

        #Flatten layer
        #Define a flattening layer called flatten.

        self.flatten = nn.Flatten()
        
        #Linear function
        #Pass a value to the parameter out_features for the linear layer.
        #Hint: this value relates to the final output dimensions.
        self.linear = nn.Linear(in_features = input_dim*3, out_features = output_dim)
        self.fc1 = nn.Linear(in_features=3*28*28, out_features=output_dim)

        #Non-linear function
        #Define a non-linear function for multi-class classification. Hint: not tanh.

        self.softmax = nn.Softmax(dim=1)

    #Define the function for the forward pass through all the layers.
    #Hint: this will use the previous attributes in the order they are in.
    #You can pass x as the input and output for each layer as this
    #still represents the passage of data.
    def forward(self, x):    
      x = self.conv1(x)
      x = self.relu1(x)
      x = self.batchnorm1(x)
      x = self.conv2(x)
      x = self.relu2(x)
      x = self.batchnorm2(x)
      x = self.flatten(x)

      #Linear function

      x = self.fc1(x)

      #Non-linear function. 

      x = self.softmax(x)

      return x

### Instantiate network. ###
#Instantiate an object from the class Net, giving it the name my_conv_net.

my_conv_net = Net()

### Method for Backpropagation. ###
#Define loss function.

criterion = nn.CrossEntropyLoss()

#Define optimization algorithm. 
#The learning rate may be set to 0.001.

optimizer = torch.optim.Adam(my_conv_net.parameters(), lr=0.001)

### Train model. ###
def train_loop(train_dataloader,net,loss_fn,optimizer): 
  num_batches = len(train_dataloader)
  train_loss=0
  for batch, (X, y) in enumerate(train_dataloader): 
    #Prediction and loss calculation.
    pred = net(X)

    #Call loss function, loss_fn, passing the required arguments. Store in 
    #object called loss

    loss = loss_fn(pred, y)

    train_loss = train_loss+loss

    #Backpropagation 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0: 
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{num_batches:>5d}]")

  return train_loss

### Test model. ###
def test_loop(test_dataloader, net, loss_fn): 
  num_batches = len(test_dataloader)
  test_loss, correct, size = 0, 0, 0

  with torch.no_grad(): 
    for X, y in test_dataloader: 
      pred = net(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      size += len(X)
  
  test_loss /= num_batches

  #Calculate the ratio of correct predictions to number of samples.
  #Store the quotient in an object called correct.

  correct /= size
  
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
  return test_loss

#Define number of epochs.

num_epochs = 25
train_lossesB = []
test_lossesB = []
accuraciesB = []

for epoch in range(num_epochs): 
  print(f"Epoch {epoch+1}\n--------------")
  train_loss = train_loop(train_dataloader, my_conv_net, loss_fn, optimizer)
  train_lossesB.append(train_loss)
  test_loss = test_loop(test_dataloader, my_conv_net, loss_fn)
  writer.add_scalar("Loss/train", test_loss, epoch)
  test_lossesB.append(test_loss)
  accuraciesB.append(accuracy)


#Plotting the training and test losses.
train_losses_detachedB = [loss.detach().item() for loss in train_lossesB]
plt.plot(train_losses_detachedB,label='Training Loss')
plt.plot(test_lossesB,label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')  
plt.legend()
plt.show()

#Plotting the test accuracies.
plt.plot(accuraciesB)
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')  
plt.show()

writer.flush()
writer.close()
print("Done!")