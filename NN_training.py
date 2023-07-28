import torch 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time

#Defining a transformation pipeline using transforms.Compose() to convert images to tensors and normalize them.
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,)),])


#Raw MNIST Dataset
"""
trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
"""


#regular dataset
mnist_trainset = datasets.MNIST(root='./data', 
                                download=True, 
                                train=True, 
                                transform=transform)


# Define the augmentations using torchvision.transforms
augmentations = transforms.Compose([transforms.RandomAffine(degrees=5, 
                                                            translate=(2/28, 2/28), 
                                                            scale=(0.9, 1.1)),
                                    transforms.ToTensor()])  # Convert the augmented images to tensors


# Apply the augmentations to create the augmented dataset
augmented_trainset = datasets.MNIST(root='./aug_data', 
                                    train=True, 
                                    download=True, 
                                    transform=augmentations)


# Use the augmented dataset for training
trainloader = torch.utils.data.DataLoader(augmented_trainset, batch_size=100, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)


#code to show random data from the database
"""
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()
"""

#creating the neural network
input_size = 784
hidden_sizes = [256, 256, 64]
output_size = 10


model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),    #the ReLU function returns 0 if x < 0 and x otherwise
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)


logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss


print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time.time()
epochs = 50


for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 dimensional vector
        images = images.view(images.shape[0], -1)
    

        # Training pass
        optimizer.zero_grad()
        

        output = model(images)
        loss = criterion(output, labels)
        

        #This is where the model learns by backpropagating
        loss.backward()
        

        #And optimizes its weights here
        optimizer.step()
        running_loss += loss.item()

    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        print("\nTraining Time (in minutes) =",(time.time()-time0)/60)


torch.save(model, './digit_classifier_v6.pt') 


print(model)