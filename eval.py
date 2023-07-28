import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from CNN_training import CNN

model = CNN()
model = torch.load('digit_classifier_v6.pt')
model.train(mode = False)

#Defining a transformation pipeline using transforms.Compose() to convert images to tensors and normalize them.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

#Loading the MNIST validation dataset (valset) and loading it using valloader
valset = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

correct_count, all_count = 0, 0
for images,labels in valloader:
    for i in range(len(labels)):
        model.eval() 

        #Reshaping the image tensor to have a shape of (1, 784) using the .view() method.
        img = images[i].view(1, 784) 

        #disable gradient calculation during inference.
        with torch.no_grad(): 
            #Passing the reshaped image tensor to the pre-trained model to obtain the log probabilities (logps) of each digit class.
            logps = model(img)

        
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1


print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count)*100)