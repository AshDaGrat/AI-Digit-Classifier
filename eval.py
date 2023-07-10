import numpy as np
import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms



model = torch.load('digit_classifier_v5.pt')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
valset = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


correct_count, all_count = 0, 0
for images,labels in valloader:
    for i in range(len(labels)):
      img = images[i].view(1, 784)
      with torch.no_grad():
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