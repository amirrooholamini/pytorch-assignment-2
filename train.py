import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from model import Model
from dataset import train_val_dataset

def accuracyCacculator(preds, lables):
  _, preds_max = torch.max(preds,1)
  acc = torch.sum(preds_max == lables , dtype=torch.float64)/len(preds)
  return acc

parser = argparse.ArgumentParser(description='train model params')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size in each iteration')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()


def accuracyCalc(preds, lables):
  _, preds_max = torch.max(preds,1)
  acc = torch.sum(preds_max == lables , dtype=torch.float64)/len(preds)
  return acc

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)

transform = transforms.Compose([
     transforms.Resize((70,70)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = torchvision.datasets.ImageFolder("MNIST_persian", transform = transform)
train_dataset = train_val_dataset(dataset)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossFunction = torch.nn.CrossEntropyLoss()


for epoch in range(epochs):
  trainLoss = 0
  trainAccuracy = 0
  for images, labels in train_data_loader:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    predictions = model(images)
    loss = lossFunction(predictions,labels)
    loss.backward()
    optimizer.step()

    trainLoss += loss
    trainAccuracy += accuracyCalc(predictions, labels)

  totalLoss = trainLoss/len(train_data_loader)
  totalAccuracy = trainAccuracy/len(train_data_loader)
  print(f"Epoch: {epoch +1}, Loss: {totalLoss}, Accuracy: {totalAccuracy}")

torch.save(model.state_dict(), "persian-mnist.pth")
