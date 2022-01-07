import argparse
import torch
import torchvision
from model import Model
import torchvision.transforms as transforms
from dataset import train_val_dataset

parser = argparse.ArgumentParser(description='test model params')
parser.add_argument('--weight_file', type=str, default='persian-mnist.pth', help='model weight file path')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
model.load_state_dict(torch.load(args.weight_file, map_location=torch.device(device)))
model.train(False)
model.eval()

def accuracyCalc(preds, lables):
  _, preds_max = torch.max(preds,1)
  acc = torch.sum(preds_max == lables)/len(preds)
  return acc

transform = transforms.Compose([
     transforms.Resize((70,70)),
     transforms.RandomRotation(10),
     transforms.ToTensor(),
     torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
dataset = torchvision.datasets.ImageFolder("MNIST_persian", transform=transform)
test_dataset = train_val_dataset(dataset, train=False)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
trainAccuracy = 0
for images, labels in test_data_loader:
  images = images.to(device)
  labels = labels.to(device)
  predictions = model(images)
  trainAccuracy += accuracyCalc(predictions, labels)

totalAccuracy = trainAccuracy/len(test_data_loader)
print(totalAccuracy)