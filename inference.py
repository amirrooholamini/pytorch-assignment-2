import argparse

import time
start = time.time()
import torch
import torchvision
import cv2 as cv
import numpy as np
from model import Model
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='inference model params')
parser.add_argument('--image', type=str, default='images/six.png', help='image path')
args = parser.parse_args()

inference_transform = transforms.Compose([
     transforms.ToTensor(),
     torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
model.load_state_dict(torch.load('persian-mnist.pth', map_location ='cpu'))

model.train(False)
model.eval()

img = cv.imread(args.image)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img,(70,70))
tensor = inference_transform(img).unsqueeze(0).to(device)
prediction = model(tensor).cpu().detach().numpy()
print(np.argmax(prediction, axis=1))
end = time.time()
print(f'time: {end-start} seconds')