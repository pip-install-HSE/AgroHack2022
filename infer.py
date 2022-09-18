import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from matplotlib import pyplot as plt

y = time.time()
model = torch.load('./model_2.pt', map_location=torch.device('cpu'))
model.cpu()
model.eval()

image_path = 'frame_23.png'

trans = transforms.Compose([transforms.ToTensor()])

demo = Image.open(image_path)
height, width = demo.height, demo.width
print(height, width)
height += 32 - (demo.height % 32) if (demo.height % 32) > 0 else 0
width += 32 - (demo.width % 32) if (demo.width % 32) > 0 else 0
print(height, width)
background_color = (0, 0, 0)
result = Image.new(demo.mode, (width, height), background_color)
result.paste(demo, (0, 0))
demo_img = trans(result)
demo_array = np.moveaxis(demo_img.numpy() * 255, 0, -1)
print(Image.fromarray(demo_array.astype(np.uint8)))
arr = np.random.random((64, 64, 3))
t = transforms.ToTensor()(arr)
t = t[None, :].cpu().float()

pred = model.predict(t)
print(time.time() - y)

x = time.time()
demo_img = demo_img[None, :].cpu()
p = model.predict(demo_img)
print(time.time() - x)
