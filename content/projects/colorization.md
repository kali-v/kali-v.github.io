---
title: Colorization Of Grayscale Images
url: projects/colorization
type: posts
desc: [
    "<b>TLDR</b>: Pytorch implementation of the Deep Koalarization paper with some tweaks.",
    "<b>code</b>: <a href=\"https://github.com/kali-v/lightflow/blob/master/extra/matmul.cc\" > here </a>",
    "<b>models</b>: <a href=\"https://drive.google.com/drive/folders/1Ey-ZRnMkdMVf5soanxvMEVzZcXTqI5Kz?usp=sharing\" > here </a>",
    ]
---

**Content Table**
- [LAB color model](#conversion-to-lab-color-model)
- [Unet baseline](#unet-baseline)
- [Deep Koalarization](#deep-koalarization)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

---


**Deep Koalarization (https://arxiv.org/pdf/1712.03400.pdf)** \
Deep Koalarization proposes a convolutional network that uses Inception-Resnet-v2 trained on ImageNet as a feature extractor. The network consists of three main parts: an encoder, fusion (with the result of Inception-Resnet-v2), and a decoder. The network output is a* and b* layers in CIE Lab color space. Deep Koalarization paper uses MSE Loss, Adam optimizer, and the training set consists of ~54000 images from ImageNet.


```python
! pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
! pip install scikit-image matplotlib validators tensorboard setuptools==59.5.0
```

# Data download

I used COCO because it's more accessible than ImageNet in og paper. I'm using test2017 + test2014 as the training set(80k images) and val2017 as the testing set.

```python
!wget http://images.cocodataset.org/zips/test2017.zip -P persistent-storage
!wget http://images.cocodataset.org/zips/test2014.zip -P persistent-storage
!wget http://images.cocodataset.org/zips/val2017.zip -P persistent-storage

```

```python
import zipfile

with zipfile.ZipFile('data/test2017.zip', 'r') as zip_ref:
    zip_ref.extractall('data/coco')

!mv data/coco/test2017/* data/coco
!rmdir data/coco/test2017

with zipfile.ZipFile('data/test2014.zip', 'r') as zip_ref:
    zip_ref.extractall('data/coco-tmp')

!mv data/coco-tmp/test2014/* data/coco
!rmdir data/coco-tmp

with zipfile.ZipFile('data/val2017.zip', 'r') as zip_ref:
    zip_ref.extractall('data/coco-val')

!mv data/coco-val/val2017/* data/coco-val
!rmdir data/coco-val/val2017
```

```python
import os
from skimage import io

# remove grayscale images
for i, image_path in enumerate(os.listdir('persistent-storage/coco')):
    im_path = f'persistent-storage/coco/{image_path}'
    image = io.imread(im_path)
    if image.shape[-1] != 3:
        os.remove(im_path)

for i, image_path in enumerate(os.listdir('persistent-storage/coco-val')):
    im_path = f'persistent-storage/coco-val/{image_path}'
    image = io.imread(im_path)
    if image.shape[-1] != 3:
        os.remove(im_path)
```



<!-- #region -->
# Conversion to LAB color model

The paper uses LAB color model. The input is luminescense component and the network is trying to predict a* and b* components. All components are further normalized to values between -1 and 1.
<!-- #endregion -->

```python
import torch
import numpy as np
from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import numpy as np

im = Image.open('persistent-storage/coco/000000096944.jpg').convert('RGB')

a = np.array(im) / 255

lab = rgb2lab(a)
l, a, b = np.moveaxis(lab, -1, 0)

f, ax = plt.subplots(2, 2)

ax[0][0].imshow(im)
ax[0][1].imshow(l, cmap='gray')
ax[1][0].imshow(a, cmap='gray')

ax[1][1].set_title('blueyellow')
ax[1][1].imshow(b, cmap='gray')
plt.show()

```

```python
def lab_comp_stats(l, a, b):
    print(f"L: min: {np.min(l)} max: {np.max(l)}")
    print(f"A: min: {np.min(a)} max: {np.max(a)}")
    print(f"B: min: {np.min(b)} max: {np.max(b)}")

lab_comp_stats(l,a,b)

nl = np.interp(l, (0, 100), (-1, +1))
na = np.interp(a, (-128, 127), (-1, +1))
nb = np.interp(b, (-128, 127), (-1, +1))

lab_comp_stats(nl,na,nb)

nnl = np.interp(nl, (-1, +1), (0, 100))
nna = np.interp(na, (-1, +1), (-128, 127))
nnb = np.interp(nb, (-1, +1), (-128, 127))

lab_comp_stats(nnl,nna,nnb)

rgb = lab2rgb(np.moveaxis(np.array([nnl, nna, nnb]), 0, -1))
plt.imshow(rgb)

```

The dataset is loading images from the directory and preprocessing them so they can be sent directly to the model.


```python
import torch
import numpy as np
from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import numpy as np
```

```python
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io, transform
from skimage.color import rgb2lab
import glob

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, device, transforms):
        self.root_dir = root_dir
        self.device = device
        self.transforms = transforms
        
        self.images = glob.glob(f'{self.root_dir}/*.jpg')
        self.len = len(self.images)

    def get_image(self, idx):
        image = io.imread(f'{self.images[idx]}')
        image = self.transforms(image)

        return image

    def preprocess_image(self, image):
        l, a, b = np.moveaxis(rgb2lab(image), -1, 0)

        # normalize image
        l = np.interp(l, (0, 100), (-1, +1))
        a = np.interp(a, (-128, 127), (-1, +1))
        b = np.interp(b, (-128, 127), (-1, +1))

        x = torch.Tensor(l).unsqueeze(0)
        y = torch.Tensor([a,b])

        return x, y

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = self.get_image(idx)
        return self.preprocess_image(image)

```

# Unet baseline

I used Unet networks as a baseline. I used both the smaller version (UNetMini) and the full version.


```python
from torch import nn

class UNetMini(nn.Module):

    def __init__(self):
        super(UNetMini, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = nn.MaxPool2d((2, 2))(out1)

        out2 = self.block2(out_pool1)
        out_pool2 = nn.MaxPool2d((2, 2))(out2)

        out3 = self.block3(out_pool2)
        out_up1 = nn.Upsample(scale_factor=2)(out3)

        out4 = torch.cat((out_up1, out2), dim=1)
        out4 = self.block4(out4)

        out_up2 = nn.Upsample(scale_factor=2)(out4)
        out5 = self.block5(torch.cat((out_up2, out1), dim=1))

        return self.out_conv(out5)

# Full unet
unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=2, init_features=32)
```

# Deep Koalarization

Deep koalarization uses pre-trained Inception-ResNet-v2 as a feature extractor. Because Efficient-Net B4 is smaller, quicker, and better performing on ImageNet a decided to use this network instead.

```python
import torch
from torch import nn
import numpy as np


class KoalaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.efficient_net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True).to("cuda")
        for param in self.efficient_net.parameters():
            param.requires_grad = False


        self.decoder = nn.Sequential(
            nn.Conv2d(1256, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        batch_size = len(x)

        enc_ten = self.encoder(x)

        # fusion
        feat_ten = self.efficient_net(x.repeat(1, 3, 1, 1))
        feat_ten = torch.permute(feat_ten.repeat(28, 28, 1, 1), (2, 3, 1, 0))
        fused_ten = torch.cat((enc_ten, feat_ten), dim=1)

        out_ten = self.decoder(fused_ten)

        return out_ten
```

# Training

Even though the original DeepKoalarization paper doesn't mention any data augmentation I decided to use RandomHorizontalFlip and random cropping which seemed to help generalize over the dataset.


```python
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((268, 268)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224))
])

```

```python
from torch.utils.data import DataLoader
from torch import optim
import torch

batch_size = 32

dataset = ColorizationDataset('persistent-storage/coco', "cuda", train_transforms)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 1000, 1000])
train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=32, shuffle=True, prefetch_factor=2)
val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=32) # 1000 images as validation set

test_dataset = ColorizationDataset('persistent-storage/coco-val', "cuda", test_transforms)
test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=16, shuffle=True)
```


Main training cycle. The training ran for about 10 hours (70 epochs) on 80k training images (on A10/A40? - can't recall). Even though the loss at the end was still slightly dropping, the visual difference between the last 5 epochs was minimal. All trained models can be found at: https://drive.google.com/drive/folders/1Ey-ZRnMkdMVf5soanxvMEVzZcXTqI5Kz?usp=sharing

```python
model = KoalaNet()
model.cuda()

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters())
```

```python
from torch import optim
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
import gc

epoch_size = len(train_dataset) / batch_size

writer = SummaryWriter()
total_losses = []
total_ssim_list= []
total_loss_list = []

for epoch in range(100):
    i = 0
    model.train()
    total_loss = 0
    for x, y in train_dataloader:
        i += 1
        x = x.to("cuda")
        y = y.to("cuda")

        optimizer.zero_grad()

        out = model(x)

        loss = loss_fn(out, y)
        total_loss += loss.item()

        loss.backward()

        if i % 50 == 0:
            rloss = total_loss / i
            print(f"running loss [{i}/{epoch_size}]: {rloss}")

            gc.collect()
            torch.cuda.empty_cache()
            break
        optimizer.step()
    
    total_losses.append(total_loss)
    torch.save(model.state_dict(), f'persistent-storage/unet.pth')
    
    model.eval()
    with torch.no_grad():
        total_ssim = 0
        total_loss = 0
        for x, y in val_dataloader:
            x = x.to("cuda")

            out = model(x).cpu()
            
            total_loss += loss_fn(out, y)

            out = np.interp(out, (-1, +1), (0, 1))
            y = np.interp(y, (-1, +1), (0, 1))
            
            for b in range(len(out)):
                total_ssim += 2 - ssim(out[b][0], y[b][0]) - ssim(out[b][1], y[b][1])
        
        print(f'val inv ssim: {total_ssim} val loss: {total_loss}')
        total_ssim_list.append(total_ssim)
        total_loss_list.append(total_loss)
        writer.add_scalar('ssim/val', total_ssim, epoch)
        writer.add_scalar('loss/val', total_loss, epoch)
```

# Evaluation

Most papers evaluate their models according to the success with which they can fool the person who is guessing which image is original to the grayscale image. If humans prefer colorized images over ground truths, then the model is considered better performing. Also, it must be taken into account that not all colorization methods are heading in the same direction. Some models try to solve the overly conservative guessing and try to guess colors "more aggressively", some just try to match reality as closely as possible without any artistic effects, etc. Because of these reasons, it's hard to come up with a universal mathematical function to evaluate colorization methods.

Some of the used methods are a Peak signal-to-noise ratio (PSNR), often used alongside compression, and Structural similarity (SSIM), also used in compression, pattern recognition, and image restoration. In the end, I used SSIM to validate progress on the validation set at the end of every epoch.

Even though the function isn't universal, this way, we can at least ensure that the model is going the right way. 

```python
from skimage.metrics import structural_similarity as ssim
# experiments with ssim, we can see the degredation curve when applying noise

x, y = val_dataset[0]

y = y.detach().cpu().numpy()
x = x.detach().cpu()[0].numpy()

l = np.interp(x, (-1, +1), (0, 100))
y0 = np.interp(y[0], (-1, +1), (0, 1))
yy0 = y0 * 0.5
y1 = np.interp(y[1], (-1, +1), (-128, 127))

print(ssim(y0, y0 * 0.1))
print(ssim(y0, y0 * 0.4))
print(ssim(y0, y0 * 0.8))
print(ssim(y0, y0 * 0.9))
```

```python
def validate_model(model):
    model.eval()
    with torch.no_grad():
        total_ssim = 0
        total_loss = 0
        for x, y in val_dataloader:
            x = x.to("cuda")

            out = model(x).cpu()
            
            total_loss += loss_fn(out, y)

            out = np.interp(out, (-1, +1), (0, 1))
            y = np.interp(y, (-1, +1), (0, 1))
            
            for b in range(len(out)):
                total_ssim += 2 - ssim(out[b][0], y[b][0]) - ssim(out[b][1], y[b][1])
        
        print(f'val inv ssim: {total_ssim} val loss: {total_loss}')
        total_ssim_list.append(total_ssim)
        total_loss_list.append(total_loss)


koalamodel = KoalaNet().cuda()
koalamodel.load_state_dict(torch.load('./persistent-storage/final-koala.pth'))

unetminimodel = UNetMini().cuda()
unetminimodel.load_state_dict(torch.load('./persistent-storage/unet-mini.pth'))

validate_model(koalamodel)
validate_model(unetminimodel)
validate_model(model)
```

```python
def predict_unseen_image(index, model=koalamodel):
    x, y = test_dataset[index]
    x = x.to("cuda")
    
    pa, pb = model(x.resize(1, 1, 224, 224))[0]

    y = y.detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    pa = pa.detach().cpu().numpy()
    pb = pb.detach().cpu().numpy()

    # scale back from -1 to 1 to LAB values
    l = np.interp(x, (-1, +1), (0, 100))
    a = np.interp(pa, (-1, +1), (-128, 127))
    b = np.interp(pb, (-1, +1), (-128, 127))

    y0 = np.interp(y[0], (-1, +1), (-128, 127))
    y1 = np.interp(y[1], (-1, +1), (-128, 127))

    l = np.squeeze(l, 0)

    orig = np.array([l, y0, y1])
    origrgb = lab2rgb(np.moveaxis(orig, 0, -1))

    lab = np.array([l, a, b])
    rgb = lab2rgb(np.moveaxis(lab, 0, -1))

    return l, origrgb, rgb

```

```python

f, ax = plt.subplots(1, 5, figsize=(20,20))
im1 = predict_unseen_image(78, unetminimodel)
im2 = predict_unseen_image(78, unet_model)
im3 = predict_unseen_image(78, koalamodel)
ax[0].imshow(im1[0], cmap='gray')
ax[1].imshow(im1[1])
ax[2].imshow(im1[2])
ax[3].imshow(im2[2])
ax[4].imshow(im3[2])
plt.show()
```

```python
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from PIL import Image


model.eval()

print("SHOWCASE")
print("each row contains three images: grayscale, original and predicted from grayscale respectively")
print("all presented images are taken from test set and thus was never seenby the model")
print("-----")
print("Common problem when colorizing images is deciding which color use when the color of the object may not be clear")
qq = [1152, 1159, 1123]
f, ax = plt.subplots(len(qq), 3, figsize=(10,10))
for i in range(len(qq)):
    im1 = predict_unseen_image(qq[i])
    ax[i][0].imshow(im1[0], cmap='gray')
    ax[i][1].imshow(im1[1])
    ax[i][2].imshow(im1[2])
plt.show()

print("Another common problem is colorizing very small objects or image with too many objects in the scene")
qq = [34, 78, 1098]
f, ax = plt.subplots(len(qq), 3, figsize=(10,10))
for i in range(len(qq)):
    im1 = predict_unseen_image(qq[i])
    ax[i][0].imshow(im1[0], cmap='gray')
    ax[i][1].imshow(im1[1])
    ax[i][2].imshow(im1[2])
plt.show()

print("On the other hand, common themes like nature are often the more easier ones.")
print("But, the same problems remains even there e.g. the small girrafes get little bit green by trees behind them, from this same problem suffers also the original DeepKoalarizationNet")
qq = [738, 32, 712, 1046, 186, 1136]
f, ax = plt.subplots(len(qq), 3, figsize=(20,60))
for i in range(len(qq)):
    im1 = predict_unseen_image(qq[i])
    ax[i][0].imshow(im1[0], cmap='gray')
    ax[i][1].imshow(im1[1])
    ax[i][2].imshow(im1[2])
plt.savefig('third.png')
plt.show()

print("Also other examples show that network often gets the general idea correctly but still missing precission")

q = [213, 834, 513, 93, 425, 1026, 1037, 943, 350, 1039]
# 934, 6
# 943, 350
f, ax = plt.subplots(len(q), 3, figsize=(20,50))
for i in range(len(q)):
    im1 = predict_unseen_image(q[i])
    ax[i][0].imshow(im1[0], cmap='gray')
    ax[i][1].imshow(im1[1])
    ax[i][2].imshow(im1[2])
plt.show()

```

# Results

Results on unseen images

Grayscale on the left; original in the middle; colorized on the right

![result4](/images/colorization/result4.png)

![result1](/images/colorization/result1.png)

![result3](/images/colorization/result3.png)

![result2](/images/colorization/result2.png)

![result5](/images/colorization/result5.png)




