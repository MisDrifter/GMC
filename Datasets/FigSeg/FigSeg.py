import sys
import os
sys.path.append(os.getcwd())
from Datasets.FigSeg.src.Unet import UNet
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

#----------------util-functions-------------------
def convert_segm_rgb_to_one_hot(segmentation_label):
    """This method is used to convert a segmentation images of 3 channels (RGB)
    to a 6 channels (segmentation classes) as a one-hot encoding for each class (color)
    input is an image of shape WxHx3 out put os WxHx6"""
    # Predefined set of color coding's for transformation from RGB to class name (hair,skin,etc)
    red_background = np.array([255, 0, 0])
    yellow_face = np.array([255, 255, 0])
    brown_hair = np.array([127, 0, 0])
    cayn_nose = np.array([0, 255, 255])
    blue_eyes = np.array([0, 0, 255])
    green_mouth = np.array([0, 255, 0])
    red_background_out = (segmentation_label == red_background).all(2)
    yellow_face_out = (segmentation_label == yellow_face).all(2)
    brown_hair_out = (segmentation_label == brown_hair).all(2)
    cayn_nose_out = (segmentation_label == cayn_nose).all(2)
    blue_eyes_out = (segmentation_label == blue_eyes).all(2)
    green_mouth_out = (segmentation_label == green_mouth).all(2)

    one_hot_segment_label = np.stack(
        [red_background_out,
         yellow_face_out, brown_hair_out, cayn_nose_out, blue_eyes_out, green_mouth_out],
        axis=2, out=None)

    return one_hot_segment_label.astype(int)


def convert_seg_one_hot_to_rgb(segm_label_one_hot):
    """This method converts the class encoding representation of a segmentation image back to RGB format for display"""
    segmentation_label_rgb = np.zeros(shape=(img_resize_factor, img_resize_factor, 3), dtype=int)

    # 6 Classes are: background-red-0, skin-yellow-1, hair-brown-2, nose-cayn-3, eyes-blue-4, mouth-green-5
    red_background_indexes = np.where(segm_label_one_hot[:, :] == 0)
    yellow_face_indexes = np.where(segm_label_one_hot[:, :] == 1)
    brown_hair_indexes = np.where(segm_label_one_hot[:, :] == 2)
    cayn_nose_indexes = np.where(segm_label_one_hot[:, :] == 3)
    blue_eyes_indexes = np.where(segm_label_one_hot[:, :] == 4)
    green_mouth_indexes = np.where(segm_label_one_hot[:, :] == 5)

    segmentation_label_rgb[red_background_indexes[0], red_background_indexes[1]] = red_background
    segmentation_label_rgb[yellow_face_indexes[0], yellow_face_indexes[1]] = yellow_face
    segmentation_label_rgb[brown_hair_indexes[0], brown_hair_indexes[1]] = brown_hair
    segmentation_label_rgb[cayn_nose_indexes[0], cayn_nose_indexes[1]] = cayn_nose
    segmentation_label_rgb[blue_eyes_indexes[0], blue_eyes_indexes[1]] = blue_eyes
    segmentation_label_rgb[green_mouth_indexes[0], green_mouth_indexes[1]] = green_mouth

    return segmentation_label_rgb

#----------------data-section-------------------
class FaceSegmentationDataset(Dataset):
    """Face part segmentation dataset"""

    def __init__(self, data_dir, label_dir, img_resize_factor, transform=None):
        """
        Args:
            data_dir (string): Directory with all the facial images.
            label_dir (string) : Directory with all the face segmentation images (the label)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_resize_factor = img_resize_factor

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        img_name = sorted(os.listdir(self.data_dir), key=lambda x:int(x[:-4]))[idx]

        face_img = cv2.imread(os.path.join(self.data_dir, img_name))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        segmentation_label = cv2.imread(os.path.join(self.label_dir, img_name))
        segmentation_label = cv2.cvtColor(segmentation_label, cv2.COLOR_RGB2BGR)
        segmentation_label = cv2.resize(segmentation_label, dsize=(self.img_resize_factor, self.img_resize_factor),
                                        interpolation=cv2.INTER_CUBIC)
        segmentation_label = convert_segm_rgb_to_one_hot(segmentation_label)

        if self.transform:
            face_img = self.transform(face_img)
            segmentation_label = torch.from_numpy(segmentation_label).permute(2, 0, 1)

        return face_img, segmentation_label
        # torch(3,256,256) torch(6,256,256)

        
def load_data(data_sets_folder, img_resize_factor, batch_size=1):
    # Transform loaded images, resize them to a symmetrical form and then normalize their pixel values
    transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(size=(img_resize_factor, img_resize_factor)),
     transforms.ToTensor(),
     transforms.Normalize([0.4283, 0.335, 0.275], [0.240, 0.219, 0.216])
     # Mean and STD Values calc by Analyze_dataset()
     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #image net values
     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # default values
     ])

    face_seg_dataset = FaceSegmentationDataset(data_dir=data_sets_folder + '/Train_RGB/',
                                           label_dir=data_sets_folder + '/Train_Labels/', img_resize_factor=img_resize_factor, transform=transform)
    train_size = len(face_seg_dataset)
    print('train_size: ', train_size)

    # Calculate the images population mean and std for better transform
    data_loader = DataLoader(face_seg_dataset, batch_size=batch_size, shuffle=False, num_workers=4)# num_workers是多进程的东西
    return data_loader


def train(model, train_loader, epoch, criterion, device):
    L = []
    for i in tqdm(range(epoch)):
        for batch_id, (face_imgs, segm_labels) in enumerate(train_loader):
            #print(batch_id)
            # zero the parameter gradients

            optimizer.zero_grad()

            # if cuda_available:
            face_imgs = face_imgs.to(device)
            segm_labels = segm_labels.to(device)

            # forward + backward + optimize
            outputs = model(face_imgs)
            labels = torch.argmax(segm_labels, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            L.append(loss.to('cpu').detach())
    plt.plot(L)
    plt.show()
    # 得到x,y,h
    for batch_id, (face_imgs, segm_labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # if cuda_available:
        face_imgs = face_imgs.to(device)

        # forward + backward + optimize
        if batch_id==0:
            h = model(face_imgs).to('cpu').detach()
            x = face_imgs.to('cpu').detach()
            y = segm_labels.to('cpu').detach()
        else:
            h = torch.cat([h, model(face_imgs).to('cpu').detach()], dim=0)
            x = torch.cat([x, face_imgs.to('cpu').detach()], dim=0)
            y = torch.cat([y, segm_labels.to('cpu').detach()], dim=0)
    print(h.shape)
    print(y.shape)
    print(x.shape)
    return h,x,y


def save_data(save_folder,h,x,y): #输出为.h文件 有三个key x y h
    save_path = os.path.join(save_folder, 'result.h')
    f = h5py.File(save_path, 'w')
    f['x'] = x
    f['y'] = y
    f['h'] = h
    f.close()

if __name__ =='__main__':
    image_channels = 3
    mask_channels = 6 #six kinds of segments
    batch_size = 10
    learning_rate = 2.5e-4
    epoch = 30
    img_resize_factor = 256

    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    data_sets_folder = os.path.join(os.getcwd(), 'datasets\FigSeg\Data')
    save_folder = os.path.join(os.getcwd(), 'datasets\FigSeg\Result')
    os.makedirs(save_folder, exist_ok=True)
    model = UNet(in_channels=image_channels, out_channels=mask_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = load_data(data_sets_folder, img_resize_factor, batch_size)

    h,x,y = train(model, train_loader, epoch, criterion, device)
    save_data(save_folder, h, x, y)
