""" for texture sharing sdk. """
import sys
sys.path.append('Library')
import numpy as np
import argparse
import time
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *

""" for face recognition. """
import face_recognition
from PIL import Image, ImageDraw

""" for segmentation model. """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import matplotlib.pyplot as plt
DATA_DIR = './data/CamVid/'
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')



""" add segmentation model dataset class. """
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)



""" parsing and configuration. """
def parse_args():
    desc = "Spout receiver/sender template"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--type', type=str, default='input-output', help='input/output/input-output')
    parser.add_argument('--spout_size', nargs = 2, type=int, default=[480, 320], help='Width and height of the spout receiver and sender')   
    parser.add_argument('--spout_input_name', type=str, default='input', help='Spout receiving name')  
    parser.add_argument('--spout_output_name', type=str, default='output', help='Spout sending name')  
    parser.add_argument('--silent', type=bool, default=False, help='Hide pygame window')
    return parser.parse_args()

""" visualize function for segmentation model. """
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

""" apply albumentations to grow accuracy for the model. """
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

""" add function for the model. """
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

""" add function for the model. """
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

""" add function for the model. """
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

""" segment with the model for video capture. """
# def capture_segment(mirror=True, size=None):
#     """segment capture video from camera"""
#     preprocess = get_preprocessing(preprocessing_fn)
#     cap = cv2.VideoCapture(0)
#     window_name = 'capture_segment'
#     while True:
#         ret, frame = cap.read()
#         if mirror is True:
#             frame = frame[:,::-1]
#         if size is not None and len(size) == 2:
#             frame = cv2.resize(frame, size)
#         image = frame
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         sample = preprocess(image=image, mask=None)
#         x_tensor = torch.from_numpy(sample["image"])
#         x_tensor = x_tensor.to(DEVICE).unsqueeze(0)
#         pr_mask = best_model.predict(x_tensor)
#         pr_mask = (pr_mask.squeeze().cpu().numpy().round())
#         cv2.imshow('capture_camera', frame)
#         cv2.imshow(window_name, pr_mask)
#         k = cv2.waitKey(1)
#         if k == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()



""" main own functions. """
def main_pipeline(data):

    """ segmentation. """
    preprocess = get_preprocessing(preprocessing_fn)

    image = data

    sample = preprocess(image=image, mask=None)
    x_tensor = torch.from_numpy(sample["image"])
    x_tensor = x_tensor.to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask *= 255
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)

    output = pr_mask

    return output



""" main script for spout. """
def main():

    # parse arguments
    args = parse_args()
    # window details
    width = args.spout_size[0] 
    height = args.spout_size[1] 
    display = (width,height)
    
    req_type = args.type
    receiverName = args.spout_input_name 
    senderName = args.spout_output_name
    silent = args.silent
    
    # window setup
    pygame.init() 
    pygame.display.set_caption(senderName)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # OpenGL init
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)

    if req_type == 'input' or req_type == 'input-output':
        # init spout receiver
        spoutReceiverWidth = width
        spoutReceiverHeight = height
        # create spout receiver
        spoutReceiver = SpoutSDK.SpoutReceiver()
	    # Its signature in c++ looks like this: bool pyCreateReceiver(const char* theName, unsigned int theWidth, unsigned int theHeight, bool bUseActive);
        spoutReceiver.pyCreateReceiver(receiverName,spoutReceiverWidth,spoutReceiverHeight, False)
        # create textures for spout receiver and spout sender 
        textureReceiveID = glGenTextures(1)
        
        # initalise receiver texture
        glBindTexture(GL_TEXTURE_2D, textureReceiveID)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # copy data into texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, spoutReceiverWidth, spoutReceiverHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, None ) 
        glBindTexture(GL_TEXTURE_2D, 0)

    if req_type == 'output' or req_type == 'input-output':
        # init spout sender
        spoutSender = SpoutSDK.SpoutSender()
        spoutSenderWidth = width
        spoutSenderHeight = height
	    # Its signature in c++ looks like this: bool CreateSender(const char *Sendername, unsigned int width, unsigned int height, DWORD dwFormat = 0);
        spoutSender.CreateSender(senderName, spoutSenderWidth, spoutSenderHeight, 0)
        # create textures for spout receiver and spout sender 
    textureSendID = glGenTextures(1)

    # loop for graph frame by frame
    while(True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                spoutReceiver.ReleaseReceiver()
                pygame.quit()
                quit()
        
        if req_type == 'input' or req_type == 'input-output':
            # receive texture
            # Its signature in c++ looks like this: bool pyReceiveTexture(const char* theName, unsigned int theWidth, unsigned int theHeight, GLuint TextureID, GLuint TextureTarget, bool bInvert, GLuint HostFBO);
            if sys.version_info[1] == 5:
                spoutReceiver.pyReceiveTexture(receiverName, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID, GL_TEXTURE_2D, False, 0)
            else:
                spoutReceiver.pyReceiveTexture(receiverName, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID.item(), GL_TEXTURE_2D, False, 0)

            glBindTexture(GL_TEXTURE_2D, textureReceiveID)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            # copy pixel byte array from received texture   
            data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)  #Using GL_RGB can use GL_RGBA 
            glBindTexture(GL_TEXTURE_2D, 0)
            # swap width and height data around due to oddness with glGetTextImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
            data.shape = (data.shape[1], data.shape[0], data.shape[2])
        else:
            data = np.ones((width,height,3))*255
        
        # call our main function
        output = main_pipeline(data)
        
        # setup the texture so we can load the output into it
        glBindTexture(GL_TEXTURE_2D, textureSendID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # copy output into texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, output )
            
        # setup window to draw to screen
        glActiveTexture(GL_TEXTURE0)
        # clean start
        glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
        # reset drawing perspective
        glLoadIdentity()
        # draw texture on screen
        glBegin(GL_QUADS)

        glTexCoord(0,0)        
        glVertex2f(0,0)

        glTexCoord(1,0)
        glVertex2f(width,0)

        glTexCoord(1,1)
        glVertex2f(width,height)

        glTexCoord(0,1)
        glVertex2f(0,height)

        glEnd()
        
        if silent:
            pygame.display.iconify()
                
        # update window
        pygame.display.flip()        

        if req_type == 'output' or req_type == 'input-output':
            # Send texture to spout...
            # Its signature in C++ looks like this: bool SendTexture(GLuint TextureID, GLuint TextureTarget, unsigned int width, unsigned int height, bool bInvert=true, GLuint HostFBO = 0);
            if sys.version_info[1] == 5:
                spoutSender.SendTexture(textureSendID, GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)
            else:
                spoutSender.SendTexture(textureSendID.item(), GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)



""" load best trained model for predict. """
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['pedestrian']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

best_model = torch.load('./best_model.pth')



""" preparation for visualise test script. """
# test_dataset = Dataset(
#     x_test_dir, 
#     y_test_dir, 
#     augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )
# test_dataloader = DataLoader(test_dataset)

# 1st visualise test. #
# dataset = Dataset(x_train_dir, y_train_dir, classes=['pedestrian'])
# image, mask = dataset[4] # get some sample
# 2nd visualise test. #
# augmented_dataset = Dataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation(), classes=['pedestrian'],)
# 3rd visualise test. #
# test_dataset_vis = Dataset(x_test_dir, y_test_dir, classes=CLASSES,)
# for i in range(5):
#     n = np.random.choice(len(test_dataset))
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
#     gt_mask = gt_mask.squeeze()
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())



""" check if run in command line. """
if __name__ == '__main__':
    main()
    # capture_segment(True, (int(1280/2), int(960/2)))

    """ visualise test script. """
    # 1st visualise test. #
    # visualize(image=image, cars_mask=mask.squeeze(),)
    # 2nd visualise test. #
    # for i in range(3):
    #     image, mask = augmented_dataset[1]
    #     visualize(image=image, mask=mask.squeeze(-1))
    # 3rd visualise test. #
    # visualize(image=image_vis, ground_truth_mask=gt_mask, predicted_mask=pr_mask)
