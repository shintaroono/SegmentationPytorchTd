### 1st setup and sample. ###
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import matplotlib.pyplot as plt


DATA_DIR = './data/CamVid/'

# if not os.path.exists(DATA_DIR):
#     print('Loading data...')
#     os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
#     print('Done!')


x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


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


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

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


# # Lets look at data we have.

# dataset = Dataset(x_train_dir, y_train_dir, classes=['pedestrian'])

# image, mask = dataset[4] # get some sample
# visualize(
#     image=image, 
#     cars_mask=mask.squeeze(),
# )




### apply albumentationsto up accuracy. ###
import albumentations as albu


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

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

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


# # Visualize resulted augmented images and masks.
augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=['pedestrian'],
)

# same image with different random transforms.
# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask.squeeze(-1))




### create model and train. ###
import torch
# import numpy as np
import segmentation_models_pytorch as smp


ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['pedestrian']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder.
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# create epoch runners.
# it is a simple loop of iterating over dataloader`s samples.
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# train model for 40 epochs.
# def main():
#     max_score = 0
    
#     for i in range(0, 40):
        
#         print('\nEpoch: {}'.format(i))
#         train_logs = train_epoch.run(train_loader)
#         valid_logs = valid_epoch.run(valid_loader)
        
#         # do something (save model, change lr, etc.).
#         if max_score < valid_logs['iou_score']:
#             max_score = valid_logs['iou_score']
#             torch.save(model, './best_model.pth')
#             print('Model saved!')
            
#         if i == 25:
#             optimizer.param_groups[0]['lr'] = 1e-5
#             print('Decrease decoder learning rate to 1e-5!')

# if __name__ == '__main__':
#     main()




### test best saved model. ###
best_model = torch.load('./best_model.pth')


# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

# logs = test_epoch.run(test_dataloader)




### visualize predictions. ###
# test dataset without transformations for image visualization. #
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)


# for i in range(5):
#     n = np.random.choice(len(test_dataset))
    
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
    
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
#     visualize(
#         image=image_vis, 
#         ground_truth_mask=gt_mask, 
#         predicted_mask=pr_mask
#     )




### cv2 web cam capture test. ###
# # import cv2

# def capture_camera(mirror=True, size=None):
#     """capture video from camera"""

#     cap = cv2.VideoCapture(0)

#     # w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     # print(w) # original is 640. #
#     # print(h) # original is 480. #

#     while True:
#         ret, frame = cap.read()

#         if mirror is True:
#             frame = frame[:,::-1]

#         if size is not None and len(size) == 2:
#             frame = cv2.resize(frame, size)

#         cv2.imshow('camera capture', frame)

#         k = cv2.waitKey(1)
#         if k == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # use web cam capture. #
# # capture_camera(True, (int(640*2), int(480*2)))


### cv2 web cam segmentation. ###
def capture_segment(mirror=True, size=None):
    """segment capture video from camera"""

    preprocess = get_preprocessing(preprocessing_fn)

    cap = cv2.VideoCapture(0)
    window_name = 'capture_segment'

    # w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(w,h)

    while True:
        ret, frame = cap.read()

        if mirror is True:
            frame = frame[:,::-1]

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = preprocess(image=image, mask=None)
        x_tensor = torch.from_numpy(sample["image"])
        x_tensor = x_tensor.to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        cv2.imshow('capture_camera', frame)
        cv2.imshow(window_name, pr_mask)
        # plt.imshow(frame)
        # plt.imshow(pr_mask)
        # plt.show()

        # visualize(
        #     image=frame,
        #     predicted_mask=pr_mask
        # )

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# capture_segment()
# capture_segment(True, (int(320), int(480)))
capture_segment(True, (int(1280/2), int(960/2)))