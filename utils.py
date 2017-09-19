import cv2, os
import numpy as np
import matplotlib.image as mpimg
import torch.utils.data as data

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 16, 32, 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def rgb2hsv(image):
    """
    Convert the image from RGB to HSV 
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = rgb2hsv(image)
    #image = resize(image)

    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument_img(image, steering_angle):
    if np.random.rand() < 0.6:
        image, steering_angle = random_flip(image, steering_angle)
        image, steering_angle = random_translate(image, steering_angle, 100, 10)
        image = random_shadow(image)
        image = random_brightness(image)
    image = preprocess(image)
    return image, steering_angle

class CarDataset(data.Dataset):
    """
    Dataset wrapping images and target steering_angle.
    """

    def __init__(self, X, y, data_dir, is_training, transform=None):

        self.X = X
        self.y = y
        self.data_dir = data_dir
        self.is_training = is_training
        self.transform = transform


    def __getitem__(self, index):
        center, left, right = self.X[index]
        steering_angle = self.y[index]
        
        if self.is_training and np.random.rand() < 0.6:
            image, steering_angle = augument(self.data_dir, center, left, right, steering_angle)
        else:
            image = load_image(self.data_dir, center) 
        image = preprocess(image)

        if self.transform is not None:
            image = self.transform(image)
            
        #label = torch.from_numpy(self.y_train[index])
        return image, steering_angle
    
    def __len__(self):
        return self.X.shape[0]
    
class CarDataset3Img(data.Dataset):
    """
    Dataset wrapping images and target steering_angle.
    """

    def __init__(self, X, y, data_dir, transform=None):

        self.X = X
        self.y = y
        self.data_dir = data_dir
        self.transform = transform


    def __getitem__(self, index):
        center, left, right = self.X[index]
        steering_angle = self.y[index]
        image_left, steering_angle_left = augument_img(load_image(self.data_dir, left), steering_angle + 0.1) 

        image_center, steering_angle_center = augument_img(load_image(self.data_dir, center), steering_angle) 

        image_right, steering_angle_right = augument_img(load_image(self.data_dir, right), steering_angle - 0.1) 

        
        if self.transform is not None:
            image_left = self.transform(image_left)
            image_center = self.transform(image_center)
            image_right = self.transform(image_right)
            
        #label = torch.from_numpy(self.y_train[index])
        return (image_center, steering_angle_center), (image_left, steering_angle_left), (image_right, steering_angle_right)
    
    def __len__(self):
        return self.X.shape[0]
