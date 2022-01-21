# source: https://github.com/bdrad/project-conv-meleon/blob/3DMNIST/MNIST_3D/MNIST_3D_data.py

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy
import torch
import cv2
from sklearn.utils import shuffle
from scipy.stats import truncnorm, norm
from scipy import ndimage


def scipy_clipped_zoom_3d(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width, length = img.shape[:3] # It's also the final desired shape
    new_height, new_width, new_length = int(height * zoom_factor), int(width * zoom_factor), int(length * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1, z1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2, max(0, new_length - length) // 2
    y2, x2, z2 = y1 + height, x1 + width, z1 + length
    bbox = np.array([y1,x1,z1,y2,x2,z2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1,x1,z1,y2,x2,z2 = bbox
    cropped_img = img[y1:y2, x1:x2, z1:z2]

    # Handle padding when downscaling
    resize_height, resize_width, resize_length = min(new_height, height), min(new_width, width), min(new_length, length)
    pad_height1, pad_width1, pad_length1 = (height - resize_height) // 2, (width - resize_width) //2, (length - resize_length) //2
    pad_height2, pad_width2, pad_length2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1, (length - resize_length) - pad_length1

    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2), (pad_length1, pad_length2)] + [(0,0)] * (img.ndim - 3)
    
    width_cropped, height_cropped, length_cropped = cropped_img.shape
    
    width_zf, height_zf, length_zf = resize_width/width_cropped, resize_height/height_cropped, resize_length/length_cropped
    
    result = ndimage.zoom(cropped_img, (width_zf, height_zf, length_zf), order = 0)
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width and result.shape[2] == length
    return result

class MNIST_3D(torch.utils.data.Dataset):
    """"The 3D MNIST data set is an extension of the MNIST dataset """

    def __init__(self, path_to_vectors, train=True, tranforms=None):
        data, targets = self.load_3D_MNIST(path_to_vectors, train)
        self.transforms = transforms
        self.data = dataset
        self.targets = targets
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    def load_3D_MNIST(self, path_to_vectors, train):
        data = np.load(path_to_vectors)
        if train:
            return data['X_train'], data['y_train']
        else:
            return data['X_test'], data['y_test']

    def display_3D_image(image, val_cutoff=1e-6):
        """Takes in a 3D image and visualizes it
        """
        plt.rcParams["figure.figsize"] = [12, 5]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        small_flags = np.abs(data) < val_cutoff
        data[small_flags] = 0
        z, x, y = data.nonzero()
        color = np.ones(z.shape) * 255
        ax.scatter(x, y, z, c=color, alpha=0.5)
        # label axis for coordinates
        plt.show()

    def get_middle_slice(image):
        return image[:, :, image.shape[2] // 2]

    def display_2D_image_slice(image):
        """Takes in a 3D image and visualizes its middle slice
        """
        plt.imshow(MNIST_3D.get_middle_slice(image))


class MNIST_3D_Growing(MNIST_3D):
    def __init__(self, data_path, zoom_func, size, train=True, num_time_points=3,
                 transforms=None,
                 class_names=['growing', 'not_growing'],
                 class_growth_factors={'growing': 1.1, 'not_growing': 1}
                 ):
        assert num_time_points >= 1, "number of zoom factors should equal number of time points"
        assert isinstance(num_time_points, int)
        assert isinstance(class_names, list)
        assert isinstance(class_growth_factors, dict)
        assert isinstance(train, bool)

        # #set device
        # if torch.cuda.is_available():
        #     print(torch.cuda.get_device_name(0))
        #     device = torch.device('cuda:0')
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # else:
        #     print('cpu')
        #     device = torch.device('cpu')

        for class_name in class_names:
            assert class_name in class_growth_factors.keys()

        data, _ = self.load_3D_MNIST(data_path, train)
        data = data[:size]
        # choose a subpart of dataa

        # split the data into even groups
        num_classes = len(class_names)
        splitted_X_lst = np.array_split(data, num_classes)

        X = {}
        X_transformed = {}
        y = {}
        for i in np.arange(num_classes):
            X_part = splitted_X_lst[i]
            class_name = class_names[i]
            X[class_name] = X_part

            X_transformed[class_name] = []

            y_part = np.full(len(X_part), i)
            y[class_name] = y_part

            # create time points for each classes
        for class_name in class_names:
            X_group = X[class_name]
            group_data = []
            for i in np.arange(len(X_group)):
                data_pt = X_group[i]

                time_points = []
                zoom_factor = 1
                for j in np.arange(num_time_points):
                    if j > 0:
                        zoom_factor = zoom_factor * class_growth_factors[class_name]

                    # TODO: put this step in the self.__get_item__ method later
                    zoomed_data = zoom_func(data_pt, zoom_factor)

                    # apply any transformations
                    if transforms:
                        zoomed_data = transforms(zoomed_data)

                    time_points.append(self.expand_dimension(zoomed_data))

                time_points = np.array(time_points)
                group_data.append(time_points)
            group_data = np.array(group_data)
            X_transformed[class_name] = group_data

        X, y = self.concate_and_shuffle(X_transformed, y)
        self.X = X
        self.y = y

    def expand_dimension(self, img):
        assert len(img.shape) == 3
        x, y, z = img.shape
        new_img = np.zeros((x, 1, y, z))
        new_img[:, 0, :, :] = img
        return new_img

    def concate_and_shuffle(self, X, y, random_state=0):
        assert len(X.keys()) > 1
        assert len(y.keys()) > 1

        classes = X.keys()

        X_total = None
        y_total = None

        for c in classes:
            if X_total is None or y_total is None:
                # initialize the variable to be the first part of data
                X_total = X[c]
                y_total = y[c]
            else:
                X_total = np.concatenate((X_total, X[c]))
                y_total = np.concatenate((y_total, y[c]))

        X_total, y_total = shuffle(X_total, y_total, random_state=random_state)
        return X_total, y_total

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.X[idx]
        # data = data.type(torch.float32)

        # convert labels into Long type tensor to fit the CrossEntropy Loss
        label = self.y[idx]
        # label = label.long()

        return data, label

    def __len__(self):
        return self.X.shape[0]

def truncnorm_distr(mean=180, std=40, lower=60, upper=360):
        """
        A distribution that produces a random doubling time (in days) following
        a truncated normal distribution parametrized by mean, variance,
        lower bound and upper bound.
        Refer to this link for details on truncnorm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        Args:
        num_data(int): number of doubling times to generate
        mean (int): mean of doubling time's distribution
        std (int): var of doubling time's distribution
        lower (int): lower bound of doubling time's distribution
        upper (int): upper bound of doubling time's distribution
        """

        a, b = (lower - mean) / std, (upper - mean) / std
        distribution = truncnorm(a, b, mean, std)
        return distribution
    
def identity(self, data):
    """
    a dummy noising function that adds zero noise.
    Args:
    data (int or np.array of int): input data
    """
    return data
    
class MNIST_3D_GrowthRate(MNIST_3D):
    def __init__(self, data_path, zoom_func, size, train=True, num_time_points=3,
                 transforms=None,
                 class_names=['malignant', 'benign'],
                 class_dtime_distri={'malignant': truncnorm_distr(mean=180, std=40, lower=60, upper=360),
                                                'benign': truncnorm_distr(mean=900, std=40, lower=720, upper=1080)},
                 class_etime_distri=truncnorm_distr(mean=105,std=10,lower=90,upper=120),
                 noise_func={'malignant': identity, 'benign': identity}

                 ):

        # checking function inputs
        assert num_time_points >= 1, "number of zoom factors should equal number of time points"
        assert isinstance(num_time_points, int)
        assert isinstance(class_names, list)
        assert isinstance(class_dtime_distri, dict)
#         assert isinstance(class_bounds, dict)
#         assert isinstance(class_mean, dict)
#         assert isinstance(class_std, dict)

        for class_name in class_names:
            assert class_name in class_dtime_distri.keys()
#             assert class_name in class_bounds.keys()
#             assert class_name in class_mean.keys()
#             assert class_name in class_std.keys()

        # select a subpart of data
        data, _ = self.load_3D_MNIST(data_path, train)
        data = data[:size]

        # split the data into even groups
        num_classes = len(class_names)
        splitted_X_lst = np.array_split(data, num_classes)

        X = {}
        X_transformed = {}
        DT = {} # doubling time labels
        ET = {} #elapsed time labels
        G = {} # growth rate labels
        y = {}

        for i in np.arange(num_classes):
            X_part = splitted_X_lst[i]
            class_name = class_names[i]
            X[class_name] = X_part

            X_transformed[class_name] = []

            # record the ground_truth labels and store them
            y_part = np.full(len(X_part), i)
            y[class_name] = y_part
            DT[class_name] = []
            G[class_name] = []
            ET[class_name] = []

        # create time points for each classes
        for class_name in class_names:
            X_curr_class = X[class_name]
            curr_class_data = []

            # doubling time generations for current class
            doubling_times = class_dtime_distri[class_name].rvs(size=len(X_curr_class))
            DT[class_name] = doubling_times

            # elapsed time should be of shape (T-1, N)
            elapsed_times = class_etime_distri[class_name].rvs(size=len(X_curr_class) * (num_time_points-1))
            elapsed_times = elapsed_times.reshape((len(X_curr_class), (num_time_points-1)))
            ET[class_name] = elapsed_times

            g_rates = growth_rates(doubling_times)
            G[class_name] = g_rates

            for i in np.arange(len(X_curr_class)):
                data_pt = X_curr_class[i]
                time_points = []
                time_elapsed = 0
                for j in np.arange(num_time_points):
                    if j == 0:
                        zoom_factor = 1
                    else:
                        time_elapsed += elapsed_times[i, (j-1)] #get the time passed for the ith data point at jth time step
                        curr_grate = noise_func[class_name](g_rates[i]) #current_growth_rate
                        zoom_factor = np.e ** (time_elapsed*curr_grate) #calculate ratio between new volume and original volume

                    zoomed_data = zoom_func(data_pt, zoom_factor)

                    # apply any transformations
                    if transforms:
                        zoomed_data = transforms(zoomed_data)

                    time_points.append(self.expand_dimension(zoomed_data))

                time_points = np.array(time_points)
                curr_class_data.append(time_points)
            curr_class_data = np.array(curr_class_data)
            X_transformed[class_name] = curr_class_data

        X, y, DT, ET, G = self.concate_and_shuffle(X_transformed, y, DT, ET, G)
        self.X = X
        self.y = y
        self.DT = DT
        self.ET = ET
        self.G = G

    def expand_dimension(self, img):
        assert len(img.shape) == 3
        x, y, z = img.shape
        new_img = np.zeros((x, 1, y, z))
        new_img[:, 0, :, :] = img
        return new_img

    def concate_and_shuffle(self, X, y, DT, ET, G, random_state=0):
        assert len(X.keys()) > 1
        assert len(y.keys()) > 1

        classes = X.keys()

        X_total = None
        y_total = None
        DT_total = None
        ET_total = None
        G_total = None

        for c in classes:
            if X_total is None:
                # initialize the variable to be the first part of data
                X_total = X[c]
                y_total = y[c]
                DT_total = DT[c]
                ET_total = ET[c]
                G_total = G[c]

            else:
                X_total = np.concatenate((X_total, X[c]))
                y_total = np.concatenate((y_total, y[c]))
                DT_total = np.concatenate((DT_total, DT[c]))
                ET_total = np.concatenate((ET_total, ET[c]))
                G_total = np.concatenate((G_total, G[c]))

        X_total, y_total, DT_total, ET_total, G_total = shuffle(X_total, y_total, DT_total, ET_total,
                                                                G_total, random_state=random_state)
        return X_total, y_total, DT_total, ET_total, G_total

    def growth_rate(self, d_time):
        """
        Calculate the growth rate given data's doubling time d_time,
        Growth rate is calculated based on the compound growth model:
        N = N_0 * (e ^kt) where k is the growth rate, t is time elapsed and N_0 is initial size
        Args:
        d_time (float/int): doubling time of the data (unit: per day)
        """
        return np.log(2) / d_time




    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.X[idx]
        # convert labels into Long type tensor to fit the CrossEntropy Loss
        label = self.y[idx]
        doubling_time = self.DT[idx]
        elapsed_time = self.ET[idx]
        growth_rate = self.G[idx]

        return data, label, doubling_time, elapsed_time, growth_rate

    def __len__(self):
        return self.X.shape[0]


class MNIST_3D(torch.utils.data.Dataset):
    """"The 3D MNIST data set is an extension of the MNIST dataset """

    def __init__(self, path_to_vectors, tranforms=None):
        with h5py.File(path_to_vectors, 'r') as dataset:
            # Split the data into training/test features/targets
            X_train = dataset["X_train"][:]
            targets_train = dataset["y_train"][:]
            X_test = dataset["X_test"][:]
            targets_test = dataset["y_test"][:]
        self.transforms = transforms
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    def display_3D_image(image, val_cutoff=1e-6):
        """Takes in a 3D image and visualizes it
        """
        plt.rcParams["figure.figsize"] = [12, 5]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        small_flags = np.abs(data) < val_cutoff
        data[small_flags] = 0
        z, x, y = data.nonzero()
        color = np.ones(z.shape) * 255
        ax.scatter(x, y, z, c=color, alpha=0.5)
        # label axis for coordinates
        plt.show()

    def get_middle_slice(image):
        return image[:, :, image.shape[2] // 2]

    def display_2D_image_slice(image):
        """Takes in a 3D image and visualizes its middle slice
        """
        plt.imshow(MNIST_3D.get_middle_slice(image))


class MNIST_3D_Standardize_Zoom(object):
    """Zoom the MNIST 3D image to a standardized shape.
    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size.
    """

    def __init__(self, output_size):
        assert len(output_size) == 3, 'Output shape should be 3-dimensional'
        self.output_size = output_size

    def __call__(self, data):
        # Check if data is an array of size (N, x_dim, y_dim, z_dim)
        assert len(data.shape) == 4, 'Input shape should be 4-dimensional'
        # Set zooming factor
        x, y, z = data[0, :, :, :].shape
        x_new, y_new, z_new = output_size
        x_zf, y_zf, z_zf = x_new / x, y_new / y, z_new / z
        # Loop through images and apply 3D zoom
        output = []
        for i in np.arange(len(data)):
            data_pt = data[i]
            zoomed_data_pt = scipy.ndimage.zoom(data_pt, (x_zf, y_zf, z_zf), order=0)
            output.append(zoomed_data_pt)
        # Convert back to numpy array
        output = np.array(output)
        _, x_out, y_out, z_out = output.shape
        # Check whether dimension of output matches (N, output_size[0], output_size[1], output_size[2])
        assert x_out == x_new and y_out == y_new and z_out == z_new
        return output


class MNIST_3D_Random_Rotation(object):
    """Randomly rotate the MNIST 3D image for data augmentation.
    Args:
        max_angle (float): Specifies an interval of (-max_angle, max_angle)
    for random 3D rotation.
        rotation_axes (list of str): Specifies the axes ('x', 'y', 'z') for 3D image rotation.
    """

    def __init__(self, max_angle, rotation_axes=['x', 'y', 'z']):
        assert max_angle >= 0, 'Max Angle should be non-negative'
        assert len(rotation_axes) <= 3, 'Cannot specify more than 3-dimensions for rotation'
        for axis in rotation_axes:
            assert axis in ['x', 'y', 'z'], 'Input must be one of x, y, z'
        self.rotation_range = (-max_angle, max_angle)
        self.rotation_axes = rotation_axes

    def __call__(self, data, reshape=False):
        assert len(data.shape) == 3, 'Input shape should be 3-dimensional'
        x_in, y_in, z_in = data.shape
        # Loop through rotation axes and apply rotation
        for axis in self.rotation_axes:
            random_angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
            if axis == 'x':
                data = scipy.ndimage.rotate(data, angle=random_angle, axes=(1, 2),
                                            reshape=reshape, order=1, cval=0)
            elif axis == 'y':
                data = scipy.ndimage.rotate(data, angle=random_angle, axes=(0, 2),
                                            reshape=reshape, order=1, cval=0)
            else:
                data = scipy.ndimage.rotate(data, angle=random_angle, axes=(0, 1),
                                            reshape=reshape, order=1, cval=0)
        # Convert back to numpy array
        x_out, y_out, z_out = data.shape
        # Check whether dimension of output matches (N, x_in, y_in, z_in)
        assert x_out == x_in and y_out == y_in and z_out == z_in
        return data


class MNIST_3D_Gaussian_Noise(object):
    """Add Gaussian noise sampled from a Gaussian(mean, variance) distribution to the MNIST 3D images for data augmentation.
    Args:
        mean (float or int): Specifies the mean for the gaussian distribution
        var (float or int): Specifies the variance for the gaussian distribution
    """

    def __init__(self, mean, var):
        assert var >= 0, "variance is non-negative"
        self.mean = mean
        self.var = var

    def __call__(self, data):
        assert len(data.shape) == 3 or len(data.shape) == 2, 'Input shape should be 3-dimensional or 2-dimensional'
        if len(data.shape) == 3:
            ch, row, col = data.shape
            sigma = self.var ** 0.5
            gauss = np.random.normal(self.mean, sigma, (ch, row, col))
            gauss = gauss.reshape(ch, row, col)
        else:
            row, col = data.shape
            sigma = self.var ** 0.5
            gauss = np.random.normal(self.mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
        noisy_im = data + gauss
        noisy_im = cv2.normalize(noisy_im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return noisy_im


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        pass