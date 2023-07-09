from os import listdir as os_listdir
from torch import multinomial as torch_multinomial
from torch import ones as torch_ones
from random import shuffle as random_shuffle
from random import seed as random_seed
from cv2 import imread as cv2_imread
from torch import from_numpy as torch_from_numpy
from torch import stack as torch_stack
from torch import min as torch_min
from torch import max as torch_max
from torch import empty as torch_empty
import torchvision.transforms as transforms
from torch import allclose as torch_allclose
from torch import set_printoptions as torch_set_printoptions
import cv2
import numpy as np

# Note: To save memory, I am storing the names of every image in their respective type list for each split, and then when generating a batch, will convert the images into matrices

class DataHandler:

    def __init__(self, device, generator, r_seed):
        
        # Re-producibility
        self.generator = generator # Generating batches
        random_seed(r_seed) # Shuffling dataset
        
        # Device used for computation (CPU / GPU)
        self.device = device

        # Load data into their lists using os.listdir
        self.data = []
        self.leaf_types = os_listdir("Dataset/Images")

        # Shuffle the data inside the lists
        for leaf_type in self.leaf_types:
            images = os_listdir(f"Dataset/Images/{leaf_type}")
            random_shuffle(images)
            self.data.append(images)

    def split_dataset(self, split):

        # Splits images into train/val/test splits, each leaf type will have the same split distribution 
        # i.e. Each split should have an equal amount of images of each type
        # image_names = List of 5 leaf types, each containing the split_distribution amount of images for that directory
        # i.e. Training split = 80%, each leaf type list will contain 80% of all images for that leaf type

        n_imgs_per_type = 100 # Number of images for each type
        split_distribution = {
                            "Train": int(0.6 * n_imgs_per_type),
                            "Val": int(0.2 * n_imgs_per_type),
                            "Test": int(0.2 * n_imgs_per_type) 
                            }

        if split == "Train":
            image_names = [self.data[i][0:split_distribution["Train"]] for i in range(len(self.data))]

        elif split == "Val":
            image_names = [self.data[i][split_distribution["Train"]:(split_distribution["Train"] + split_distribution["Val"])] for i in range(len(self.data))]

        elif split == "Test":
            image_names = [self.data[i][(split_distribution["Train"] + split_distribution["Val"]):] for i in range(len(self.data))]

        return image_names
    
    def generate_batch(self, batch_size, image_names_split):

        # Choose batch_size types
        u_distrib = torch_ones(5, device = self.device) / 5 # Single vector with uniform distribution
        type_idxs = torch_multinomial(input = u_distrib, num_samples = batch_size, replacement = True, generator = self.generator)
        
        # Choose batch_size images from the split
        n_split_images = len(image_names_split[0]) # Number of images in this split for each leaf type
        u_distrib = torch_ones(n_split_images, device = self.device) / n_split_images # Single vector with uniform distribution
        img_idxs = torch_multinomial(input = u_distrib, num_samples = batch_size, replacement = True, generator = self.generator)

        # Note: 
        # - type_idxs will contain all the types selected for this batch
        # - img_idxs will contain the image selected in each leaf type directory for each leaf type selected

        # Returns matrices of each image and the labels for each image
        return self.get_matrices(t_idxs = type_idxs, i_idxs = img_idxs, image_names_split = image_names_split), type_idxs

    def get_matrices(self, t_idxs, i_idxs, image_names_split):

        matrices = [] 

        for type_idx, image_idx in zip(t_idxs, i_idxs):
            # Type of leaf
            l_type = self.leaf_types[type_idx]
            
            # Name of the image in the leaf type directory
            image_name = image_names_split[type_idx][image_idx]

            # Name of the leaf image
            if type(image_name) == str:
                # Convert the image into a matrix and add it to the list
                matrices.append(self.get_image_matrix(image_name = image_name, type_name = l_type))
                # print("GENERATE_MATRIX", type_idx, image_idx)
            else:
                # "image_name" will already be a PyTorch tensor of the images produced through data augmentation
                matrices.append(image_name.to(device = self.device))
                # print("TENSOR", type_idx, image_idx)

        # Convert from Python list to PyTorch tensor
        matrices = torch_stack(matrices, dim = 0)

        return matrices

    def get_image_matrix(self, image_name, type_name):
        
        bgr_image = cv2_imread(f"Dataset/Images/{type_name}/{image_name}")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        matrix_PT = torch_from_numpy(np.transpose(rgb_image, (2, 0, 1))).float().to(device = self.device)

        # Normalise values between 0 and 1 (Transforms.ToTensor already does this)
        minimums = torch_empty(3, device = self.device)
        maximums = torch_empty(3, device = self.device)
        for channel in range(3):
            minimums[channel] = torch_min(matrix_PT[channel])
            maximums[channel] = torch_max(matrix_PT[channel])

        matrix_PT -= minimums[:, None, None]
        matrix_PT /= (maximums - minimums)[:, None, None]
            
        # Standardize image to make pixel values mean 0, std 1
        # Mean + STD across the 3 RGB channels
        mean = matrix_PT.mean(dim = (1, 2))
        std = matrix_PT.std(dim = (1, 2))
        # Takes away the mean / std from each RGB channel
        matrix_PT -= mean[:, None, None] 
        matrix_PT /= std[:, None, None]

        # print("STD", matrix_PT.std(dim = (1, 2)))
        # print("MEAN", matrix_PT.mean(dim = (1, 2)))

        return matrix_PT
    
    def add_DA_images(self, image_names_split, num_duplications): # Data augmentation
        # Note: Adds more images into each list of the image names for each leaf type
        # - Should only add images to the training split 
        
        leaf_type_paths = [f"Dataset/Images/{l_type}" for l_type in self.leaf_types]
        print(self.leaf_types)
        print(leaf_type_paths)

        transformation = transforms.Compose(
                                        [
                                        transforms.ToTensor(), # Converts nd array / PIL image to PyTorch tensor and makes pixel values between 0 and 1

                                        # Shifts the image left / right / up / down
                                        transforms.Resize(size = (511 + (73 * 2), 511 + (73 * 2))), # 73 = a factor of 511
                                        transforms.RandomCrop(size = (511, 511)),

                                        # Rotates image between (-x and x degrees)
                                        transforms.RandomRotation(degrees = 30),

                                        # Standardises image pixels to be of mean 0, std 1 [transforms.Normalize but takes finds the mean and std of the image inputted]
                                        DynamicNormalize() 
                                        ]
                                        )

        
        for lt_num, lt_list in enumerate(image_names_split):

            print("Original length", len(lt_list))

            # Make a copy so that we only add existing images (Not the added augmented images)
            # Note: Required so that the tensors created from data-augmentation won't be selected for the transformation
            original_lt_list = lt_list.copy() 

            for image_name in original_lt_list:
                
                # print("Name", f"{leaf_type_paths[lt_num]}/{image_name}")
                
                for _ in range(num_duplications): # Repeats the data augmentation "num_duplication" times for each image in the split

                    # Retrieve np array from the image and convert from BGR to RGB
                    bgr_image = cv2_imread(f"{leaf_type_paths[lt_num]}/{image_name}")
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                    # Input the np array of the image and apply the transformations
                    # Note: DA.tensor.dtype = torch.float32, DA_tensor.shape = [3, 511, 511]
                    DA_tensor = (transformation(rgb_image)) # Returns the augmented image as a PyTorch tensor

                    # Add the tensor to the list of all the possible leaf "images" to pick
                    # Note: Since these aren't image names, they will not use the "get_matrices" method, but will simply be added to the batch when "generate_batch" is called
                    lt_list.append(DA_tensor)

                    # # Visualise image from the augmented tensor
                    # self.tensor_to_image(tensor = DA_tensor, original_image_path = f"{leaf_type_paths[lt_num]}/{image_name}")

            print("New length", len(lt_list))
            
            # Shuffle the list after adding images
            random_shuffle(lt_list)

    def tensor_to_image(self, tensor, original_image_path): 
        # Used to visualise the images from data augmentation

        original = cv2_imread(original_image_path)
        reverse_to_np = tensor.permute(1, 2, 0) * 255 # (Channels, Height, Width) --->  (Height, Width, Channels) and scale pixel values to be from [0, 255] instead of [0, 1]
        reverse_to_np = reverse_to_np.cpu().numpy().astype(np.uint8) # Convert back to numpy array 
        reverse_to_np = cv2.cvtColor(reverse_to_np, cv2.COLOR_RGB2BGR) # Convert from RGB array to BGR array so that it can be visualised properly using CV2

        print("DTYPE", original.dtype, reverse_to_np.dtype)
        print("Original", original.shape)
        print(original[250][250])
        print("After", reverse_to_np.shape)
        print(original[250][250])
        print((original == reverse_to_np).all())

        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.imshow('Original', original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.namedWindow('After', cv2.WINDOW_NORMAL)
        cv2.imshow('After', reverse_to_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class DynamicNormalize(transforms.Normalize): # Used so that the transformations don't need to be declared every time for each image (due to needing to calculate mean + std)
    def __init__(self, mean = None, std = None):
        super().__init__(mean, std)
        
    def __call__(self, tensor):
        # Takes in a 3-D tensor input, finds the mean and std of the 1st and 2nd dimensions
        mean = tensor.mean(dim = (1, 2))
        std = tensor.std(dim = (1, 2))
        
        self.mean = mean.tolist()
        self.std = std.tolist()
        
        return super().__call__(tensor)