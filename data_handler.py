from os import listdir as os_listdir
from torch import manual_seed as torch_manual_seed
from torch import multinomial as torch_multinomial
from torch import ones as torch_ones
from random import shuffle as random_shuffle
from cv2 import imread as cv2_imread
from torch import from_numpy as torch_from_numpy
from torch import stack as torch_stack
from torch.cuda import is_available as torch_cuda_is_available

# Note: To save memory, I am storing the names of every image in their respective type list for each split, and then when generating a batch, will convert the images into matrices

class DataHandler:

    def __init__(self, device = "cuda" if torch_cuda_is_available else "cpu"):
        
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
                            "Train": int(0.8 * n_imgs_per_type),
                            "Val": int(0.1 * n_imgs_per_type),
                            "Test": int(0.1 * n_imgs_per_type) 
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
        type_idxs = torch_multinomial(input = u_distrib, num_samples = batch_size, replacement = True)

        # Choose batch_size images from the split
        n_split_images = len(image_names_split[0]) # Number of images in this split for each leaf type
        u_distrib = torch_ones(n_split_images, device = self.device) / n_split_images # Single vector with uniform distribution
        img_idxs = torch_multinomial(input = u_distrib, num_samples = batch_size, replacement = True)

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

            # Conver the image into a matrix and add it to the list
            matrices.append(self.get_image_matrix(image_name = image_name, type_name = l_type))

        # Convert from Python list to PyTorch tensor
        matrices = torch_stack(matrices, dim = 0)

        return matrices

    def get_image_matrix(self, image_name, type_name):

        # # Read image as numpy array
        # img_np = cv2_imread(f"Dataset/Images/{type_name}/{image_name}")

        # # Convert numpy array to torch, with dtype as torch.float32 (for matrix multiplication with model weights)
        # matrix = torch_from_numpy(img_np).view(3, 511, 511).float() # Convert shape from (511, 511, 3) to (3, 511, 511) [511 being the width and height of the image]

        # del img_np

        # return matrix

        # No intermediaries: 
        return torch_from_numpy(cv2_imread(f"Dataset/Images/{type_name}/{image_name}")).view(3, 511, 511).float()
    

# Seed for reproducibility
torch_manual_seed(2004)


DH = DataHandler()

train_image_names = DH.split_dataset("Train")
val_image_names = DH.split_dataset("Val")
test_image_names = DH.split_dataset("Test")

# Note:
# - X.shape = (batch_size, colour_channels, image_ize[0], image_size[1])
# - Y.shape = (batch_size)
X, Y = DH.generate_batch(30, train_image_names)

print(X.shape)
print(Y.shape)