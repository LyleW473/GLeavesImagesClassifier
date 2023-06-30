import torch 
import torch.nn as nn
import torch.nn.functional as F

from data_handler import DataHandler
from tools import evaluate_accuracy


s_device = "cuda" if torch.cuda.is_available else "cpu" # Selected device
m_seed = 2004
torch.manual_seed(m_seed) # Seed for the model
G = torch.Generator(device = s_device) # Generator for the data handler
G.manual_seed(m_seed)

DH = DataHandler(device = s_device, generator = G, r_seed = m_seed)

train_image_names = DH.split_dataset("Train")
val_image_names = DH.split_dataset("Val")
test_image_names = DH.split_dataset("Test")

# Note:
# - TEST_X.shape = (batch_size, colour_channels, image_ize[0], image_size[1])
# - Y.shape = (batch_size)
TEST_X, TEST_Y = DH.generate_batch(30, train_image_names)

# TEST_X.shape = [30, 3, 511, 511]
# TEST_Y.shape = [30]
print(TEST_X.shape)
print(TEST_Y.shape)


model = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3),
                    nn.BatchNorm2d(6),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3),

                    nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = 3),
                    nn.BatchNorm2d(9),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3),

                    # Size before flatten = [batch size, number of feature mapss, height of feature map, width of feature map]
                    nn.Flatten(), # Size --> [batch_size, number of feature mapss * height of feature map * width of feature map]

                    nn.Linear(in_features = 9 * 55 * 55, out_features = 5445),
                    nn.BatchNorm1d(5445),
                    nn.ReLU(),

                    nn.Linear(5445, 5)
                    
                    )

model.to(device = s_device)
optimiser = torch.optim.AdamW(params = model.parameters(), lr = 5e-4)

epochs = 1000
batch_size = 20

for i in range(epochs):
    
    # Generate inputs, labels
    X, Y = DH.generate_batch(batch_size, train_image_names)
    
    # Forward pass
    logits = model(X)
    
    # Cross entropy loss
    loss = F.cross_entropy(logits, Y)

    # Zero-grad
    optimiser.zero_grad()
    
    # Back-propagation
    loss.backward()

    # Update model parameters
    optimiser.step()
    
    if i == 0 or (i + 1) % 50 == 0:
        print(f"Epoch: {i + 1} | Loss: {loss.item()}")

print("------------------------------------")
print("Train accuracy")
evaluate_accuracy(steps = 500, batch_size = 40, generate_batch_f = DH.generate_batch, model = model, image_names_split = train_image_names)
print("------------------------------------")
print("Val accuracy")
evaluate_accuracy(steps = 500, batch_size = 40, generate_batch_f = DH.generate_batch, model = model, image_names_split = val_image_names)