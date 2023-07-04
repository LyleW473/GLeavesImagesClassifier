import torch 
import torch.nn as nn
import torch.nn.functional as F

from data_handler import DataHandler
from tools import evaluate_accuracy
from matplotlib import pyplot as plt

s_device = "cuda" if torch.cuda.is_available else "cpu" # Selected device
m_seed = 2004
torch.manual_seed(m_seed) # Seed for the model
G = torch.Generator(device = s_device) # Generator for the data handler
G.manual_seed(m_seed)

DH = DataHandler(device = s_device, generator = G, r_seed = m_seed)

train_image_names = DH.split_dataset("Train")
DH.add_DA_images(image_names_split = train_image_names, num_duplications = 10) # num_duplications = Create x duplications for each image inside the image split
val_image_names = DH.split_dataset("Val")
test_image_names = DH.split_dataset("Test")
print(f"Split Lengths| Train: {len(train_image_names[0]) * 5} | Val: {len(val_image_names[0]) * 5}, Test: {len(test_image_names[0]) * 5}")

# Note:
# - TEST_X.shape = (batch_size, colour_channels, image_ize[0], image_size[1])
# - Y.shape = (batch_size)
TEST_X, TEST_Y = DH.generate_batch(30, train_image_names)

# TEST_X.shape = [30, 3, 511, 511]
# TEST_Y.shape = [30]
print(TEST_X.shape)
print(TEST_Y.shape)

# # Check if all matrices in the batch are the same type and that labels are correct
# for matrix in TEST_X:
#     print(matrix.dtype, matrix.device, type(matrix))
# print(TEST_Y)

model = nn.Sequential(

                    # 1
                    nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3),
                    nn.BatchNorm2d(6),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3),

                    nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = 3),
                    nn.BatchNorm2d(9),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3),

                    # Size before flatten = [batch size, number of feature mapss, height of feature map, width of feature map]
                    nn.Flatten(), # Size --> [batch_size, number of feature maps * height of feature map * width of feature map]

                    nn.Linear(in_features = 9 * 55 * 55, out_features = 5445),
                    nn.BatchNorm1d(5445),
                    nn.ReLU(),

                    # nn.Dropout1d(p = 0.25, inplace = False),
                    nn.Linear(5445, 5)

                    # 2
                    # nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3),
                    # nn.BatchNorm2d(3),
                    # nn.ReLU(),
                    # nn.MaxPool2d(kernel_size = 10),

                    # nn.Flatten(),

                    # nn.Linear(in_features = 3 * 50 * 50, out_features = 1500),
                    # nn.BatchNorm1d(1500),
                    # nn.ReLU(),

                    # nn.Dropout1d(p = 0.5, inplace = False),
                    # nn.Linear(1500, 5)

                    )

model.to(device = s_device)
optimiser = torch.optim.AdamW(params = model.parameters(), lr = 0.0001)
# print(torch.cuda.memory_summary()) # Check what tensors are being stored

epochs = 1000
batch_size = 50
losses_i = []
val_losses_i = []

for i in range(epochs):
    
    # Generate inputs, labels
    Xtr, Ytr = DH.generate_batch(batch_size, train_image_names)
    
    # Forward pass
    logits = model(Xtr)

    # Cross entropy loss
    loss = F.cross_entropy(logits, Ytr)

    # Validation forward pass
    with torch.no_grad():
        Xva, Yva = DH.generate_batch(batch_size, val_image_names)
        v_logits = model(Xva)
        v_loss = F.cross_entropy(v_logits, Yva)
        val_losses_i.append(v_loss.log10().item())

    # Zero-grad
    optimiser.zero_grad()
    
    # Back-propagation
    loss.backward()

    # Update model parameters
    optimiser.step()


    # --------------------------------------------------
    # Stats tracking
    
    losses_i.append(loss.log10().item())

    if i == 0 or (i + 1) % 10 == 0:
        print(f"Epoch: {i + 1} | TrainLoss: {loss.item()} | ValLoss: {v_loss.item()}")

print("------------------------------------")
print("Train accuracy")
evaluate_accuracy(steps = 500, batch_size = 40, generate_batch_f = DH.generate_batch, model = model, image_names_split = train_image_names)
print("------------------------------------")
print("Val accuracy")
evaluate_accuracy(steps = 500, batch_size = 40, generate_batch_f = DH.generate_batch, model = model, image_names_split = val_image_names)

losses_i = torch.tensor(losses_i).view(-1, 10).mean(1)
val_losses_i = torch.tensor(val_losses_i).view(-1, 10).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(epochs / 10))], losses_i, label = "Train")
ax.plot([i for i in range(int(epochs / 10))], val_losses_i, label = "Validation")
ax.legend()

# plt.plot(losses_i)
plt.show()



# ------------------------------------------------------------------------
# Tests:

# (batch_size = 50, epochs = 1000, lr = 0.0001)
# Notes: 
# - Image inputs were not normalised (pixel values between 0 and 1) and standardised (mean 0, std 1) for this test
# - Validation loss for set-up 2 increases more (set-up 1 stays at roughly the same validation loss for the entire training)

# ------------------------------------
# 1 [without dropout layer]

# Epoch: 1000 | TrainLoss: 8.832030289340764e-05 | ValLoss: 1.7492263317108154
# Train accuracy
# Correct predictions: 20000 / 20000 | Accuracy(%): 100.0
# Val accuracy
# Correct predictions: 9436 / 20000 | Accuracy(%): 47.18

# ------------------------------------
# 1 [with dropout layer (p = 0.25)]

# Epoch: 1000 | TrainLoss: 0.41923847794532776 | ValLoss: 1.90195894241333
# Train accuracy
# Correct predictions: 15954 / 20000 | Accuracy(%): 79.77
# Val accuracy
# Correct predictions: 8030 / 20000 | Accuracy(%): 40.150000000000006

# ------------------------------------
# 2

# Epoch: 1000 | TrainLoss: 0.8374345898628235 | ValLoss: 2.1230554580688477
# Train accuracy
# Correct predictions: 11922 / 20000 | Accuracy(%): 59.61
# Val accuracy
# Correct predictions: 5834 / 20000 | Accuracy(%): 29.17

# ------------------------------------------------------------------------
# Standardised + normalised inputs (+ fixed bug with tensor.view())

# ------------------------------------
# 1 [without dropout layer]

# Epoch: 1000 | TrainLoss: 6.035612750565633e-05 | ValLoss: 2.0396666526794434
# Train accuracy
# Correct predictions: 20000 / 20000 | Accuracy(%): 100.0
# Val accuracy
# Correct predictions: 9867 / 20000 | Accuracy(%): 49.335

# ------------------------------------
# 1 [with dropout layer (p = 0.25)]

# Epoch: 1000 | TrainLoss: 0.4191969633102417 | ValLoss: 2.349715232849121
# Train accuracy
# Correct predictions: 15954 / 20000 | Accuracy(%): 79.77
# Val accuracy
# Correct predictions: 8019 / 20000 | Accuracy(%): 40.095


# ------------------------------------------------------------------------
# With images created from data augmentation (normalised + standardised inputs)

# num_duplications = 10

# ------------------------------------
# 1 [without dropout layer]

# Epoch: 1000 | TrainLoss: 0.0008139178389683366 | ValLoss: 1.6594018936157227
# Train accuracy
# Correct predictions: 20000 / 20000 | Accuracy(%): 100.0
# Val accuracy
# Correct predictions: 8314 / 20000 | Accuracy(%): 41.57

# ------------------------------------
# 1 [with dropout layer (p = 0.25)]

# Epoch: 1000 | TrainLoss: 0.42324337363243103 | ValLoss: 2.2393290996551514
# Train accuracy
# Correct predictions: 15939 / 20000 | Accuracy(%): 79.69500000000001
# Val accuracy
# Correct predictions: 7491 / 20000 | Accuracy(%): 37.455