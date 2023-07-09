import torch 
import torch.nn as nn
import torch.nn.functional as F

from data_handler import DataHandler
from tools import evaluate_accuracy, find_accuracy
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

                    nn.Dropout1d(p = 0.25, inplace = False),
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
train_accuracy_i = []
val_accuracy_i = []

for i in range(epochs):
    
    # Generate inputs, labels
    Xtr, Ytr = DH.generate_batch(batch_size, train_image_names)
    
    # Forward pass
    logits = model(Xtr)

    # Cross entropy loss
    loss = F.cross_entropy(logits, Ytr)

    with torch.no_grad():
        
        # Note: Must set to eval mode as BatchNorm layers and Dropout layers behave differently during training and evaluation
        # BatchNorm layers - stops updating the moving averages in BatchNorm layers and uses running statistics instead of per-batch statistics
        # Dropout layers - Dropout layers are de-activated during evaluation
        model.eval()

        # Train accuracy on current batch
        preds = F.softmax(logits, dim = 1)
        train_accuracy = find_accuracy(predictions = preds, targets = Ytr, batch_size = batch_size)
        train_accuracy_i.append(find_accuracy(predictions = preds, targets = Ytr, batch_size = batch_size))

        # Validation forward pass
        Xva, Yva = DH.generate_batch(batch_size, val_image_names)
        v_logits = model(Xva)
        v_loss = F.cross_entropy(v_logits, Yva)
        val_losses_i.append(v_loss.log10().item())

        # Validation accuracy on current batch
        v_preds = F.softmax(v_logits, dim = 1) # Softmax to find probability distribution
        val_accuracy = find_accuracy(predictions = v_preds, targets = Yva, batch_size = batch_size)
        val_accuracy_i.append(val_accuracy)

        model.train()
    
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
        print(f"Epoch: {i + 1} | TrainLoss: {loss.item()} | ValLoss: {v_loss.item()} | TrainAcc: {train_accuracy} | ValAcc: {val_accuracy}")

# Set model to evaluation mode (For dropout + batch norm layers)
model.eval()

# Plotting losses during training
print("-----------------------------------------------------------------")
print("Losses during training")

A = 20
losses_i = torch.tensor(losses_i).view(-1, A).mean(1)
val_losses_i = torch.tensor(val_losses_i).view(-1, A).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(epochs / A))], losses_i, label = "Train")
ax.plot([i for i in range(int(epochs / A))], val_losses_i, label = "Validation")
ax.legend()

plt.show()

# Plotting accuracies during training
print("-----------------------------------------------------------------")
print("Accuracy during training")

B = 20
train_accuracy_i = torch.tensor(train_accuracy_i).view(-1, B).mean(1)
val_accuracy_i = torch.tensor(val_accuracy_i).view(-1, B).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(epochs / B))], train_accuracy_i, label = "Train")
ax.plot([i for i in range(int(epochs / B))], val_accuracy_i, label = "Validation")
ax.legend()
plt.show()

# Plotting accuracies after training
print("-----------------------------------------------------------------")
print("Accuracy after training")
accuracy_steps = 1000
accuracy_bs = 20
C = 20
CHECK_INTERVAL = 50
train_accuracies = evaluate_accuracy(
                                    steps = accuracy_steps, 
                                    batch_size = accuracy_bs, 
                                    generate_batch_f = DH.generate_batch, 
                                    model = model, 
                                    image_names_split = train_image_names, 
                                    split_name = "Train",
                                    check_interval = CHECK_INTERVAL
                                    )
                                    
val_accuracies = evaluate_accuracy(
                                    steps = accuracy_steps, 
                                    batch_size = accuracy_bs, 
                                    generate_batch_f = DH.generate_batch, 
                                    model = model, 
                                    image_names_split = val_image_names, 
                                    split_name = "Val",
                                    check_interval = CHECK_INTERVAL
                                    )

train_accuracies = torch.tensor(train_accuracies).view(-1, C).mean(1)
val_accuracies = torch.tensor(val_accuracies).view(-1, C).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(accuracy_steps / C))], train_accuracies, label = "Train")
ax.plot([i for i in range(int(accuracy_steps / C))], val_accuracies, label = "Validation")
ax.legend()

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

# ------------------------------------
# (Alternating between train and evaluation mode when finding accuracy and evaluation mode after training)

# 1 [without dropout layer]

# Epoch: 1000 | TrainLoss: 0.0008298794273287058 | ValLoss: 1.5287092924118042 | TrainAcc: 100.0 | ValAcc: 48.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 8873 / 20000 | ValAccuracy(%): 44.365

# ------------------------------------
# 1 [with dropout layer (p = 0.25)]

# Epoch: 1000 | TrainLoss: 0.40037113428115845 | ValLoss: 1.6299127340316772 | TrainAcc: 82.0 | ValAcc: 44.0
# Correct predictions: 19932 / 20000 | TrainAccuracy(%): 99.66000000000001
# Correct predictions: 8799 / 20000 | ValAccuracy(%): 43.995