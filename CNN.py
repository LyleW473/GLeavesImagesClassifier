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

                    nn.Dropout1d(p = 0.1, inplace = False),
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
# Note: Abbreviations: os = over steps, pt = per type
accuracy_steps = 1000
accuracy_bs = 20
C = 20
CHECK_INTERVAL = 50
train_accuracies_os, tra_accuracies_pt = evaluate_accuracy(
                                                        steps = accuracy_steps, 
                                                        batch_size = accuracy_bs, 
                                                        generate_batch_f = DH.generate_batch, 
                                                        model = model, 
                                                        image_names_split = train_image_names, 
                                                        split_name = "Train",
                                                        check_interval = CHECK_INTERVAL
                                                        )

tra_correct = [tra_accuracies_pt[i][0] for i in range(5)] # Number of correct predictions for each type
tra_generated = [tra_accuracies_pt[i][1] for i in range(5)] # Number of examples generated for each type
tra_accuracies = [(n_correct / n_generated) * 100 for n_correct, n_generated in zip(tra_correct, tra_generated)] # Accuracy

                                    
val_accuracies_os, val_accuracies_pt = evaluate_accuracy(
                                                    steps = accuracy_steps, 
                                                    batch_size = accuracy_bs, 
                                                    generate_batch_f = DH.generate_batch, 
                                                    model = model, 
                                                    image_names_split = val_image_names, 
                                                    split_name = "Val",
                                                    check_interval = CHECK_INTERVAL
                                                    )

val_correct = [val_accuracies_pt[i][0] for i in range(5)] # Number of correct predictions for each type
val_generated = [val_accuracies_pt[i][1] for i in range(5)] # Number of examples generated for each type
val_accuracies = [(n_correct / n_generated) * 100 for n_correct, n_generated in zip(val_correct, val_generated)] # Accuracy

print("-----------------------------------------------------------------")
print(f"LeafTypes: {DH.leaf_types}")

print(f"TrainCorrect: {tra_correct}")
print(f"TrainGenerated:{tra_generated}")
print(f"TrainAccuracies: {tra_accuracies}")

print(f"ValCorrect: {val_correct}")
print(f"ValGenerated:{val_generated}")
print(f"ValAccuracies: {val_accuracies}")

train_accuracies_os = torch.tensor(train_accuracies_os).view(-1, C).mean(1)
val_accuracies_os = torch.tensor(val_accuracies_os).view(-1, C).mean(1)

fig, ax = plt.subplots()
ax.plot([i for i in range(int(accuracy_steps / C))], train_accuracies_os, label = "Train")
ax.plot([i for i in range(int(accuracy_steps / C))], val_accuracies_os, label = "Validation")
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

# ------------------------------------------------------------------------
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

# + (Shuffled each list after adding images from data augmentation)

# ------------------------------------
# 1 [without dropout layer]

# Epoch: 1000 | TrainLoss: 0.000957684766035527 | ValLoss: 1.6948890686035156 | TrainAcc: 100.0 | ValAcc: 57.99999999999999
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9828 / 20000 | ValAccuracy(%): 49.14

# Epoch: 3000 | TrainLoss: 0.0002263062197016552 | ValLoss: 2.87349271774292 | TrainAcc: 100.0 | ValAcc: 34.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9272 / 20000 | ValAccuracy(%): 46.36


# ------------------------------------
# 1 [with dropout layer (p = 0.25)]

# Epoch: 1000 | TrainLoss: 0.3878507912158966 | ValLoss: 1.4953268766403198 | TrainAcc: 82.0 | ValAcc: 54.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9634 / 20000 | ValAccuracy(%): 48.17

# Epoch: 3000 | TrainLoss: 0.3862666189670563 | ValLoss: 2.425055503845215 | TrainAcc: 76.0 | ValAcc: 34.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9316 / 20000 | ValAccuracy(%): 46.58


# ------------------------------------
# 1 [with dropout layer (p = 0.1)]

# Epoch: 1000 | TrainLoss: 0.09772449731826782 | ValLoss: 1.541168212890625 | TrainAcc: 94.0 | ValAcc: 62.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9822 / 20000 | ValAccuracy(%): 49.11

# Epoch: 3000 | TrainLoss: 0.22533057630062103 | ValLoss: 2.4722952842712402 | TrainAcc: 86.0 | ValAcc: 34.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9274 / 20000 | ValAccuracy(%): 46.3


# ------------------------------------------------------------------------
# Changed split distribution to: [0.6, 0.2, 0.2] (num_duplications = 10, batch_size = 50)

# 1 [with dropout layer (p = 0.1)]

# Epoch: 1000 | TrainLoss: 0.09960095584392548 | ValLoss: 1.7756178379058838 | TrainAcc: 94.0 | ValAcc: 50.0
# Correct predictions: 19977 / 20000 | TrainAccuracy(%): 99.885
# Correct predictions: 10610 / 20000 | ValAccuracy(%): 53.05

# TrainCorrect: [3912, 4008, 4068, 3964, 4025]
# TrainGenerated:[3912, 4008, 4073, 3982, 4025]
# TrainAccuracies: [100.0, 100.0, 99.87724036336853, 99.54796584630839, 100.0]

# ValCorrect: [3420, 1606, 2609, 1535, 1440]
# ValGenerated:[4003, 4032, 3995, 3917, 4053]
# ValAccuracies: [85.43592305770672, 39.8313492063492, 65.30663329161452, 39.18815419964258, 35.529237601776465]

# ------------------------------------------------------------------------
# Changed split distribution to: [0.7, 0.15, 0.15] (num_duplications = 10, batch_size = 50)

# 1 [with dropout layer (p = 0.1)]

# Epoch: 1000 | TrainLoss: 0.09772907197475433 | ValLoss: 2.0164947509765625 | TrainAcc: 94.0 | ValAcc: 48.0
# Correct predictions: 20000 / 20000 | TrainAccuracy(%): 100.0
# Correct predictions: 9019 / 20000 | ValAccuracy(%): 45.095

# TrainCorrect: [3912, 4008, 4073, 3982, 4025]
# TrainGenerated:[3912, 4008, 4073, 3982, 4025]
# TrainAccuracies: [100.0, 100.0, 100.0, 100.0, 100.0]

# ValCorrect: [2078, 1072, 2693, 1840, 1336]
# ValGenerated:[4003, 4032, 3995, 3917, 4053]
# ValAccuracies: [51.91106669997502, 26.58730158730159, 67.40926157697122, 46.97472555527189, 32.96323710831483]