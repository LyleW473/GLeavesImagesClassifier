from torch import no_grad as torch_no_grad
from torch.nn.functional import softmax as F_softmax
from torch import max as torch_max
from torch import sum as torch_sum

def evaluate_accuracy(steps, batch_size, generate_batch_f, model, image_names_split, split_name, check_interval):
    
    num_correct = 0
    num_tested = 0
    accuracies = []
    
    with torch_no_grad():
        
        for i in range(steps):
            Xa, Ya = generate_batch_f(batch_size, image_names_split)

            logits = model(Xa)
            preds = F_softmax(logits, dim = 1) # Softmax for probability distribution

            num_correct += count_correct_preds(predictions = preds, targets = Ya)
            num_tested += batch_size
            accuracies.append((num_correct / num_tested) * 100)
            
            if (i + 1) % check_interval == 0:
                print(f"Correct predictions: {num_correct} / {num_tested} | {split_name}Accuracy(%): {(num_correct / num_tested) * 100}")
    
    return accuracies

def count_correct_preds(predictions, targets):
    # Find the predictions of the model
    _, output = torch_max(predictions, dim = 1) 

    # Return the number of correct predictions
    return torch_sum(output == targets).item()

def find_accuracy(predictions, targets, batch_size):
    return (count_correct_preds(predictions, targets) / batch_size) * 100