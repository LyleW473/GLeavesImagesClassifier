from torch import no_grad as torch_no_grad
from torch.nn.functional import softmax as F_softmax
from torch import max as torch_max
from torch import sum as torch_sum

def evaluate_accuracy(steps, batch_size, generate_batch_f, model, image_names_split, split_name, check_interval):
    
    num_correct = 0
    num_tested = 0
    accuracies_over_steps = []
    accuracies_per_type = {i:[0, 0] for i in range(5)} # i:[number of times correct predicted, number of times appeared in total over every batch]

    with torch_no_grad():
        
        for i in range(steps):
            Xa, Ya = generate_batch_f(batch_size, image_names_split)
            
            logits = model(Xa)
            preds = F_softmax(logits, dim = 1) # Softmax for probability distribution

            # Add the number of appearances of each type in this batch
            all_nums, all_appearances = Ya.unique(return_counts = True)
            for type_num, n_appearances in zip(all_nums, all_appearances):
                accuracies_per_type[type_num.item()][1] += n_appearances.item()

            # Add the number of times the model correctly predicted each type
            type_nums, batch_num_correct = count_correct_preds(predictions = preds, targets = Ya)
            for type_n, n_correct in zip(type_nums, batch_num_correct):
                accuracies_per_type[type_n.item()][0] += n_correct.item()
                num_correct += n_correct.item() # Running num correct over all steps
            
            num_tested += batch_size
            accuracies_over_steps.append((num_correct / num_tested) * 100)
            
            if (i + 1) % check_interval == 0:
                print(f"Correct predictions: {num_correct} / {num_tested} | {split_name}Accuracy(%): {(num_correct / num_tested) * 100}")
    
    return accuracies_over_steps, accuracies_per_type

def count_correct_preds(predictions, targets):
    # Find the predictions of the model
    _, output = torch_max(predictions, dim = 1) 
    
    # Find the number of correct predictions by the model
    matching_vals = output[output == targets] # The values of the items in the two tensors at indices where they were the same
    type_nums, counts = matching_vals.unique(return_counts = True) # Returns the type number and the number of times they were correctly guessed
    
    # Return the type numbers of each correct prediction and the number of times they were correctly predicted
    return type_nums, counts

def find_accuracy(predictions, targets, batch_size):
    _ , counts = count_correct_preds(predictions, targets)
    return (torch_sum(counts).item() / batch_size) * 100