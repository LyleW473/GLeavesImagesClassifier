from torch import no_grad as torch_no_grad
from torch.nn.functional import cross_entropy as F_cross_entropy
from torch import max as torch_max
from torch import sum as torch_sum

def evaluate_accuracy(steps, batch_size, generate_batch_f, model, image_names_split):
    
    test_losses_i = []
    num_correct = 0
    num_tested = 0

    with torch_no_grad():
        
        for i in range(steps):
            Xa, Ya = generate_batch_f(batch_size, image_names_split)

            logits = model(Xa)
            loss = F_cross_entropy(logits, Ya)

            num_correct += count_correct_preds(predictions = logits, targets = Ya)
            num_tested += batch_size
            test_losses_i.append(loss.log10().item())
            
            if (i + 1) % 50 == 0:
                print(f"Correct predictions: {num_correct} / {num_tested} | Accuracy(%): {(num_correct / num_tested) * 100}")


def count_correct_preds(predictions, targets):
    # Find the predictions of the model
    _, output = torch_max(predictions, dim = 1) 

    # Return the number of correct predictions
    return torch_sum(output == targets).item()