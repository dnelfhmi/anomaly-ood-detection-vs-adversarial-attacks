import time
import torch
import model


def return_predictions(model, inputs, outputs):
    logits = model(inputs)
    correct = logits.max(dim=1)[1] == outputs
    return logits, torch.sum(correct.detach().type(torch.float64)).item(), torch.mean(correct.detach().type(torch.float64)).item(), correct.shape

       
def test(seed=0, batch_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Start measuring time
    start_time = time.perf_counter()

    # Set random seed for reproducability
    torch.manual_seed(seed)

    # Load dataset
    valid_data = torch.load("dataset.pt")

    # Load weights
    weights_base = torch.zeros([9, 1, 3, 3])
    weights = torch.load('model_chkpt.pt', map_location=torch.device(device))
    
    # Copy the model for validation  
    valid_model_2 = model.Averager(1, 10, 0.125)
    valid_model_2.model.load_state_dict(weights)
    valid_model_2.model.to(dtype)
    valid_model_2.model.to(device)
    
    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    # Train and validate
    print("\n Count \t Total \t Acc. \t Percentage")
    val_time = 0.0

    # Flush CUDA pipeline for more accurate time measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    valid_count, total_count = 0, 0
    # Validation
    for i in range(0, len(valid_data), batch_size):
        (data, targets) = valid_data[i : i + batch_size]

        valid_model_2.train(False)        
        
        _, count_acc, _, shape_acc = return_predictions(valid_model_2, data, targets)
        valid_count += count_acc
        total_count += shape_acc[0]
        
        print(f"{i} \t {len(valid_data)} \t {(count_acc / shape_acc[0]):1.3f} \t {100*i / len(valid_data)}")        
        


    # Accuracy
    valid_acc = valid_count / total_count
    
    valid_time = time.perf_counter() - start_time

    print(f"Validation Time: {valid_time:3.2f} Validation Accuracy: {valid_acc:1.4f}")

    # Attacks: PLEASE FILL IN CODE BELOW HERE
    successful_attack_count, failed_predictions, total_count = 0, 0, 0
    # Modify as appropriate

    print(f"Total successful attacks [perc]: {100*successful_attack_count/total_count:2.4f}, Failed Predictions [perc]:{100*failed_predictions/total_count:2.4f}")    


def print_info():
    print("PyTorch:", torch.__version__)
    


if __name__ == "__main__":
    print_info()

    test()
