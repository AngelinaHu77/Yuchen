import matplotlib.pyplot as plt
# Function to extract metric values based on the provided format
def extract_metric_values_simplified(data, metric_name_prefix):
    metric_values = {}
    metrics = ['Accuracy', 'MCC', 'F1 Score', 'losses']

    # Split the data by newline and then by colon to get the metric values
    for line in data.split("\n"):
        for metric in metrics:
            if line.startswith(f"{metric_name_prefix} {metric}"):
                value = float(line.split(":")[1].strip())
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(value)
    return metric_values


# Load the data from the files
with open("./FinBert_model_train_results.txt", 'r') as f:
    train_data_content = f.read()

with open("./FinBert_model_evaluation_results.txt", 'r') as f:
    eval_data_content = f.read()

# Extract metrics for training and validation data
train_metrics_simplified = extract_metric_values_simplified(train_data_content, "train")
eval_metrics_simplified = extract_metric_values_simplified(eval_data_content, "Validation")
# Function to plot metrics and save the plots in high resolution
def plot_metric_and_save(train_values, eval_values, metric_name, save_path, dpi=300):
    epochs = range(1, len(train_values) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, 'b', label=f'Training {metric_name}')
    plt.plot(epochs, eval_values, 'r', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()

# Save the four metrics as high-resolution images
for metric in ['Accuracy', 'MCC', 'F1 Score', 'losses']:
    save_path = f"./{metric}_plot.png"
    plot_metric_and_save(train_metrics_simplified[metric], eval_metrics_simplified[metric], metric, save_path)


