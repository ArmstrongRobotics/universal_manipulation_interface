import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_logs(file_path):
    data = []
    
    # 1. Parse the JSON Lines file
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("No data found in the file. Exiting.")
        return
    df = pd.DataFrame(data)

    print(df.head())  # Print first few rows for verification
    print("Setting up plot for training loss...")
    # 2. Setup the visualization
    plt.figure(figsize=(12, 6))
    
    # Plot raw loss with low alpha (transparency)
    plt.plot(df['global_step'], df['train_loss'], 
             label='Raw Loss', color='skyblue', alpha=0.4)
    
    # Plot smoothed loss (Moving Average) to see the trend
    df['smoothed_loss'] = df['train_loss'].rolling(window=10).mean()
    plt.plot(df['global_step'], df['smoothed_loss'], 
             label='Smoothed Loss (Window=10)', color='navy', linewidth=2)

    # 3. Formatting
    plt.title('Training Loss Over Time', fontsize=14)
    plt.xlabel('Global Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('training_loss_plot.png')
    wait = input("Press Enter to continue...")

# Replace 'logs.jsonl' with your actual filename
plot_training_logs("/home/armstrong/umi/data/outputs/2026.01.26/21.28.24_train_diffusion_unet_timm_umi/logs.json.txt")