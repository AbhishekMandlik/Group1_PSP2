import kagglehub
import shutil
import os

# Download to default location
path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")

# Define your custom directory
custom_dir = "./text_summary/data"
os.makedirs(custom_dir, exist_ok=True)

# Move downloaded files to custom directory
for filename in os.listdir(path):
    shutil.move(os.path.join(path, filename), os.path.join(custom_dir, filename))

print("Dataset moved to:", custom_dir)