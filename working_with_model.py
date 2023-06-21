import io
import os
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import torch


# Load the saved model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('model/m1.pt')
model.eval()


classes = {0:"Neutral",1:"happiness",2:"Sadness",3:"Surprise",4:"Anger",5:"Fear",6:"Disgust"}

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5752, 0.4495, 0.4012],
                                     std=[0.2086, 0.1911, 0.1827])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor=tensor.to(device)
    output = model.forward(tensor)
    probs = torch.nn.functional.softmax(output[1], dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), classes.item()

frame_folder_name = "video3"
folder_path = "video_frames/"+frame_folder_name  # Change this to the path of your folder

# Define the number of rows and columns for subplots
total_cells = len(os.listdir(folder_path))
num_cols = 3
num_rows = total_cells//num_cols

# Create the subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))

for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".jpg"):  # Change this to the extension of your images
        image_path = os.path.join(folder_path, filename)

        # Read the image and plot it in the appropriate subplot
        image = plt.imread(image_path)
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].imshow(image)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            conf,y_pre=get_prediction(image_bytes=image_bytes)
            axs[row_idx, col_idx].set_title(classes[y_pre])
            print(y_pre, ' at confidence score:{0:.2f}'.format(conf))
            
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.show()