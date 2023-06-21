import sys
import os
import subprocess
import io
import torch
from torchvision import datasets, transforms,models
from PIL import Image
import torch
from torch import nn
from resnet import resnet18
from collections import Counter
import matplotlib.pyplot as plt

# def createVideoFrame():
base_path = #Place your file wrapper file path here
#Put the video under a file called video_files
video_path = base_path + "/" + "video_files"
##Create a file called video_frames in the base bath
result_path = base_path+"/"+"video_frames"
video_name = sys.argv[1].split('.')[0]
command_p = 'mkdir -p '+result_path+'/'+video_name
result_path = result_path+"/"+video_name
out = subprocess.check_output(command_p,shell=True).decode()
filename = video_path+"/"+sys.argv[1]

#Load model
model = resnet18(num_classes=7)
# Load the state dictionary of the pretrained model
state_dict = torch.load('model_state/m1_1.pt')
model = nn.DataParallel(model)
# Load the state dictionary to the model
model.load_state_dict(state_dict)
# # Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=torch.load('model/model1_1.pt')
model.eval()

classes = {0:"Neutral",1:"happiness",2:"Sadness",3:"Surprise",4:"Anger",5:"Fear",6:"Disgust"}

# filename = os.path.abspath(os.getcwd())+"/"+sys.argv[1]

# print(filename)
    #filename = r'D:\Learning\cv\video\video1.mp4'

def save_i_keyframes(video_fn):
#    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    command = 'ffmpeg -i '+video_fn+' -vf "select=eq(n\,0)" -q:v 3 '+result_path+'/frame-00.jpg'
    out = subprocess.check_output(command,shell=True).decode()
    command = 'ffmpeg -i '+ video_fn+ ' -vf "select=\'gt(scene,0.008)\'" -fps_mode vfr '+result_path+'/frame-%2d.jpg'
    out = subprocess.check_output(command,shell=True).decode()
    #print(out)


    # if __name__ == '__main__':
    #     save_i_keyframes(filename)


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(2)
   

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
#     print(output)
    probs = torch.nn.functional.softmax(output[1], dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), classes.item()

def showPredictions():
    print("showing predictions... ")
    folder_path = "video_frames/"+video_name  # Change this to the path of your folder
    #Hardcoded folder path
    # folder_path = "val_images/"
    # Define the number of rows and columns for subplots
    total_cells = len(os.listdir(folder_path))
    num_cols = 2
    num_rows = (total_cells + num_cols - 1) // num_cols

    # num_rows = total_cells//num_cols
    list_pred = []

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize = (8*num_cols, 8*num_rows))
    # Hide X and Y axes label marks

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
                axs[row_idx, col_idx].set_title(classes[y_pre]+" at confidence: "+str(round(conf,2)))
                axs[row_idx, col_idx].axis('off') # remove axis
                axs[row_idx, col_idx].set_aspect('equal') # remove scale
                list_pred.append(y_pre)
    #             print(y_pre, ' at confidence score:{0:.2f}'.format(conf)) 
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.savefig(folder_path+'/plot_'+sys.argv[1].split('.')[0]+'.png')
    plt.show()
    plt.pause(0.001)

    reaction_outputs = most_frequent(list_pred)
    for i in range (len(reaction_outputs)):
        out = reaction_outputs[i]
        out_reaction_class = classes[out[0]]
        out_percentage = round((out[1]/total_cells)*100,2)
        print("The number "+ str(i+1) +" common reaction is "+ str(out_reaction_class)+" ("+str(out_percentage)+"%)")


if __name__ == '__main__':
    save_i_keyframes(filename)
    showPredictions()
    


