import os

# Specify the path to the folder containing your images
folder_path = 'D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\dataset\data4k\labels'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]

# Specify the name of the text file where you want to save the image names
output_file = 'labels.txt'

# Write the image names (without extensions) to the text file
with open(output_file, 'w') as file:
    for image_name in image_files:
        file.write(os.path.splitext(image_name)[0] + '\n')

print(f"Image names have been saved to {output_file}")
