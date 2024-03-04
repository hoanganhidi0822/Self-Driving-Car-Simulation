import os
from PIL import Image
from glob import glob
DATA_PATH = os.path.join(os.getcwd())
print(DATA_PATH)
train_path = DATA_PATH + '/data4k/images/'
print(train_path)
label_path = DATA_PATH + '/data4k/labels/'
print(label_path)
def get_files(data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'jpg'))

def get_label_file(data_folder):
        #
        
        return  glob("{}/*.{}".format(data_folder, 'png'))


data_files = os.listdir(train_path)
data_files.sort(key= lambda i: int(i.lstrip('data').rstrip('.jpg')))
label_files = os.listdir(label_path)
label_files.sort(key= lambda i: int(i.lstrip('data').rstrip('.png')))
data_folder = []
label_folder = []

for f, datas in enumerate(data_files):

    file_data = train_path + data_files[f]
    file_label = label_path + label_files[f]
    data_folder.append(file_data)
    label_folder.append(file_label)
#print(label_folder)
# print(data_folder[1])
