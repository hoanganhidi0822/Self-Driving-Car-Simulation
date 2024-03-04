import torch

state_dict = torch.load('Trained_Model/best_model.pth', map_location='cpu')
#print(state_dict)
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v
#print(compatible_state_dict)

mapping = {
     0 : (31,120,180),     # road
     1 : (227,26,28) ,     # people
     2 : (106,61,154),     # car
     3 : (0, 0, 0)   ,     # no label
}
rev_mapping = {mapping[k]: k for k in mapping}

print(mapping)
