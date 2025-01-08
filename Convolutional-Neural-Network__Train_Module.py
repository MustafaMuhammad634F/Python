import torchvision

from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from collections import OrderedDict
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]
trn_img_trnsfrms = transforms.Compose([ transforms.Resize((224,224)),
                                        transforms.RandomRotation(30),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean= mean_vals, std=std_vals)])

tst_img_trnsfrms = transforms.Compose([transforms.Resize((255,255)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean_vals, std=std_vals)])

vld_img_trnsfrms = transforms.Compose([transforms.Resize((255,255)), 
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(), 
                                       transforms.Normalize(mean=mean_vals, std=std_vals)])

trn_img_dataset = datasets.ImageFolder(train_dir, transform=trn_img_trnsfrms)

tst_img_dataset = datasets.ImageFolder(test_dir, transform=tst_img_trnsfrms)

vld_img_dataset = datasets.ImageFolder(valid_dir, transform=vld_img_trnsfrms)


trn_img_loader = torch.utils.data.DataLoader(trn_img_dataset, batch_size = 64, shuffle = True)
tst_img_loader = torch.utils.data.DataLoader(tst_img_dataset, batch_size = 64)
vld_img_loader = torch.utils.data.DataLoader(vld_img_dataset, batch_size = 64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_prcs_mdl = models.vgg16(pretrained = True)


for param in img_prcs_mdl.parameters():
    param.requires_grad = False

img_prcs_classifier = nn.Sequential(nn.Linear(25088, 7104), 
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(7104, 102), 
                                    nn.LogSoftmax(dim=1))

img_prcs_mdl.classifier = img_prcs_classifier

# ╚Loss Function <cost function>
mdl_lss_func = nn.NLLLoss()

# ╚Network Optimisation
mdl_optimiser = optim.Adam(img_prcs_mdl.classifier.parameters(), lr= 0.002)

img_prcs_mdl = img_prcs_mdl.to(device)
# •Training Neural Network
epochs = 5
step = 0    
ttl_loss = 0
batch_count = 5

for i in range(epochs):

    for img, lbl in trn_img_loader:
        img, lbl = img.to(device), lbl.to(device)
        step += 1
        prcsd_img = img_prcs_mdl(img)
        lss_val = mdl_lss_func(prcsd_img, lbl)
        mdl_optimiser.zero_grad()
        lss_val.backward()
        mdl_optimiser.step()
        
        ttl_loss += lss_val

            # Validating for each 10 batches
        if step % batch_count == 0:
            tst_lss_val = 0
            accuracy = 0
            img_prcs_mdl.eval()

            with torch.no_grad():
                for img, lbls in vld_img_loader:
                    img, lbls = img.to(device), lbls.to(device)
                    probe_vals = img_prcs_mdl(img)
                    lss_val = mdl_lss_func(probe_vals, lbls)
                    tst_lss_val += lss_val.item()
                    probes = torch.exp(probe_vals)
                    top_probe, top_clss = probes.topk(1, dim=1)
                    probe_corel = top_clss == lbls.view(*top_clss.shape)
                    accuracy += torch.mean(probe_corel.type(torch.FloatTensor)).item()
            print(f"Validation accuracy: {accuracy/len(vld_img_loader)}")
            print(f"Training loss: {ttl_loss/batch_count}")
            ttl_loss = 0
            img_prcs_mdl.train()

tot_test_loss = 0
test_correct = 0  # Number of correct predictions on the validation set
        
        # Turn off gradients for validation, saves memory and computations
with torch.no_grad():
    img_prcs_mdl.eval()
    for images, labels in tst_img_loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = img_prcs_mdl(images)
            loss = mdl_lss_func(log_ps, labels)
            tot_test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_correct += torch.mean(equals.type(torch.FloatTensor))
    #img_prcs_mdl.train()
    print("Test Accuracy: {:.3f}".format(test_correct / len(tst_img_loader)))
  

# Saving the check point of the deduct module in order to be reused
mdl_chk_dict = {
    'input size': 25088,
    'output size': 102,
    'hidden nodes':7104,
    'Drop out':0.2,
    'state_dict': img_prcs_mdl.state_dict()
}
torch.save(mdl_chk_dict,'chk_point.pth')

def load_chk_point(file_pth):
    loaded_params = torch.load(file_pth)
    loaded_mdl = nn.sequential(nn.Linear(loaded_params['input size'],   loaded_params['hidden nodes']), 
                               nn.Relu(), 
                               nn.Dropout(loaded_params['Drop out']),
                               nn.Linear(loaded_params['hidden nodes'], loaded_params['output size']),
                               nn.LogSoftmax(dim=1))
    pre_trnd_mdl = models.vgg16(pretrained = True)
    pre_trnd_mdl.classifier = loaded_mdl
    mdl_optimiser = optim.Adam(img_prcs_mdl.classifier.parameters(), lr= 0.002)
    loaded_mdl.load_state_dict(loaded_params['state_dict'])
    return pre_trnd_mdl, mdl_optimiser


