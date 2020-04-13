import torch
from torch.utils import data
from dataset import Dataset
from models import Model
import os
import argparse
import sys

import cv2
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SHA', type=str, help='dataset')
parser.add_argument('--data_path', default='/home/share/new_vgg10/ASD-PB/', type=str, help='path to dataset')
parser.add_argument('--save_path', default='/home/share/new_vgg10/new_attention/', type=str, help='path to save checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')

args = parser.parse_args()

test_dataset = Dataset(args.data_path, args.dataset, False)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

def save_density_map(density_map,output_dir, fname='results.png'): 
    mp = sio.loadmat('/home/share/new_vgg10/new_attention/map.mat')
    mp = mp['c']
    mp = mp[::-1]
    #density_map  = density_map.astype(np.float32, copy=False)
    #print(density_map.shape)
   #print(density_map.shape)
    #density_map = density_map.reshape((density_map.shape[2],density_map.shape[3]))
    #print(density_map.shape)
    density_map = density_map.data.cpu().numpy()
    #print(density_map.shape)
    #print(density_map.shape)
    #density_map = np.int64(density_map)

    density_map = 255*density_map/np.max(density_map)
    density_map = density_map[0][0]
    new_density = np.ones([density_map.shape[0], density_map.shape[1], 3], dtype=np.float)

    for i in range(density_map.shape[0]):
        for j in range(density_map.shape[1]):
            new_density[i][j] = mp[int(density_map[i][j])]*255
            new_density[i][j] = [int(ele) for ele in new_density[i][j]]
    #new_density = np.array(new_density*255)
    cv2.imwrite(os.path.join(output_dir, fname), new_density)
    #density_map = 255*density_map/np.max(density_map)
    #density_map= density_map[0][0]
    #cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
    
    
    


output_dir = '/home/share/new_vgg10/new_attention/output'
start_epoch = 0
end_epoch = 900
count = 0
best_mae = sys.maxsize
device = torch.device('cuda:' + str(args.gpu))

model = Model().to(device)

checkpoint = torch.load(os.path.join(args.save_path, '/home/share/new_vgg10/new_attention/checkpoint_latest_33.pth'))

#checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_latest_{}.pth'.format(epoch)))
model.load_state_dict(checkpoint['model'])
print('Epoch:{} MAE:{} MSE:{}'.format(checkpoint['epoch'], checkpoint['mae'], checkpoint['mse']))
model.eval()
with torch.no_grad():
    mae, mse = 0.0, 0.0
    for i, (images, gt) in enumerate(test_loader):
        images = images.to(device)
        #print(images.shape)
        predict, atm1,atm2,_,_,branch1,branch2,test1,test2 = model(images)

        print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), gt.item()))
        #mae += torch.abs(predict.sum() - gt).item()
        #mse += ((predict.sum() - gt) ** 2).item()
        
        save_density_map(predict, output_dir, 'output_' + str(count) + '.png')
        save_density_map(branch1, output_dir, 'output1_' + str(count) + '.png')
        save_density_map(branch2, output_dir, 'output2_' + str(count) + '.png')
        save_density_map(atm1, output_dir, 'atm1_' + str(count) + '.png')
        save_density_map(atm2, output_dir, 'atm2_' + str(count) + '.png')
        save_density_map(test1, output_dir, 'test1_' + str(count) + '.png')
        save_density_map(test2, output_dir, 'test2_' + str(count) + '.png')
        
        count = count + 1

    mae /= len(test_loader)
    mse /= len(test_loader)
    mse = mse ** 0.5
#    if mae < best_mae:
#       best_mae = mae
#       best_mse = mse
#       best_epoch = epoch
#    print ('\nMAE: %0.2f, MSE: %0.2f, num: %0.2f' % (best_mae,best_mse,best_epoch))
    print('MAE:{} MSE:{}'.format(mae, mse))
