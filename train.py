import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
import vtk
import glob


from reconstruction import AE, run, eval_error
from datasets import MeshData, PLYDataset
from torch.utils.data import DataLoader
from utils import utils, mesh_sampling
from utils.mesh_sampling import Mesh

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--inputs', type=str, default='data/CoMA')
parser.add_argument('--outputs', type=str, default='out')
parser.add_argument('--exp_name', type=str, default='test')

parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 32, 32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=16)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

#Prewpare OutputDirectory
checkpoints_dir = osp.join(args.outputs, args.exp_name, 'checkpoints')
os.makedirs(checkpoints_dir)


#Prepare some Hardware things
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True


#Meake Dataset
files = glob.glob( os.path.join(args.inputs, "raw", "**", "*.ply"), recursive=True)


nTrain = int( len(files) * 9 / 10)
trainDataset = PLYDataset(files[:nTrain])
testDataset = PLYDataset(files[nTrain:])

train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

#Calculate Mean and Std of Trainset
print("Calculating Mean and Std...")
tensors = []
for idx, data in enumerate(train_loader):
    tensors.append(data)
result = torch.cat(tensors, 0)

print("Done")
std = result.std(dim=0).to(device)
mean = result.mean(dim=0).to(device)


transform_fp = osp.join(args.inputs, 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')

    reader = vtk.vtkPLYReader()
    reader.SetFileName(files[0])
    reader.Update()
    mesh = Mesh()
    mesh.SetPolyData(reader.GetOutput())

    ds_factors = [4, 4, 4, 4]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }
    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')




spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]


model = AE(args.in_channels, args.out_channels, args.latent_channels, spiral_indices_list, down_transform_list, up_transform_list, std, mean).to(device)
print('Number of parameters: {}'.format(utils.count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.lr_decay)

run(model, train_loader, test_loader, args.epochs, optimizer, scheduler, checkpoints_dir, device)
# eval_error(model, test_loader, device, meshdata, args.out_dir)
