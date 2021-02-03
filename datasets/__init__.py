from .coma import CoMA
from .faust import FAUST
from .meshdata import MeshData

import torch
from torch.utils.data import Dataset
import vtk
from vtk.util import numpy_support

__all__ = [
    'CoMA',
    'FAUST',
    'MeshData',
]


class PLYDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        reader = vtk.vtkPLYReader()
        reader.SetFileName( self.files[idx] )
        reader.Update()
        polydata = reader.GetOutput()


        points = numpy_support.vtk_to_numpy( polydata.GetPoints().GetData() )
        
 
        return torch.tensor(points)
