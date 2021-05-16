import vtk
import os
import pickle
import utils
import torch
from reconstruction import AE
import numpy as np
from utils import DataLoader
from datasets import MeshData

#Initialize Renderer
ren = vtk.vtkRenderer()
ren.GradientBackgroundOn()
ren.SetBackground(135/255, 206/255, 235/255)
ren.SetBackground2(44/255, 125/255, 158/255)
renWin = vtk.vtkRenderWindow()
renWin.SetFullScreen(False)
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

#JPolydata
reader  = vtk.vtkOBJReader()
reader.SetFileName("data/CoMA/template/template.obj")
reader.Update()
polydata = reader.GetOutput()

#Set Torch device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"




def getInputData(polydata):
    nPoints = polydata.GetNumberOfPoints()

    result = []

    for pid in range(nPoints):
        point = polydata.GetPoint(pid)
        result.append(point)


    tensor = torch.tensor([result])

    return tensor


def getOutputPoly(polydata, pred):
    output = vtk.vtkPolyData()
    output.DeepCopy(polydata)

    for pid, pos in enumerate(pred[0]):
        output.GetPoints().SetPoint(pid, pos[0], pos[1], pos[2])
    

    output.GetPoints().Modified()

    return output


def updatePoly(polydata, pred):
    for pid, pos in enumerate(pred[0]):
        polydata.GetPoints().SetPoint(pid, pos[0], pos[1], pos[2])
    

    polydata.GetPoints().Modified()


def MakeActor(polydata):
    
    #Visualize
    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputData(polydata)
    # mapper.SetFragmentShaderCode(frag)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor



if __name__ == "__main__":
    
    dilation = [1, 1, 1, 1]
    seq_length = [9, 9, 9, 9]

    transform_fp = os.path.join( "data", "CoMA", "transform.pkl" )
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], seq_length[idx], tmp['vertices'][idx], dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        utils.to_sparse(down_transform, device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform, device)
        for up_transform in tmp['up_transform']
    ]


    meshdata = MeshData("data/CoMA", "data/CoMA/template/template.obj", split="interpolation", test_exp="bareteeth")
    

    mean = meshdata.mean
    std = meshdata.std




    model = AE(3, [32, 32, 32,64], 16, spiral_indices_list, down_transform_list, up_transform_list, std, mean).to(device)
    checkpoint = torch.load("out/interpolation_exp/checkpoints/checkpoint_040.pt")
    model.load_state_dict( checkpoint["model_state_dict"] )
    model.eval()


    dummy_input = torch.zeros([1,5023,3])

    pred = model(dummy_input)

    torch.onnx.export(
        model, dummy_input, os.path.join( os.path.dirname(__file__), "spiralnet.onnx" ),
        verbose = True,
        do_constant_folding = True,
        opset_version = 12,
        input_names = ["input"],
        output_names = ["output"],        
    )

    print("Done")



