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
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
iren.SetRenderWindow(renWin)

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
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform).to(device)
        for up_transform in tmp['up_transform']
    ]


    meshdata = MeshData("data/CoMA", "data/CoMA/template/template.obj", split="interpolation", test_exp="bareteeth")
    

    mean = meshdata.mean
    std = meshdata.std


    reader  = vtk.vtkOBJReader()
    reader.SetFileName("data/CoMA/template/template.obj")
    reader.Update()
    polydata = reader.GetOutput()


    model = AE(3, [32, 32, 32,64], 16, spiral_indices_list, down_transform_list, up_transform_list).to(device)
    checkpoint = torch.load("out/interpolation_exp/checkpoints/checkpoint_300.pt")
    model.load_state_dict( checkpoint["model_state_dict"] )
    model.eval()


    train_loader = DataLoader(meshdata.train_dataset, batch_size=1, shuffle=False)

    inputTensor = getInputData(polydata)
    inputTensor = (inputTensor - mean) / std
    inputTensor = inputTensor.to(device)

    x = meshdata.train_dataset[10].x.unsqueeze(0).to(device)


    # x = inputTensor
    out = model(x)
    loss = torch.nn.functional.l1_loss(out, x, reduction='mean')
    print(loss.item())


    out = (x.cpu() * std) +mean 

    outpoly = getOutputPoly(polydata, out)

    actor = MakeActor(outpoly)

    ren.AddActor(actor)
    renWin.Render()
    iren.Initialize()
    iren.Start()

    exit()



    # inputTensor = getInputData(polydata)
    # # inputTensor = (inputTensor - mean) / std
    # inputTensor = inputTensor.to(device)

    # pred = model(inputTensor)
    # # pred = (pred.cpu() * std) + mean
    # output = getOutputPoly(polydata, pred)



    # actor = MakeActor(output)

    # ren.AddActor(actor)

    # renWin.Render()
    # iren.Initialize()
    # iren.Start()



