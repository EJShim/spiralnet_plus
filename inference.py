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



class Encoder(torch.nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.encoder(x)

class Decoder(torch.nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, z):
        return self.model.decoder(z)




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



class LatentInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, model, target, samples, parent=None):
        
        self.AddObserver("LeftButtonPressEvent", self.LeftButtonPressed)
        self.AddObserver("MouseMoveEvent", self.MouseMove)
        self.AddObserver("LeftButtonReleaseEvent", self.LeftButtonReleased)

        # self.model = model    
        self.encoder = Encoder(model)   
        self.decoder = Decoder(model) 


        #Initialize Plane        
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(0, 0, 0)
        planeSource.Update()
        planePoly = planeSource.GetOutput()
        planePoly.GetPointData().RemoveArray("Normals")
        self.planeActor = MakeActor(planePoly)
        self.planeActor.GetProperty().SetRepresentationToWireframe()
        self.planeActor.GetProperty().SetColor(1, 0, 0)


    

        ren.AddActor(self.planeActor)


        #ADd Target Actor
        # target = target*std +mean
        self.polydata = getOutputPoly(polydata, target)
        self.actor = MakeActor(self.polydata)
        ren.AddActor(self.actor)




        bounds = self.planeActor.GetBounds()
        self.latentPositions = np.array( [
            [bounds[0], bounds[2], 0],
            [bounds[0], bounds[3], 0],
            [bounds[1], bounds[2], 0],
            [bounds[1], bounds[3], 0]
        ])
        

        self.latentSize = 16
        self.outputLatents = []
        #Add Sample Actor
        for idx, sample in enumerate(samples):

            #ADd Target Actor
            z = self.encoder(sample.to(device))            
            self.latentSize = z.shape[1]
            self.outputLatents.append(z[0])

            print(z)
            
            outpoly = getOutputPoly(polydata, sample)
            actor = MakeActor(outpoly)
            actor.SetPosition(self.latentPositions[idx])
            ren.AddActor(actor)

        self.pickedPosition = -1
            

    def LeftButtonPressed(self, obj, ev):
        
        self.OnLeftButtonDown()

        pos = obj.GetInteractor().GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.planeActor)
        picker.Pick(pos[0], pos[1], 0, ren)

    

        position = picker.GetPickPosition()
        
        if picker.GetActor() == self.planeActor:
            self.pickedPosition = position

            

    def MouseMove(self, obj, ev):
        if self.pickedPosition == -1:
            self.OnMouseMove()
            return

        pos = obj.GetInteractor().GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.planeActor)
        picker.Pick(pos[0], pos[1], 0, ren)
        
        
        position = picker.GetPickPosition()
        targetPos = np.array([position[0], position[1], 0])


        if targetPos[0] < -0.5 : targetPos[0] = -0.5
        elif targetPos[0] > 0.5 : targetPos[0] = 0.5
        if targetPos[1] < -0.5 : targetPos[1] = -0.5
        elif targetPos[1] > 0.5 : targetPos[1] = 0.5

        distances = []
        for sample in self.latentPositions:            
            distances.append(np.linalg.norm(targetPos-sample))
                

        weights = np.array(distances)

        weights[weights > 1] = 1
        weights = 1 - weights
        
        calculatedLatent = torch.zeros(self.latentSize).to(device)

        for idx, weight in enumerate(weights):
            calculatedLatent += self.outputLatents[idx] * weight
        
            
        out = self.decoder(calculatedLatent)
        out = out.detach().cpu()

        print(out)


         
        
        updatePoly(self.polydata, out)
        renWin.Render()


    def LeftButtonReleased(self, obj, ev):

        self.pickedPosition = -1
        self.OnLeftButtonUp()


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


    meshdata = MeshData("data/CoMA", "data/CoMA/template/template.obj", split="interpolation", test_exp="bareteeth", normalize=False)
    

    mean = meshdata.mean
    std = meshdata.std




    model = AE(3, [32, 32, 32,64], 16, spiral_indices_list, down_transform_list, up_transform_list, std, mean).to(device)
    checkpoint = torch.load("out/interpolation_exp/checkpoints/checkpoint_040.pt")
    model.load_state_dict( checkpoint["model_state_dict"] )
    model.eval()

    
    train_loader = DataLoader(meshdata.train_dataset, batch_size=1, shuffle=False)

    x = meshdata.train_dataset[10].x.unsqueeze(0)

    samples = [
        meshdata.train_dataset[10].x.unsqueeze(0),
        meshdata.train_dataset[5768].x.unsqueeze(0),
        meshdata.train_dataset[8654].x.unsqueeze(0),
        meshdata.train_dataset[2054].x.unsqueeze(0)
    ]

    
    #Add Interactor Style
    interactorStyle = LatentInteractorStyle(model, x, samples)
    iren.SetInteractorStyle(interactorStyle)
    
    renWin.Render()
    iren.Initialize()
    iren.Start()



