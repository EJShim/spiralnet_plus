import vtk
import os
import glob
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



class LatentInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, model, samples, parent=None):


        
        self.AddObserver("LeftButtonPressEvent", self.LeftButtonPressed)
        self.AddObserver("MouseMoveEvent", self.MouseMove)
        self.AddObserver("LeftButtonReleaseEvent", self.LeftButtonReleased)


        self.model = model


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
        reader = vtk.vtkPLYReader()
        reader.SetFileName(samples[0])
        reader.Update()

        self.polydata = reader.GetOutput()
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
            
            sampleReader = vtk.vtkPLYReader()
            sampleReader.SetFileName(sample)
            sampleReader.Update()
            samplePoly = sampleReader.GetOutput()
            actor = MakeActor(samplePoly)
            actor.SetPosition(self.latentPositions[idx])
            ren.AddActor(actor)



            #ADd Target Actor
            sampleBatch = getInputData(samplePoly)
            z = self.model.encoder(sampleBatch.to(device))
            self.latentSize = z.shape[1]
            self.outputLatents.append(z[0])
            

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
        

        out = self.model.decoder(calculatedLatent)
        # out = out.detach().cpu()

        # target = out*self.std + self.mean
        updatePoly(self.polydata, out.detach().cpu())
        renWin.Render()


    def LeftButtonReleased(self, obj, ev):

        self.pickedPosition = -1
        self.OnLeftButtonUp()


if __name__ == "__main__":


    dataList = glob.glob( "data/CoMA/**/*.ply", recursive=True )



    checkpoint = torch.load("out/test/checkpoints/checkpoint_068.tar")
    model = AE(checkpoint['in_channels'],
                checkpoint['out_channels'], 
                checkpoint['latent_channels'], 
                checkpoint['spiral_indices'], 
                checkpoint['down_transform'], 
                checkpoint['up_transform'],
                checkpoint['std'].to(device),
                checkpoint['mean'].to(device)
                ).to(device)
    model.load_state_dict( checkpoint["model_state_dict"] )
    model.eval()


    samples = [
        dataList[10],
        dataList[5768],
        dataList[8654],
        dataList[2054]
    ]

    
    #Add Interactor Style
    interactorStyle = LatentInteractorStyle(model, samples)
    iren.SetInteractorStyle(interactorStyle)
    
    renWin.Render()
    iren.Initialize()
    iren.Start()



