# Updated 15/05/2025, fixed problem with overlapping sketch in rail extrude cut.

from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

# Non-abaqus modules
import json
import os

# Import parameters from setup
with open('import_params.json') as f:
    params = json.load(f)

JobName         = params['JobName']
MeshVal         = params['MeshVal']
RailRadius      = params['RailRadius']
WheelRadiusX    = params['WheelRadiusX']
WheelRadiusY    = params['WheelRadiusY']
AxleLoad        = params['AxleLoad']

# Parameterize These same as above
# RailRadius = 300
# WheelRadiusX = 300
# WheelRadiusY = 250
# AxleLoad = -100000.0

### Wheel Sketch
mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=500.0)
mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0), 
    point2=(25.0, 25.0))
mdb.models['Model-1'].Part(dimensionality=THREE_D, name='Wheel', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].parts['Wheel'].BaseSolidExtrude(depth=25.0, sketch=
    mdb.models['Model-1'].sketches['__profile__'])
del mdb.models['Model-1'].sketches['__profile__']
mdb.models['Model-1'].ConstrainedSketch(gridSpacing=2.16, name='__profile__', 
    sheetSize=86.6, transform=
    mdb.models['Model-1'].parts['Wheel'].MakeSketchTransform(
    sketchPlane=mdb.models['Model-1'].parts['Wheel'].faces[2], 
    sketchPlaneSide=SIDE1, 
    sketchUpEdge=mdb.models['Model-1'].parts['Wheel'].edges[9], 
    sketchOrientation=RIGHT, origin=(25.0, 12.5, 12.5)))
mdb.models['Model-1'].parts['Wheel'].projectReferencesOntoSketch(filter=
    COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(point1=(-32.4, 
    32.4), point2=(23.22, 32.4))
mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
    addUndoState=False, entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[6])
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-33.48, -3.24), 
    point2=(-33.48, -21.6))
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
    False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[7])
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-33.48, -21.6), 
    point2=(11.88, -21.6))
mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
    addUndoState=False, entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[8])
mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
    addUndoState=False, entity1=
    mdb.models['Model-1'].sketches['__profile__'].geometry[7], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[8])
mdb.models['Model-1'].sketches['__profile__'].ArcByCenterEnds(center=(12.42, 
    15.66), direction=COUNTERCLOCKWISE, point1=(-14.04, 0.0), point2=(11.88, 
    -16.74))
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-33.48, -3.24), 
    point2=(-14.04, 0.0))
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(11.9076240879424, 
    -15.0825547234569), point2=(11.88, -21.6))
mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[10])
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[11])
mdb.models['Model-1'].sketches['__profile__'].TangentConstraint(entity1=
    mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[4])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(entity1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[8], entity2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[3])
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -8.02595138549805, -28.6103477478027), value=100.0, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[5], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[6])
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -96.8258666992188, -9.78799629211426), value=25.0, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[4], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[5])
mdb.models['Model-1'].sketches['__profile__'].RadialDimension(curve=
    mdb.models['Model-1'].sketches['__profile__'].geometry[9], radius=30.0, 
    textPoint=(3.88365936279297, 4.04998779296875))
mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(point1=(12.5, 
    12.5), point2=(12.5, -9.97119522094727))
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
    False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[12])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
    addUndoState=False, entity1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[2], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[12])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(entity1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[9], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[12])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(entity1=
    mdb.models['Model-1'].sketches['__profile__'].geometry[12], entity2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[9])
mdb.models['Model-1'].sketches['__profile__'].delete(objectList=(
    mdb.models['Model-1'].sketches['__profile__'].geometry[12], ))
mdb.models['Model-1'].sketches['__profile__'].dimensions[2].setValues(value=300)
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -85.4636077880859, 12.089527130127), value=50.0, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[4], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[7])
mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
    mdb.models['Model-1'].sketches['__profile__'].geometry[6], entity2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[8], textPoint=(
    32.272876739502, 0.542987823486328), value=250)
mdb.models['Model-1'].parts['Wheel'].CutRevolve(angle=90.0, 
    flipRevolveDirection=OFF, sketch=
    mdb.models['Model-1'].sketches['__profile__'], sketchOrientation=RIGHT, 
    sketchPlane=mdb.models['Model-1'].parts['Wheel'].faces[2], sketchPlaneSide=
    SIDE1, sketchUpEdge=mdb.models['Model-1'].parts['Wheel'].edges[9])
del mdb.models['Model-1'].sketches['__profile__']
mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=500.0)
mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0), 
    point2=(25.0, -25.0))
mdb.models['Model-1'].Part(dimensionality=THREE_D, name='rail', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].parts['rail'].BaseSolidExtrude(depth=25.0, sketch=
    mdb.models['Model-1'].sketches['__profile__'])
del mdb.models['Model-1'].sketches['__profile__']

########################################################################################################
### Rail Sketch
########################################################################################################
mdb.models['Model-1'].ConstrainedSketch(gridSpacing=2.16, name='__profile__', 
    sheetSize=86.6, transform=
    mdb.models['Model-1'].parts['rail'].MakeSketchTransform(
    sketchPlane=mdb.models['Model-1'].parts['rail'].faces[0], 
    sketchPlaneSide=SIDE1, 
    sketchUpEdge=mdb.models['Model-1'].parts['rail'].edges[2], 
    sketchOrientation=RIGHT, origin=(25.0, -12.5, 12.5)))
mdb.models['Model-1'].parts['rail'].projectReferencesOntoSketch(filter=
    COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-31.32, -2.7), 
    point2=(-31.32, 18.3600000001024))
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
    False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-31.32, 
    18.3600000001024), point2=(8.1, 18.3600000001024))
mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
    addUndoState=False, entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[7])
mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
    addUndoState=False, entity1=
    mdb.models['Model-1'].sketches['__profile__'].geometry[6], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[7])
mdb.models['Model-1'].sketches['__profile__'].ArcByCenterEnds(center=(12.42, 
    -15.66), direction=COUNTERCLOCKWISE, point1=(12.5, 12.5), point2=(-20.52, 
    -8.1))
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-31.32, -2.7), 
    point2=(-15.026530430604, -9.36079629461548))
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(8.1, 
    18.3600000001024), point2=(12.5, 12.5))
mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(point1=(12.5, 
    12.5), point2=(12.5, -12.5))
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
    False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[11])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
    addUndoState=False, entity1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[11])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
    addUndoState=False, entity1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[2], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[11])
mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(entity1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[8], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[11])
mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[9])
mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(entity=
    mdb.models['Model-1'].sketches['__profile__'].geometry[10])
mdb.models['Model-1'].sketches['__profile__'].TangentConstraint(entity1=
    mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
    mdb.models['Model-1'].sketches['__profile__'].geometry[4])
mdb.models['Model-1'].sketches['__profile__'].RadialDimension(curve=
    mdb.models['Model-1'].sketches['__profile__'].geometry[8], radius=300, 
    textPoint=(-0.850262641906738, -4.01655769348145))

### Dimension set to RailRadius*2 to ensure cut-extrude sketch does not overlap
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -16.1393032073975, 45.2653312683105), value=RailRadius*2, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[5], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[6])

mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    16.1838989257812, 17.2579798698425), value=10.0, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[6], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[3])
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -548.372680664062, -106.250274658203), value=250.0, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[4], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[5])
mdb.models['Model-1'].parts['rail'].CutExtrude(flipExtrudeDirection=OFF, 
    sketch=mdb.models['Model-1'].sketches['__profile__'], sketchOrientation=
    RIGHT, sketchPlane=mdb.models['Model-1'].parts['rail'].faces[0], 
    sketchPlaneSide=SIDE1, sketchUpEdge=
    mdb.models['Model-1'].parts['rail'].edges[2])
del mdb.models['Model-1'].sketches['__profile__']


########################################################################################################
### UPDATE DIMENSIONS - EDIT RADII HERE
########################################################################################################

mdb.models['Model-1'].rootAssembly.regenerate()
mdb.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=
    mdb.models['Model-1'].parts['Wheel'].features['Cut revolve-1'].sketch)
mdb.models['Model-1'].parts['Wheel'].projectReferencesOntoSketch(filter=
    COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__edit__'], 
    upToFeature=mdb.models['Model-1'].parts['Wheel'].features['Cut revolve-1'])

########################################################################################################
# Edit radius for wheel here vvv
########################################################################################################
mdb.models['Model-1'].sketches['__edit__'].dimensions[4].setValues(value=WheelRadiusY)
mdb.models['Model-1'].sketches['__edit__'].dimensions[2].setValues(value=WheelRadiusX)
########################################################################################################
# Edit X radius for wheel here ^^^
########################################################################################################

mdb.models['Model-1'].parts['Wheel'].features['Cut revolve-1'].setValues(
    sketch=mdb.models['Model-1'].sketches['__edit__'])
del mdb.models['Model-1'].sketches['__edit__']
mdb.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=
    mdb.models['Model-1'].parts['rail'].features['Cut extrude-1'].sketch)
mdb.models['Model-1'].parts['rail'].projectReferencesOntoSketch(filter=
    COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__edit__'], 
    upToFeature=mdb.models['Model-1'].parts['rail'].features['Cut extrude-1'])
########################################################################################################
# Edit X radius for Rail here vvv
########################################################################################################
mdb.models['Model-1'].sketches['__edit__'].dimensions[0].setValues(value=RailRadius)
########################################################################################################
# Edit X radius for Rail here ^^^
########################################################################################################
mdb.models['Model-1'].parts['rail'].features['Cut extrude-1'].setValues(sketch=
    mdb.models['Model-1'].sketches['__edit__'])
del mdb.models['Model-1'].sketches['__edit__']
mdb.models['Model-1'].parts['rail'].regenerate()
mdb.models['Model-1'].parts['Wheel'].regenerate()


from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
mdb.models['Model-1'].Material(name='WheelMaterial')
mdb.models['Model-1'].materials['WheelMaterial'].Density(table=((0.00785, ), ))
mdb.models['Model-1'].materials['WheelMaterial'].Elastic(table=((200000.0, 
    0.28), ))
mdb.models['Model-1'].Material(name='RailMaterial')
mdb.models['Model-1'].materials['RailMaterial'].Density(table=((0.00785, ), ))
mdb.models['Model-1'].materials['RailMaterial'].Elastic(table=((200000.0, 
    0.28), ))

# Plasticity Parameters for Rail
mdb.models['Model-1'].materials['RailMaterial'].Plastic(dataType=PARAMETERS, 
    hardening=COMBINED, scaleStress=None, table=((419.0, 3200.0, 0.168421), ))
mdb.models['Model-1'].materials['RailMaterial'].plastic.CyclicHardening(
    parameters=ON, table=((419.0, 45.0, 981.0), ))


mdb.models['Model-1'].HomogeneousSolidSection(material='RailMaterial', name=
    'RailSection', thickness=None)
mdb.models['Model-1'].parts['rail'].Set(cells=
    mdb.models['Model-1'].parts['rail'].cells.getSequenceFromMask(('[#1 ]', ), 
    ), name='RailSet')
mdb.models['Model-1'].parts['rail'].SectionAssignment(offset=0.0, offsetField=
    '', offsetType=MIDDLE_SURFACE, region=
    mdb.models['Model-1'].parts['rail'].sets['RailSet'], sectionName=
    'RailSection', thicknessAssignment=FROM_SECTION)
mdb.models['Model-1'].HomogeneousSolidSection(material='WheelMaterial', name=
    'WheelSection', thickness=None)
mdb.models['Model-1'].parts['Wheel'].Set(cells=
    mdb.models['Model-1'].parts['Wheel'].cells.getSequenceFromMask(('[#1 ]', ), 
    ), name='WheelSet')
mdb.models['Model-1'].parts['Wheel'].SectionAssignment(offset=0.0, offsetField=
    '', offsetType=MIDDLE_SURFACE, region=
    mdb.models['Model-1'].parts['Wheel'].sets['WheelSet'], sectionName=
    'RailSection', thicknessAssignment=FROM_SECTION)

########################################################################################################
# Edit Wheel Mesh here vvv
########################################################################################################
mdb.models['Model-1'].parts['Wheel'].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=MeshVal)
########################################################################################################
# Edit Wheel Mesh here ^^^
########################################################################################################

mdb.models['Model-1'].parts['Wheel'].generateMesh()

########################################################################################################
# Edit Wheel Mesh here vvv
########################################################################################################
mdb.models['Model-1'].parts['rail'].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=MeshVal)
########################################################################################################
# Edit Wheel Mesh here ^^^
########################################################################################################

mdb.models['Model-1'].parts['rail'].generateMesh()
mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Wheel-1', part=
    mdb.models['Model-1'].parts['Wheel'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='rail-1', part=
    mdb.models['Model-1'].parts['rail'])
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Wheel-1', ), 
    vector=(0.0, 0.0001, 0.0))
mdb.models['Model-1'].rootAssembly.ReferencePoint(point=(0.0, 30.0, 0.0))
mdb.models['Model-1'].rootAssembly.Set(name='CouplingSet', referencePoints=(
    mdb.models['Model-1'].rootAssembly.referencePoints[6], ))
mdb.models['Model-1'].rootAssembly.Surface(name='WheelTop', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Wheel-1'].faces.getSequenceFromMask(
    ('[#10 ]', ), ))
mdb.models['Model-1'].Coupling(alpha=0.0, controlPoint=
    mdb.models['Model-1'].rootAssembly.sets['CouplingSet'], couplingType=
    KINEMATIC, influenceRadius=WHOLE_SURFACE, localCsys=None, name=
    'ConstraintCoupling', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['WheelTop'], u1=ON, u2=ON, u3=
    ON, ur1=ON, ur2=ON, ur3=ON)
mdb.models['Model-1'].ContactProperty('IntProp-1')
mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
    dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None, 
    formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION, 
    pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF, 
    table=((0.3, ), ), temperatureDependency=OFF)
mdb.models['Model-1'].interactionProperties['IntProp-1'].NormalBehavior(
    allowSeparation=ON, constraintEnforcementMethod=DEFAULT, 
    pressureOverclosure=HARD)
mdb.models['Model-1'].ContactStd(createStepName='Initial', name='IntContact')
mdb.models['Model-1'].interactions['IntContact'].includedPairs.setValuesInStep(
    stepName='Initial', useAllstar=ON)
mdb.models['Model-1'].interactions['IntContact'].contactPropertyAssignments.appendInStep(
    assignments=((GLOBAL, SELF, 'IntProp-1'), ), stepName='Initial')
mdb.models['Model-1'].rootAssembly.Set(faces=
    mdb.models['Model-1'].rootAssembly.instances['Wheel-1'].faces.getSequenceFromMask(
    ('[#2 ]', ), ), name='WheelSymX')
mdb.models['Model-1'].XsymmBC(createStepName='Initial', localCsys=None, name=
    'WheelSymX', region=mdb.models['Model-1'].rootAssembly.sets['WheelSymX'])
mdb.models['Model-1'].rootAssembly.Set(faces=
    mdb.models['Model-1'].rootAssembly.instances['Wheel-1'].faces.getSequenceFromMask(
    ('[#4 ]', ), ), name='WheelSymZ')
mdb.models['Model-1'].ZsymmBC(createStepName='Initial', localCsys=None, name=
    'WheelSymZ', region=mdb.models['Model-1'].rootAssembly.sets['WheelSymZ'])
mdb.models['Model-1'].rootAssembly.Set(faces=
    mdb.models['Model-1'].rootAssembly.instances['rail-1'].faces.getSequenceFromMask(
    ('[#2 ]', ), ), name='RailSymX')
mdb.models['Model-1'].XsymmBC(createStepName='Initial', localCsys=None, name=
    'RailSymX', region=mdb.models['Model-1'].rootAssembly.sets['RailSymX'])
mdb.models['Model-1'].rootAssembly.Set(faces=
    mdb.models['Model-1'].rootAssembly.instances['rail-1'].faces.getSequenceFromMask(
    ('[#20 ]', ), ), name='RailSymZ')
mdb.models['Model-1'].ZsymmBC(createStepName='Initial', localCsys=None, name=
    'RailSymZ', region=mdb.models['Model-1'].rootAssembly.sets['RailSymZ'])
mdb.models['Model-1'].rootAssembly.Set(faces=
    mdb.models['Model-1'].rootAssembly.instances['rail-1'].faces.getSequenceFromMask(
    ('[#8 ]', ), ), name='RailBotSet')
mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial', 
    distributionType=UNIFORM, fieldName='', localCsys=None, name='RailBotBC', 
    region=mdb.models['Model-1'].rootAssembly.sets['RailBotSet'], u1=UNSET, u2=
    SET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
mdb.models['Model-1'].StaticStep(name='StepContact', previous='Initial')
mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName=
    'StepContact', distributionType=UNIFORM, fieldName='', fixed=OFF, 
    localCsys=None, name='ContactBC', region=
    mdb.models['Model-1'].rootAssembly.sets['CouplingSet'], u1=UNSET, u2=
    -0.001, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
mdb.models['Model-1'].StaticStep(name='StepLoad', previous='StepContact')
mdb.models['Model-1'].boundaryConditions['ContactBC'].deactivate('StepLoad')


###### ADDED BY AXEL vvv
mdb.models['Model-1'].rootAssembly.Set(name='RP-1BCSet', referencePoints=(
    mdb.models['Model-1'].rootAssembly.referencePoints[6], ))

mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial', 
    distributionType=UNIFORM, fieldName='', localCsys=None, name='RP-1 BC', 
    region=mdb.models['Model-1'].rootAssembly.sets['RP-1BCSet'], u1=UNSET, u2=
    UNSET, u3=UNSET, ur1=SET, ur2=SET, ur3=SET)
###### ADDED BY AXEL ^^^


########################################################################################################
# Edit Wheel Load here vvv
########################################################################################################
mdb.models['Model-1'].ConcentratedForce(cf2=AxleLoad, createStepName=
    'StepLoad', distributionType=UNIFORM, field='', localCsys=None, name=
    'LoadForce', region=mdb.models['Model-1'].rootAssembly.sets['CouplingSet'])
########################################################################################################
# Edit Wheel Load here ^^^
########################################################################################################



mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
    explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
    memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
    multiprocessingMode=DEFAULT, name=JobName, nodalOutputPrecision=
    SINGLE, numCpus=1, numGPUs=0, numThreadsPerMpiProcess=1, queue=None, 
    resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=
    0, waitMinutes=0)

mdb.models['Model-1'].steps['StepContact'].setValues(maxNumInc=1000)
mdb.models['Model-1'].steps['StepContact'].setValues(initialInc=1, maxInc=1.0
    , maxNumInc=100)

mdb.models['Model-1'].steps['StepLoad'].setValues(initialInc=0.0001, minInc=1E-6, maxInc=1.0
    , maxNumInc=100)
mdb.models['Model-1'].steps['StepLoad'].setValues(nlgeom=ON)

mdb.jobs[JobName].submit(consistencyChecking=OFF)
mdb.jobs[JobName].waitForCompletion()

