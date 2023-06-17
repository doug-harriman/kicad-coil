# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:47:25 2021

@author: meindert.norg
"""

#%% Import libraries
import femm                              # "pip install pyfemm"from Anaconda prompt
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go  # pip install plotly
import plotly.express as px  # pip install plotly
from plotly.offline import plot
from plotly.subplots import make_subplots


def make_movie(DirName,MovieFileName):
# =============================================================================
#     import os
#     import moviepy.video.io.ImageSequenceClip
# # see https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
#     fps=1
# 
#     image_files = [DirName+'/'+img for img in os.listdir(DirName) if img.endswith(".png")]
#     clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
#     clip.write_videofile(MovieFileName)
# 
# =============================================================================
    import cv2
    import os
    
    image_folder = DirName
    video_name = MovieFileName
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()



#%% Make the FEMM object and configure


# Create a femm object
# - 0 to show the FEMM GUI
# - 1 to hide the FEMM GUI
femm.openfemm(0)

# Open a new model
# - 0 for a magnetics problem, 
# - 1 for an electrostatics problem, 
# - 2 for a heat flow problem, or 
# - 3 for a current flow problem
femm.newdocument(0)
   
# Define the problem and solver
# - freq:       Set freq to the desired frequency in Hertz
# - units:      The units parameter specifies the units used for measuring length in the problem domain. Valid ’units’ entries are ’inches’, ’millimeters’, ’centimeters’, ’mils’, ’meters’, and ’micrometers’ 
# - type:       Set the parameter problemtype to ’planar’ for a 2-D planar problem, or to ’axi’ for an axisymmetric problem. 
# - precision:  The precision parameter dictates the precision required by the solver. For example, entering 1E-8 requires the RMS of the residual to be less than 10−8. 
# - depth:      depth of the problem in the into-the-page direction for 2-D planar problems.  Specify the depth to be zero for axisymmetric problems. 
# - minangle:   The sixth parameter represents the minimum angle constraint sent to the mesh generator – 30 degrees is the usual choice for this parameter.
# - (acsolver)  The acsolver parameter specifies which solver is to be used for AC problems: 0 for successive approximation, 1 for Newton.
femm.mi_probdef(0,"meters","axi")

# Set FEMM window size
femm.main_resize(500,1000)

# Gather 'material'
magnetMaterial          = "N52"
coilMaterial            = "36 AWG"
airMaterial             = "Air"
femm.mi_getmaterial(magnetMaterial)     # Magnet material
femm.mi_getmaterial(coilMaterial)       # Copper wire AWG 32
femm.mi_getmaterial(airMaterial)        # Surrounding material is AIR

# supporting parameters
createDistanceZ         = -1            # [m]      Parts are 'created' with this offset, and then moved to the desired location

#%% Build the model

#=======================
# MAGNET
#=======================

# Define the magnet.
magnetLength            = 10e-3         # [m]
magnetOD                = 5e-3          # [m]
magnetGroupNr           = 10            # [-]
# Sraw the magnet
# femm.mi_drawrectangle(r1,y1,r2,y2)
# - (r1,y1) : first corner
# - (r2,y2) : opposing corner
#
# Note that for an "axi" system: x becomes r and y becomes z in FEMM
r1                      = 0
z1                      = -magnetLength/2 + createDistanceZ
r2                      =  magnetOD/2
z2                      = +magnetLength/2 + createDistanceZ
femm.mi_drawrectangle(r1,z1,r2,z2)

# Add magnet to group
femm.mi_selectrectangle(r1,z1,r2,z2)
femm.mi_setgroup(magnetGroupNr)
femm.mi_clearselected()
    
# Add Block label at center of manget
magnetLabel_r           = magnetOD/4
magnetLabel_z           = createDistanceZ
femm.mi_addblocklabel(magnetLabel_r, magnetLabel_z)
femm.mi_selectlabel(magnetLabel_r, magnetLabel_z)
femm.mi_setgroup(magnetGroupNr)
femm.mi_clearselected()

# Move to desired Z-position
femm.mi_selectgroup(magnetGroupNr)
femm.mi_movetranslate(0, -createDistanceZ)
femm.mi_clearselected()

# Assign material
femm.mi_selectlabel(magnetLabel_r, magnetLabel_z)
# – Block property ’blockname’.
# – automesh: 0 = mesher defers to mesh size constraint defined in meshsize, 1 = mesher automatically chooses the mesh density.
# – meshsize: size constraint on the mesh in the block marked by this label.
# – Block is a member of the circuit named ’incircuit’
# – The magnetization is directed along an angle in measured in degrees denoted by the parameter magdir
# – A member of group number group
# – The number of turns associated with this label is denoted by turns (- sign indicates reversed winding direction)
femm.mi_setblockprop(magnetMaterial,1,0.05,"",+90,magnetGroupNr)
# TODO: is this next line needed?
# femm.mi_setgroup(Magnet_GroupNr)   
femm.mi_clearselected()

# Zoom image
femm.mi_zoomnatural()
femm.mi_zoomout()


#=======================
# Electrical Circuit
#=======================
# mi_addcircprop(’circuitname’, i, circuittype) adds a new circuit property with 
# - ’circuitname’   - Name of circuit
# - i               - a prescribed current. 
# - circuittype     - 0 for a parallel-connected circuit and 1 for a series-connected circuit.
circuitName             = 'CoilCircuit'
circuitCurrent          = 10             # [A]
femm.mi_addcircprop(circuitName,circuitCurrent,1)   # 0 = parallel, 1 = series

#=======================
# COIL
#=======================

# Define the magnet.
coilLength              = 15e-3         # [m]
coilID                  = 10e-3          # [m]
coilOD                  = 12e-3         # [m]
coilGroupNr             = 20            # [-]
# Sraw the magnet
# femm.mi_drawrectangle(r1,y1,r2,y2)
# - (r1,y1) : first corner
# - (r2,y2) : opposing corner
#
# Note that for an "axi" system: x becomes r and y becomes z in FEMM
r1                      = coilID/2
z1                      = -coilLength/2 + createDistanceZ
r2                      = coilOD/2
z2                      = +coilLength/2 + createDistanceZ
femm.mi_drawrectangle(r1,z1,r2,z2)

# Add magnet to group
femm.mi_selectrectangle(r1,z1,r2,z2)
femm.mi_setgroup(coilGroupNr)
femm.mi_clearselected()
    
# Add Block label at center of manget
coilLabel_r             = (coilOD + coilID) / 4
coilLabel_z             = createDistanceZ
femm.mi_addblocklabel(coilLabel_r, coilLabel_z)
femm.mi_selectlabel(coilLabel_r, coilLabel_z)
femm.mi_setgroup(coilGroupNr)
femm.mi_clearselected()

# Move to desired Z-position
femm.mi_selectgroup(coilGroupNr)
femm.mi_movetranslate(0, -createDistanceZ)
femm.mi_clearselected()

# Assign material
femm.mi_selectlabel(coilLabel_r, coilLabel_z)
# – Block property ’blockname’.
# – automesh: 0 = mesher defers to mesh size constraint defined in meshsize, 1 = mesher automatically chooses the mesh density.
# – meshsize: size constraint on the mesh in the block marked by this label.
# – Block is a member of the circuit named ’incircuit’
# – The magnetization is directed along an angle in measured in degrees denoted by the parameter magdir
# – A member of group number group
# – The number of turns associated with this label is denoted by turns (- sign indicates reversed winding direction)
femm.mi_setblockprop(coilMaterial,1,0.05,circuitName,0,coilGroupNr,300)
# TODO: is this next line needed?
# femm.mi_setgroup(Magnet_GroupNr)   
femm.mi_clearselected()

# Zoom image
femm.mi_zoomnatural()
femm.mi_zoomout()

#===========================================
# Make ABC (Improvised Asymptotic Boundary Condition))
#===========================================

boundaryGroupNr         = 30

# https://www.femm.info/wiki/OpenBoundaryExample
# - The n parameter contains the number of shells to be used (should be between 1 and 10), 
# - R is the radius of the solution domain, and 
# - (x,y) denotes the center of the solution domain.
# - The bc parameter should be specified as 0 for a Dirichlet outer edge or 1 for a Neumann outer edge. 
femm.mi_makeABC(7,50e-3,0,0,1)

femm.mi_addblocklabel(coilOD*2,0)
femm.mi_selectlabel(coilOD*2,0)
femm.mi_setblockprop(airMaterial,1,0.3,"",0,boundaryGroupNr)
femm.mi_clearselected()

# Zoom image
femm.mi_zoomnatural()
femm.mi_zoomout()

#============================================
# Save file, create mesh
#============================================
fileName                = 'MyFemmModel.FEM'
femm.mi_saveas(fileName)
    
femm.mi_createmesh()

# Zoom image
femm.mi_zoomnatural()



#%% Move the magnet
#============================================
# Evaluate force as function of magnet position
#============================================

magnetStartPos          = -15e-3        # [m]
magnetEndPos            = 15e-3         # [m]
magnetStepSize          = 1e-3          # [m]       

# Move to start position
femm.mi_selectgroup(magnetGroupNr)
femm.mi_movetranslate(0, magnetStartPos)
femm.mi_clearselected()


# I found a weird thing regarding to python's rounding: 
# with:
    # magnetStartPos          = -15e-3        # [m]
    # magnetEndPos            = 15e-3         # [m]
    # magnetStepSize          = 5e-3          # [m] 
    # df_magnetPositions      = pd.Series(np.arange(magnetStartPos,magnetEndPos,magnetStepSize))
# results in 
    # df_magnetPositions
    # Out[164]: 
    # 0   -1.500000e-02
    # 1   -1.000000e-02
    # 2   -5.000000e-03
    # 3    3.469447e-18
    # 4    5.000000e-03
    # 5    1.000000e-02
    # dtype: float64
# This is one step short from where we want to end
# but, when extending the last step by +magnetStepSize
#    df_magnetPositions      = pd.Series(np.arange(magnetStartPos,magnetEndPos+magnetStepSize,magnetStepSize))
# results inone too many steps:
    # df_magnetPositions
    # Out[166]: 
    # 0   -1.500000e-02
    # 1   -1.000000e-02
    # 2   -5.000000e-03
    # 3    3.469447e-18
    # 4    5.000000e-03
    # 5    1.000000e-02
    # 6    1.500000e-02
    # 7    2.000000e-02
    # dtype: float64
# Solution: extend last step with a small delta.
df_magnetPositions      = pd.Series(np.arange(magnetStartPos,magnetEndPos+1e-6,magnetStepSize))
df_magneticData         = pd.DataFrame(index=df_magnetPositions)
# do one mock iteration to prepare output screen


imageDirName            = 'FEMM_Images'
if not os.path.exists(imageDirName):
    os.mkdir(imageDirName)
# Following line fails. Manually remove the files
# os.remove(os.path.join(imageDirName + "*.*"))     # removes any old files


for nrPosition in df_magnetPositions.index:
    magnetPosition      = df_magnetPositions.loc[nrPosition]
    print(f"Iteration {nrPosition+1} : {len(df_magnetPositions)} . Z-pos = {magnetPosition} [m].")
    

    # = = = = = = = = = = = 
    # Get Magnetic Force constant
    # = = = = = = = = = = = 

    # Set current to 1 A
    # mi_modifycircprop(’CircName’,propnum,value)
    # propnum:
    # - CircName    Name of the circuit property
    # - propnum     0 = CircName Name of the circuit property
    #               1 = Total current
    #               2 = CircType 0 = Parallel, 1 = Series
    # - value       New value for propnum
    
    femm.mi_modifycircprop(circuitName,1,circuitCurrent)
    
    # Perform analysis
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_showgrid()
    
    # grab metrics    
    femm.mo_clearblock()
    femm.mo_groupselectblock(coilGroupNr)
    fz              = femm.mo_blockintegral(19)
    df_magneticData.loc[magnetPosition,"MagnetForce"]      = fz
    df_magneticData.loc[magnetPosition,"ForceConstant"]    = fz / circuitCurrent
    for Nr in range(0,31):
        print(str(Nr) + " : " + str(femm.mo_blockintegral(Nr)))
    
    # = = = = = = = = = = = 
    # Get Linked Flux
    # = = = = = = = = = = = 

    # Set current to 0 A
    femm.mi_modifycircprop(circuitName,1,0)

    # Perform analysis
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_showgrid()

    # grab metrics from circuit
    # mo_getcircuitproperties("circuit")Used primarily to obtain impedance information
    # associated with circuit properties. Properties are returned for the circuit property named
    # "circuit". Three values are returned by the function. In order, these results are:
    # - current Current carried by the circuit
    # – volts Voltage drop across the circuit
    # – flux_re Circuit’s flux linkage
    v0,i0,fl_link      = femm.mo_getcircuitproperties(circuitName)
    
    df_magneticData.loc[magnetPosition,"FluxLinkage"]      = fl_link


    # Save image    
    # mo_showdensityplot(legend,gscale,upper_B,lower_B,type)
    # – legend      Set to 0 to hide the plot legend or 1 to show the plot legend.
    # – gscale      Set to 0 for a colour density plot or 1 for a grey scale density plot.
    # – upper_B     Sets the upper display limit for the density plot.
    # – lower_B     Sets the lower display limit for the density plot.
    # – type        Type of density plot to display. Valid entries are "bmag", "breal", and "bimag"
    #               for magnitude, real component, and imaginary component of flux density (B), respectively;
    #               "hmag", "hreal", and "himag" for magnitude, real component, and imaginary
    #               component of field intensity (H); and "jmag", "jreal", and "jimag" for magnitude,
    #               real component, and imaginary component of current density (J).
    # if legend is set to -1 all parameters are ignored and default values are used e.g.: mo_showdensityplot(-1)
    femm.mo_showdensityplot(1,0,1.2,0,'bmag')
    imageFileName           = 'FemmImage' + str(nrPosition).zfill(4) + '.png'
    fullImageFileName       = os.path.join(imageDirName,imageFileName) 
    femm.mo_savebitmap(fullImageFileName)
    # femm.mo_zoomnatural()

    # move magnet
    femm.mi_selectgroup(magnetGroupNr)
    femm.mi_movetranslate(0, magnetStepSize)
    femm.mi_clearselected()



  
# process data AFTER iteration

# = = = = = = = = = = = = = = = =
# differentiate flux to position
# = = = = = = = = = = = = = = = =
df_dFlux_Linkage        = pd.DataFrame()
df_dFlux_Linkage["dfFluxLinkage"] = df_magneticData["FluxLinkage"].diff() / (magnetStepSize)
# remove first row (contains NaNs)
df_dFlux_Linkage    = df_dFlux_Linkage.iloc[1:]
df_dFlux_Linkage.set_index((df_magneticData.index.values[:-1] + df_magneticData.index.values[1:])/2,inplace=True)
df_dFlux_Linkage.index.name = "MagnetPososion"

# # = = = = = = = = = = = = = = = =
# # calculate motor constant Kv
# # = = = = = = = = = = = = = = = =
# # There are several different motor constand definitions. For our 
# # application we care about the voltage generated as function of speed. 
# # Linear: [V0p / mps]
# #   1V_rms (= 1/sqrt(2)) --> 
# #   Kv is sqrt(0.5) * 1 / max(d_flux/drot) (notes 2021-05-21)
# #   
# # 1 m/s --> 1 Vrms

# FEMM_OutputList["GenKe_V0ppmps"]   = max(df_dFlux_Linkage.max(axis=0))

# # Get the max force at 1 A. This is the force constant.
# # Get the max from each column, then the max of the resulting df.
# # 
# # df_Magnetic_FEMM[["CircuitA_ForceConstant_NpA","CircuitB_ForceConstant_NpA","CircuitC_ForceConstant_NpA"]].max(axis=0)
# # CircuitA_MagnetForce_N    0.011557
# # CircuitB_MagnetForce_N    0.000388
# # CircuitC_MagnetForce_N    0.422790
# # dtype: float64
# FEMM_OutputList["GenKf_NpA"]        = df_Magnetic_FEMM[["CircuitA_ForceConstant_NpA","CircuitB_ForceConstant_NpA","CircuitC_ForceConstant_NpA"]].max(axis=0).max(axis=0)


# # =============================================================================
# femm.closefemm()

#%% Plot results
fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Scatter(x=df_magneticData.index*1e3, y=df_magneticData["MagnetForce"],name = "Magnet Force"),row=1,col=1)
fig.add_trace(go.Scatter(x=df_magneticData.index*1e3, y=df_magneticData["FluxLinkage"],name = "Flux Linkage"),row=2,col=1)
fig.add_trace(go.Scatter(x=df_dFlux_Linkage.index*1e3, y=df_dFlux_Linkage["dfFluxLinkage"],name = "Flux derivative"),row=3,col=1)

fig['layout']['xaxis3']['title']        ='Position [mm]'

fig['layout']['yaxis']['title']         ='Magnet Force [N]'
fig['layout']['yaxis2']['title']        ='Flux Linkage [Wb]'
fig['layout']['yaxis3']['title']        ='d_Flux/dx [Wb/m] <br> Motor_const [V/ m/s]'
fig.update_layout(xaxis_range=[-15,15])
fig.update_layout(xaxis2_range=[-15,15])
fig.update_layout(xaxis3_range=[-15,15])

plot(fig)

# create movie
movieFileName       = "FEMM_Movie.mp4"
make_movie(imageDirName,movieFileName)
    

    
