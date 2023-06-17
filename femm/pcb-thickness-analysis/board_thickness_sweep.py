# Sweeps simple coil magnet model through different PCB thicknesses for a two layer board.
#

import pandas as pd
import webbrowser
import plotly.graph_objects as go
from pint import Quantity as Q
import numpy as np
from femm_model import Femm, CoilConductorsDistributed, Magnet

# Basic analysis config
gui = True
# gui = False
n_pts = 20

analysis_current = Q(20, 'A')

pcb_way_thicknesses = Q(np.array([0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2,
                                 1.6, 2.0, 2.4, 2.6, 2.8, 3.0, 3.2]), 'mm')
thicknesses = pcb_way_thicknesses[pcb_way_thicknesses < Q(2, 'mm')]

# Test Coil
# PCB design limit info:
# https://github.com/doug-harriman/kicad-coil/blob/main/design-info.md
track_thickness = Q(34.8, 'um')
track_spacing = Q(6, 'milliinch')
track_width = track_spacing
dia_inside = Q(1.6, 'mm')  # Measured from KiCAD PCB
turns = 12

# Create Model
model = Femm(gui=gui)
# Top layer
coil_l1 = CoilConductorsDistributed(model=model,
                                    turns=turns,
                                    track_thickness=track_thickness,
                                    track_width=track_width,
                                    track_spacing=track_spacing,
                                    dia_inside=dia_inside,
                                    current=analysis_current)
print(f'C1 LL: {coil_l1.ll}')

# Bottom layer
coil_l2 = CoilConductorsDistributed(model=model,
                                    turns=turns,
                                    track_thickness=track_thickness,
                                    track_width=track_width,
                                    track_spacing=track_spacing,
                                    dia_inside=dia_inside,
                                    y_center=Q(-10, 'mm'),
                                    current=analysis_current)
print(f'C2 LL: {coil_l2.ll}')

# Create magnet
mag = Magnet(model,
             width=Q(6, 'mm'),
             height=Q(1, 'mm'))
mag.ll_set(x=Q(2, 'mm') - mag.width/2, y=0.2)  # ~max force position

# Generate boundary conditions
model.boundary_generate(radius=25)

# Loop on board thicknesses, calculating magnet force.
x = coil_l1.ll[0]
forces = Q(np.zeros(len(thicknesses)), 'mN')
for i, thickness in enumerate(thicknesses):
    # Set coil Y position
    y = coil_l1.center[1] - thickness
    coil_l2.ul_set(x=x, y=y)

    model.analyze()

    if gui:
        model.save_image(b_max_tesla=0.6)

    forces[i] = mag.force()[0]  # X-component

    print(f'Thickness={thickness:0.3}, Force={forces[i]:0.2}', flush=True)

# Normalize force to unit current.
forces /= analysis_current.to('A').magnitude

fig = go.Figure()
trc = go.Scatter(x=thicknesses, y=forces)
fig.add_trace(trc)
fig.update_layout(title=f"PCB Thickness Analysis<br>Tolerance = {model._precision}<br>Analysis Current (normalized) = {analysis_current}",
                  xaxis_title=f'Thickness [{thicknesses.units}]',
                  yaxis_title=f'X-Direction Force on Magnet [{forces.units}]')
# file_html = 'test-plot.html'
file_html = 'pcb-thickness-analysis.html'
fig.write_html(file_html)

webbrowser.open(file_html)

# Save data
df = pd.DataFrame()
df['PCB Thicknesses'] = pd.Series(thicknesses)
df['Force'] = pd.Series(forces)
df.to_csv('pcb-thickness-analysis.csv')

if gui:
    from make_movie import make_movie
    make_movie('.', 'pcb-thickness-analysis.avi')
