# PCB coil simulation
# TODO: Move coil generation to the KiCAD coil generator.

from pint import Quantity as Q
import numpy as np


def CuWeight2Thickness(weight: Q = Q(1, 'oz/ft^2')) -> Q:
    """
    Converts PCB copper weight to thickness using Pint quantities.
    """

    # Mapping table per: https://pcbprime.com/pcb-tips/how-thick-is-1oz-copper/
    pcb_weight = Q(np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]), 'oz/ft^2')
    trace_thickness = Q(np.array(
        [34.80, 52.20, 69.60, 104.39, 139.19, 173.99, 208.79, 243.59, 278.38, 313.18]), 'um')

    # Interpolate
    thickness = np.interp(weight, pcb_weight,
                          trace_thickness, left=np.NaN, right=np.NaN)
    return thickness


# Board thicknesses available:
# thicknesses_pcbway = [.2, .3, .4, .6, .8, 1.0, 1.2, 1.6, 2.0, 2.4, 2.6, 2.8, 3.0, 3.2]
# track_spacing_width_pcbway = Q([3,4,5,6,8],'milliinch')
# copper_weight_pcbway = Q(np.arange(1,14),'oz/ft**2')

# Design as of 03-Jan-2023
copper_weight = Q(1, 'oz/ft^2')
track_spacing = Q(6, 'milliinch').to('mm')
track_width = track_spacing
board_layers = 2  # Must be even
board_thickness = Q(1.6, 'mm')

track_thickness = CuWeight2Thickness(copper_weight).to('mm')
print(track_thickness)
