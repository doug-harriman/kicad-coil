from kicad_phase_coil import *
from pint import Quantity as Q
import os
import numpy as np

# Supporting elements
v = Via()
v.drill = 0.3
v.size = 0.65

hole = GrCircle()
hole.diameter = 3

id = 2
od = id + 18
layers = ["F.Cu", "B.Cu"]
width = Q(8e-3, "in").to("mm").magnitude  # mils to mm.
spacing = Q(6e-3, "in").to("mm").magnitude  # mils to mm.

x = 50
dx = 35
y = 50


filename = "tmp.txt"
if os.path.isfile(filename):
    os.remove(filename)

# Base coil.  Generate a new one upon request.
def make_coil() -> MultiPhaseCoil:
    c = MultiPhaseCoil()

    c.dia_outside = od
    c.dia_inside = id

    c.width = width
    c.spacing = spacing

    c.layers = layers
    c.via = v
    c.mount_hole = hole
    c.mount_hole_pattern_diameter = c.dia_outside + hole.radius + 5

    c.text_pattern_diameter = c.mount_hole_pattern_diameter + hole.radius + 5

    return c


def code_append(c: MultiPhaseCoil) -> None:

    with open(filename, "a") as fp:
        fp.write(c.ToKiCad())


# ----------------------------------------------------------------------------
# 2-phase 4 quadrant
# ----------------------------------------------------------------------------
c = make_coil()
c.name = "2-Phase, M=2"

n = 4
nets = np.arange(1, 1 + n, dtype=int)
c.nets = list(nets)
c.multiplicity = 1

# Coil mounting hole angles
delta_th = 360 / len(c.nets)
th = np.arange(0, 360, delta_th) + delta_th / 2 / c.multiplicity
c.mount_hole_pattern_angles = th * np.pi / 180
print(f"{c.name}: angles{th}, nets: {c.nets} ")
c.Generate()

c.Translate(x, y)

code_append(c)

# ----------------------------------------------------------------------------
# 2-phase 4-quadrant +/-
# ----------------------------------------------------------------------------
c = make_coil()
c.name = "2-Phase, M=4"

n = 4
n1 = nets.max() + 1
nets = np.arange(n1, n1 + n, dtype=int)
c.nets = list(nets)
c.multiplicity = 2

# Coil mounting hole angles
delta_th = 360 / len(c.nets)
th = np.arange(0, 360, delta_th) + delta_th / 2 / c.multiplicity
c.mount_hole_pattern_angles = th * np.pi / 180
print(f"{c.name}: angles{th}, nets: {c.nets} ")
c.Generate()

x += dx
c.Translate(x, y)

code_append(c)


# ----------------------------------------------------------------------------
# 3-phase
# ----------------------------------------------------------------------------
c = None
c = make_coil()
c.name = "3-Phase, M=1"

n = 3
n1 = nets.max() + 1
nets = np.arange(n1, n1 + n, dtype=int)
c.nets = list(nets)
c.multiplicity = 1

# Coil mounting hole angles
delta_th = 360 / len(c.nets)
th = np.arange(0, 360, delta_th) + delta_th / 2 / c.multiplicity
c.mount_hole_pattern_angles = th * np.pi / 180
print(f"{c.name}: angles{th}, nets: {c.nets} ")
c.Generate()

x += dx
c.Translate(x, y)

code_append(c)

# ----------------------------------------------------------------------------
# 3-phase +/-
# ----------------------------------------------------------------------------
c = None
c = make_coil()
c.name = "3-Phase, M=2"
n = 6
n1 = nets.max() + 1
nets = np.arange(n1, n1 + n, dtype=int)
c.nets = list(nets)
c.multiplicity = 1

# Coil mounting hole angles
delta_th = 360 / len(c.nets)
th = np.arange(0, 360, delta_th) + delta_th / 2 / c.multiplicity
c.mount_hole_pattern_angles = th * np.pi / 180
print(f"{c.name}: angles{th}, nets: {c.nets} ")
c.Generate()

x += dx
c.Translate(x, y)

code_append(c)
