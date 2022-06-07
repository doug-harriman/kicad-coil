# coil_3ph.py
# Three phase coil generation script.
# Inspired by: https://gist.github.com/JoanTheSpark/e5afd7081d9d1b9ad91a
#
# Tracks doc:
# URL: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/#_graphic_items_section:~:text=on%20the%20board.-,Tracks%20Section,-This%20section%20lists


import numpy as np
import plotly.graph_objects as go
import configparser
from pint import UnitRegistry
import os

# KiCAD Python
# pip install kicad_python
# URL https://github.com/pointhi/kicad-python

# Python netlist generator:
# URL: https://skidl.readthedocs.io/en/latest/readme.html

# TODO: Classes for elements (lines, arcs, text) so that I can provide methods to flip, rotate, move, etc.
# TODO: Convert data structure from tuple to dict for more flexibility.
# TODO: Add bottom layer coil.  Will need to mirror/flip numerically myself.
# TODO: Add via to bottom layer
# TODO: Add B & C phases, rotated.
# TODO: Add text label "gr_text" to each phase
# ex:  (gr_text "PhA+" (at 116.84 91.44 45) (layer "F.SilkS") (effects (font (size 1.5 1.5) (thickness 0.3)))
# TODO: Add coil on bottom side. How to connect?
# TODO: Add optional mounting holes: count & radius
# Video link to custom hole geometries:
# URL: https://youtu.be/5Be7XOMmPQE?t=1592
# Video link to non-plated through holes:
# URL: https://youtu.be/5Be7XOMmPQE?t=1653
# TODO: Add optional center hole: radius
# TODO: Use 'tstamp' UUID to locate elements & replace?
3 URL: https://dev-docs.kicad.org/en/file-formats/sexpr-intro/#_text_effects:~:text=the%20key%20attribute.-,Universally%20Unique%20Identifier,-The%20uuid%20token
# TODO: Inject comments as "gr_text" elements on layer "User.Comments"
#       Can we create a special layer for this?
# TODO: Create items as a group.  Looks simple if have UUID's.
# UUID's:
# >> import uuid
# >> id = uuid.uuid4()
# TODO: Estimate coil trace resistance
# TODO: Output turn count per coil
# TODO: Estimate coil inductance?  
# TODO: Output code for FEMM model generation.  

class Coil3Ph:
    """
    3-phase coil geometry generator for KiCad.
    """

    def __init__(self, cfgfile: str = None):
        """
        Constructor.
        """

        # Load file if specified
        if cfgfile is not None:
            self.Load(cfgfile=cfgfile)

        # Defaults
        self._units = "mm"
        self._layers = ["F.Cu", "B.Cu"]
        self._net_phA = 1
        self._net_phB = 1
        self._net_phC = 1
        self._width = 0.2
        self._spacing = 0.2
        self._od = 50
        self._id = 10
        self._replication = 1
        self._center = np.array([0, 0])

        self._geo = None

        # Set up handling of mils
        ureg = UnitRegistry()
        ureg.define("mil = 0.001 in")
        self.Q = ureg.Quantity

    def Load(self, cfgfile: str = None):
        if cfgfile is None:
            raise FileExistsError("No configuration file specified.")

        config = configparser.ConfigParser()
        res = config.read(cfgfile)
        if len(res) == 0:
            raise FileExistsError(
                f"Config file not found or contents invalid: {cfgfile}"
            )  # noqa: E501

    def Plot(self):
        """
        Generates Plotly plot of coil.
        """

        if self._geo is None:
            self.GenerateGeo()

        # Create figure
        fig = go.Figure()

        def arc(radius, angles, n=100):
            """
            Plot an arc.
            """

            d_theta = np.diff(angles) / n

            theta = np.arange(angles[0], angles[1], d_theta)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            return x, y

        # Single coil
        x = np.array([])
        y = np.array([])
        for seg in self._geo:
            if seg[0] == "line":
                x = np.append(x, seg[1][0])
                x = np.append(x, seg[1][2])
                y = np.append(y, seg[1][1])
                y = np.append(y, seg[1][3])

            elif seg[0] == "arc":
                # Pretty arc.
                radius = np.sqrt(seg[1][0] ** 2 + seg[1][1] ** 2)
                a1 = np.arctan2(seg[1][1], seg[1][0])
                a2 = np.arctan2(seg[1][5], seg[1][4])
                arc_x, arc_y = arc(radius, [a1, a2])

                x = np.append(x, arc_x)
                y = np.append(y, arc_y)

            else:
                raise ValueError(f"Unsupported geometry type: {seg[0]}")

        # Generate the plot
        trc = go.Scatter(x=x, y=y)
        fig.add_trace(trc)

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        fig.show()

    def GenerateGeo(self):
        """
        Generates coil geometry segments.

        All segments define the midpoints of the track.

        """

        # Angle per phase section.
        sections = 3 * self._replication
        angle_section = 360 / sections  # degrees
        angle_section *= np.pi / 180  # radians

        def angle_change(radius):
            """
            Calculates the change in angle based on current OD
            and track geometry.
            """

            # Effective arc length based on track geometry
            len_arc = self._spacing + self._width
            len_arc *= np.pi / 2

            # Angle to get that arc at the current radius
            theta = len_arc / radius

            return theta

        # OD is started outside of coil.
        # First loop will have outside edge on the OD
        rad_out = self._od / 2 + self._width / 2
        rad_pitch = self._width + self._spacing

        radii = np.arange(rad_out, 0, -rad_pitch)
        radii = radii[np.where(radii > self._id / 2)]

        angles = np.array([[0, angle_section - angle_change(radii[0])]])
        for i, radius in enumerate(radii):
            if i == 0:
                continue

            delta = angle_change(radius)
            angles_new = angles[i - 1, :] + np.array([1, -1]) * delta

            angles = np.vstack((angles, angles_new))

        # At some the radius gets too small, and lines may start crossing.
        # Keep the sector angle > 0
        da = np.diff(angles)
        idx = np.where(da > 0)[0]

        # Generate list of angles to process through
        radii = radii[idx]
        angles = angles[idx, :]
        angles = angles.flatten()

        # Generate list of radii to process through
        r = np.zeros((len(radii) * 2,))
        r[1::2] = radii
        r[::2] = np.flip(radii)
        r = r[: int(len(r) / 2)]

        self._turns = 0
        self._geo = []

        # First segment is the entry segment from the connection trace.
        line = [rad_out + self._width + self._spacing, 0, r[0], 0]  # Inside radius
        self._geo.append(("line", line))

        i = 0
        while i + 1 < len(r):

            # Arc
            arc = [
                r[i] * np.cos(angles[i]),
                r[i] * np.sin(angles[i]),
                r[i] * np.cos(angles[i : i + 2].mean()),
                r[i] * np.sin(angles[i : i + 2].mean()),
                r[i] * np.cos(angles[i + 1]),
                r[i] * np.sin(angles[i + 1]),
            ]
            self._geo.append(("arc", arc))

            # Line
            line = [
                r[i] * np.cos(angles[i + 1]),
                r[i] * np.sin(angles[i + 1]),
                r[i + 1] * np.cos(angles[i + 1]),
                r[i + 1] * np.sin(angles[i + 1]),
            ]
            self._geo.append(("line", line))

            i += 1

        # Line to center of coil for via
        angle_1 = angles[i]
        angle_2 = angles[i : i + 2].mean()
        arc = [
            r[i] * np.cos(angle_1),
            r[i] * np.sin(angle_1),
            r[i] * np.cos(np.mean([angle_1, angle_2])),
            r[i] * np.sin(np.mean([angle_1, angle_2])),
            r[i] * np.cos(angle_2),
            r[i] * np.sin(angle_2),
        ]
        self._geo.append(("arc", arc))

        # Reprocess geometry adding center offsets.
        self._geo = self.Translate(self._geo, self._center)

    def Translate(self, geo, delta):
        """
        Translates geometry by given offset [x,y].
        """

        # Process geometry list
        for seg in geo:
            seg[1][0::2] += delta[0]
            seg[1][1::2] += delta[1]

        return geo

    def ToKiCad(self, filename: str = None, to_stdout: bool = False):
        """
        Converts geometry to KiCAD PCB format.
        """

        if self._geo is None:
            self.GenerateGeo()

        eol = os.linesep
        s = ""

        # TODO: Update these when ready to handle them
        layer = self._layers[0]
        net = self._net_phA

        # Process all geometry data
        for seg in self._geo:
            if seg[0] == "line":
                segment = (
                    "  (segment "
                    f"(start {seg[1][0]:0.5f} {seg[1][1]:0.5f}) "
                    f"(end {seg[1][2]:0.5f} {seg[1][3]:0.5f}) "
                    f"(width {self._width}) "
                    f'(layer "{layer}") '
                    f"(net {net}) )"
                )

                s += segment + eol

            elif seg[0] == "arc":
                # Note: There is an error in the KiCAD documentation for the arc.
                #       The width is a single value, not X & Y widths.
                # URL: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/#_header_section:~:text=the%20line%20object.-,Track%20Arc,-The%20arc%20token
                arc = (
                    "  (arc "
                    f"(start {seg[1][0]:0.5f} {seg[1][1]:0.5f}) "
                    f"(mid {seg[1][2]:0.5f} {seg[1][3]:0.5f}) "
                    f"(end {seg[1][4]:0.5f} {seg[1][5]:0.5f}) "
                    f"(width {self._width}) "
                    f'(layer "{layer}") '
                    f"(net {net}) )"
                )

                s += arc + eol

            else:
                raise ValueError(f"Unsupported geometry type: {seg[0]}")

        s += eol

        # Output options
        if filename is not None:
            with open(filename, "w") as fp:
                fp.write(s)

        if to_stdout:
            print(s)

        return s


#%%
if __name__ == "__main__":
    fn = "coil.config"

    # coil = Coil3Ph(cfgfile=fn)
    coil = Coil3Ph()
    coil._replication = 1
    coil._od = 20
    coil._id = 4
    coil._width = 0.25
    coil._spacing = 0.25
    coil._layers = ["F.Cu"]
    coil._center = np.array([120, 120])

    coil.GenerateGeo()
    # coil.Plot()
    coil.ToKiCad(to_stdout=True)
