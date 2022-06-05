# coil_3ph.py
# Three phase coil generation script.
# Inspired by: https://gist.github.com/JoanTheSpark/e5afd7081d9d1b9ad91a
#

from math import radians
from click import FileError
import numpy as np
import plotly.graph_objects as go
import configparser
from pint import UnitRegistry


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
        self._width = 0.2
        self._spacing = 0.2
        self._od = 50
        self._id = 10
        self._replication = 1

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

        da = np.diff(angles)
        idx = np.where(da > 0)[0]

        radii = radii[idx]
        angles = angles[idx, :]
        angles = angles.flatten()

        r = np.zeros((len(radii) * 2,))
        r[1::2] = radii
        r[::2] = np.flip(radii)
        r = r[: int(len(r) / 2)]

        print(r)
        print(angles)

        self._turns = 0
        self._geo = []

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

        # self.Plot()


# %%
if __name__ == "__main__":
    fn = "coil.config"

    # coil = Coil3Ph(cfgfile=fn)
    coil = Coil3Ph()
    coil._replication = 1
    coil._od = 20
    coil._id = 10
    coil._width = 0.25
    coil._spacing = 0.25
    coil.GenerateGeo()
    print(coil._geo)
    coil.Plot()
