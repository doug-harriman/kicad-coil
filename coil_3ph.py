# coil_3ph.py
# Three phase coil generation script.
# Inspired by: https://gist.github.com/JoanTheSpark/e5afd7081d9d1b9ad91a
#
# Tracks doc:
# URL: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/#_graphic_items_section:~:text=on%20the%20board.-,Tracks%20Section,-This%20section%20lists


from typing import Tuple
import numpy as np
import plotly.graph_objects as go
import configparser
from pint import UnitRegistry
import os
import uuid
import copy
import warnings

# KiCAD Python
# pip install kicad_python
# URL https://github.com/pointhi/kicad-python

# Python netlist generator:
# URL: https://skidl.readthedocs.io/en/latest/readme.html

# TODO: Add bottom layer coil.  Will need to mirror/flip numerically myself to properly align.
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
# TODO: Inject comments as "gr_text" elements on layer "User.Comments"
#       Can we create a special layer for this?
# TODO: Coil needs to capture number of turns in GenerateGeo().
# TODO: Estimate coil trace resistance.
#       * TraceLen implemented.
#       * Need to capture Copper thickness/weight
# TODO: Output turn count per coil
# TODO: Estimate coil inductance?
# TODO: Get Plotly plotting working again.
# TODO: Output code for FEMM model generation.
# TODO: Ability to read/write PCB file directly.
#       * Paren parsing.
#       * Delete elements by UUID or group name.
#       * Add elements to end.
# TODO: Completeness: Property getter/setters for Segment and Arc
# TODO: Completeness: Property getter/setters for SectorCoil


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        """
        Returns x coordinate.
        """
        return self._x

    @x.setter
    def x(self, value: float = 0.0):
        """
        Set X coordinate.

        Args:
            value (float, optional): X coordinate. Defaults to 0.0.
        """

        value = float(value)
        self._x = value

    @property
    def y(self) -> float:
        """
        Returns y coordinate.
        """
        return self._y

    @y.setter
    def y(self, value: float = 0.0):
        """
        Set Y coordinate.

        Args:
            value (float, optional): Y coordinate. Defaults to 0.0.
        """

        value = float(value)
        self._y = value

    def __repr__(self):

        return f"Point(x={self.x},y={self.y})"

    def Translate(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the Point by the given distances.
        """
        self._x += x
        self._y += y

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the point about the given x,y coordinates by the given angle in radians.
        """

        # Translate to origin
        self.Translate(-x, -y)

        # Rotate
        v = np.array([self._x, self._y])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        v = R.dot(v)

        self._x = v[0]
        self._y = v[1]

        # Translate back
        self.Translate(x, y)

    def ToKiCad(self) -> str:
        """
        Returns string representation of Point in KiCAD format.
        """

        return f"{self._x:0.6f} {self._y:0.6f}"

    def ToNumpy(self) -> Tuple:
        """
        Returns Numpy arrays for X & Y coordinates.
        """

        return np.array(self._x), np.array(self._y)


class Track:
    def __init__(self, net: int = 1):
        """
        Creates a Track base class in the given schematic net.

        Assigns UUID to the object as the 'id' property.
        """
        self._net = net
        self._id = uuid.uuid4()

    def __deepcopy__(self, memo):
        """
        Deep copy of Track class with new UUIO.
        """

        # Per:
        # https://stackoverflow.com/questions/57181829/deepcopy-override-clarification

        from copy import deepcopy

        cls = self.__class__  # Extract the class of the object
        # Create a new instance of the object based on extracted class
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():

            # Copy over attributes by copying directly or in case of complex
            # objects like lists for exaample calling the `__deepcopy()__`
            # method defined by them. Thus recursively copying the whole tree
            # of objects.
            setattr(result, k, deepcopy(v, memo))

        result._id = uuid.uuid4()

        return result

    @property
    def net(self) -> int:
        """
        Returns the net ID for the Track.
        """
        return self._net

    @net.setter
    def net(self, value: int = 1) -> None:
        """
        Sets the net ID for the Track.
        """
        self._net = int(value)

    @property
    def id(self) -> uuid:
        """
        Returns the UUID of the Track.
        """
        return self._id

    def ToKiCad(self) -> str:
        """
        Converts Track to KiCAD string.
        """

        return f"(net {self.net}) (tstamp {str(self.id)})"


class Segment(Track):
    def __init__(
        self,
        start: Point = Point(),
        end: Point = Point(0, 1),
        width: float = 0.1,
        layer: str = "F.Cu",
        net: int = 1,
    ):
        """
        Creates a linear segment Track object.
        """

        super().__init__(net)

        self._start = start
        self._end = end
        self._width = width
        self._layer = layer

    def __repr__(self):
        return self.ToKiCad()

    @property
    def layer(self) -> str:
        """
        Returns layer for Segment.

        Returns:
            str: Layer
        """

        return self._layer

    @layer.setter
    def layer(self, value: str = "F.Cu"):
        """
        Sets layer for Segment.

        Args:
            value (str, optional): Layer name. Defaults to 'F.Cu'.
        """

        value = str(value)
        self._layer = value

    @property
    def width(self) -> float:
        """
        Segment track width property getter.

        Returns:
            float: Segment track width.
        """

        return self._width

    @width.setter
    def width(self, value: float = 0.2) -> None:
        """
        Segment track width setter.

        Args:
            value (float, optional): Track width. Defaults to 0.2.
        """

        width_orig = value
        width = float(value)
        if len(width) != 1:
            raise ValueError(f"Track width must be scalar: {width_orig}")
        if width <= 0.0:
            raise ValueError(f"Track width positive: {width_orig}")

        self._width = width

    def Translate(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the Segment Track by the given distances.
        """
        self._start.Translate(x, y)
        self._end.Translate(x, y)

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the Segment about the given x,y coordinates by the given angle in radians.
        """

        self._start.Rotate(angle, x, y)
        self._end.Rotate(angle, x, y)

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts Segment to KiCAD string.
        """

        s = (
            f"{indent}"
            f"(segment "
            f"(start {self._start.ToKiCad()}) "
            f"(end {self._end.ToKiCad()}) "
            f"(width {self._width}) "
            f'(layer "{self._layer}") '
            f"{super().ToKiCad()}"
            f")"
            f"{os.linesep}"
        )
        return s

    def ToNumpy(self) -> Tuple:
        """
        Returns Numpy arrays for Segment X & Y coordinates.

        Suitable for plotting.
        """

        x1, y1 = self._start.ToNumpy()
        x2, y2 = self._end.ToNumpy()

        return np.append(x1, x2), np.append(y1, y2)

    def TraceLen(self) -> float:
        """
        Calculates Segment trace length.

        Returns:
            float: Trace length.
        """

        return np.sqrt(
            (self._start.x - self._end.x) ** 2 + (self._start.y - self._end.y) ** 2
        )


class Arc(Track):
    def __init__(
        self,
        center: Point = Point(),
        radius: float = 1.0,
        start: float = 0.0,
        end: float = np.pi,
        width: float = 0.1,
        layer: str = "F.Cu",
        net: int = 1,
    ):
        """
        Creates an Arc Track object.
        """

        super().__init__(net)

        self._center = center
        self._radius = radius
        self._start = start
        self._end = end
        self._width = width
        self._layer = layer

    def __repr__(self):
        return self.ToKiCad()

    @property
    def layer(self) -> str:
        """
        Returns layer for Arc.

        Returns:
            str: Layer
        """

        return self._layer

    @layer.setter
    def layer(self, value: str = "F.Cu"):
        """
        Sets layer for Arc.

        Args:
            value (str, optional): Layer name. Defaults to 'F.Cu'.
        """

        value = str(value)
        self._layer = value

    @property
    def width(self) -> float:
        """
        Arc track width property getter.

        Returns:
            float: Arc track width.
        """

        return self._width

    @width.setter
    def width(self, value: float = 0.2) -> None:
        """
        Arc track width setter.

        Args:
            value (float, optional): Track width. Defaults to 0.2.
        """

        width_orig = value
        width = float(value)
        if len(width) != 1:
            raise ValueError(f"Track width must be scalar: {width_orig}")
        if width <= 0.0:
            raise ValueError(f"Track width positive: {width_orig}")

        self._width = width

    def Translate(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the Arc Track by the given distances.
        """
        self._center.Translate(x, y)

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the Arc about the given x,y coordinates by the given angle in radians.
        """

        self._center.Rotate(angle, x, y)
        self._start += angle
        self._end += angle

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts Arc to KiCAD string.
        """

        pt_rad = Point(self._radius, 0)

        # start  # mid  # end
        pts = [copy.copy(pt_rad), copy.copy(pt_rad), copy.copy(pt_rad)]

        angles = [self._start, np.mean([self._start, self._end]), self._end]

        for i, pt in enumerate(pts):
            pt.Rotate(angles[i])
            pt.Translate(self._center.x, self._center.y)

        s = (
            f"{indent}"
            f"(arc "
            f"(start {pts[0].ToKiCad()}) "
            f"(mid {pts[1].ToKiCad()}) "
            f"(end {pts[2].ToKiCad()}) "
            f"(width {self._width}) "
            f'(layer "{self._layer}") '
            f"{super().ToKiCad()}"
            f")"
            f"{os.linesep}"
        )
        return s

    def ToNumpy(self, n: int = 100) -> Tuple:
        """
        Returns Numpy arrays for Arc X & Y coordinates.

        n: Number of points to represent arc.  Default = 100.

        Suitable for plotting.
        """

        d_theta = np.diff(self._end - self._start) / n

        theta = np.arange(self._start, self._end, d_theta)
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)

        return x, y

    def TraceLen(self) -> float:
        """
        Calculates Arc trace length.

        Returns:
            float: Trace length.
        """

        return self._radius * np.abs(self._start - self._end)


class Group:
    def __init__(self, members: list = None, name: str = ""):
        """
        Creates a KiCAD group from the given list of member PCB elements.

        Assigns UUID to the object as the 'id' property.
        """
        self._id = uuid.uuid4()
        self._name = name
        self._members = []

        if len(self._members) > 0:
            for member in members:
                try:
                    self.AddMember(member)
                except TypeError:
                    warnings.warn(f'Skipping member with no "id" attribute: {member}')

    def __repr__(self):
        return self.ToKiCad()

    def __deepcopy__(self, memo):
        """
        Deep copy of Group class with new UUIO.
        """

        # Per:
        # https://stackoverflow.com/questions/57181829/deepcopy-override-clarification

        from copy import deepcopy

        cls = self.__class__  # Extract the class of the object
        # Create a new instance of the object based on extracted class
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():

            # Copy over attributes by copying directly or in case of complex
            # objects like lists for exaample calling the `__deepcopy()__`
            # method defined by them. Thus recursively copying the whole tree
            # of objects.
            setattr(result, k, deepcopy(v, memo))

        result._id = uuid.uuid4()

        return result

    @property
    def id(self) -> uuid:
        """
        Returns the UUID of the Track.
        """
        return self._id

    @property
    def name(self) -> str:
        """
        Returns the name of the group.
        """
        return self._name

    @name.setter
    def name(self, value: str = ""):
        """
        Sets name of group.

        Args:
            value (str, optional): Group name. Defaults to "".

        Raises:
            TypeError: Invalid name type.

        Returns:
            None
        """

        if not isinstance(value, str):
            raise TypeError('Given name value is not of type "str".')

        self._name = value

    @property
    def members(self) -> list:
        """
        Returns the child element list.
        """
        return self._members

    def AddMember(self, member):
        """
        Adds object to list of group members.

        Args:
            member (_type_): Member object.

        Raises:
            FileExistsError: _description_
            FileExistsError: _description_
            ValueError: _description_

        Returns:
            None
        """

        # Verify child has the required method.
        if hasattr(member, "id"):
            self._members.append(member)
        else:
            raise TypeError(f'Member has no "id" attribute: {member}')

    def Translate(self, x: float = 0.0, y: float = 0.0):
        """
        Translates geometry by given offset [x,y].
        """

        # Process member list
        for g in self._members:
            g.Translate(x, y)

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the Group members about the given x,y coordinates by the given angle in radians.
        """

        # Process member list
        for g in self._members:
            g.Rotate(angle, x, y)

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts Group list and all children to KiCAD string.

        Returns:
            str: KiCAD PCB string representation of group.
        """

        indent = "  "
        eol = os.linesep
        s = ""

        # Generate S-expressions for children objects first.
        for m in self._members:
            s += m.ToKiCad(indent=indent)

        # Group Header
        s += indent + f'(group "{self._name}" (id {str(self.id)})' + eol
        s += 2 * indent + "(members" + eol

        for m in self._members:
            s += 3 * indent + f"{m.id}" + eol

        # Footer
        s += 2 * indent + ")" + eol
        s += indent + ")" + eol

        return s

    def TraceLen(self) -> float:
        """
        Calculates total trace length for all Track elements in group.
        WARNING: Assumes that all elements are connected in series.

        Returns:
            float: Trace length.
        """

        l = 0
        for m in self.members:
            if isinstance(m, Track):
                l += m.TraceLen()

        return l


class SectorCoil(Group):
    """
    Class for creating circular sector coil geometry for KiCAD.
    """

    def __init__(self):
        """
        SectorCoil constructor.
        """

        super().__init__()

        self._dia_outside = 20
        self._dia_inside = 5
        self._center = Point()
        self._angle = np.pi / 2

        self._layer = "F.Cu"
        self._width = 0.2
        self._spacing = 0.2
        self._net = 1

        # Initial generation of geometry.
        self.Generate()

    @property
    def net(self) -> int:
        """
        Net ID property getter.
        """
        return self._net

    @property
    def layer(self) -> str:
        """
        Returns layer for SectorCoil.

        Returns:
            str: Layer
        """

        return self._layer

    @layer.setter
    def layer(self, value: str = "F.Cu"):
        """
        Sets layer for SectorCoil.

        Args:
            value (str, optional): Layer name. Defaults to 'F.Cu'.
        """

        value = str(value)
        self._layer = value

    @property
    def width(self) -> float:
        """
        Sector track width property getter.

        Returns:
            float: Sector track width.
        """

        return self._width

    @width.setter
    def width(self, value: float = 0.2) -> None:
        """
        Sector track width setter.

        Args:
            value (float, optional): Track width. Defaults to 0.2.
        """

        width_orig = value
        width = float(value)
        if len(width) != 1:
            raise ValueError(f"Track width must be scalar: {width_orig}")
        if width <= 0.0:
            raise ValueError(f"Track width positive: {width_orig}")

        self._width = width

    @net.setter
    def net(self, value: int = 1) -> None:
        """
        Net property setter.

        Args:
            value (int, optional): ID of net. Defaults to 1.
        """

        net = int(value)
        if net < 1:
            raise ValueError(f"Net ID < 1: {value}")

        self._net = net

        # For regeneration of geometry
        self.Generate()

    def Generate(self):
        """
        Generates coil geometry segments.

        All segments define the midpoints of the track.

        """

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
        rad_out = self._dia_outside / 2 + self._width / 2
        rad_pitch = self._width + self._spacing

        radii = np.arange(rad_out, 0, -rad_pitch)
        radii = radii[np.where(radii > self._dia_inside / 2)]

        angles = np.array([[0, self._angle - angle_change(radii[0])]])
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

        # First segment is the entry segment from the connection trace.
        seg = Segment(
            Point(rad_out + self._width + self._spacing, 0),
            Point(r[0], 0),
            width=self._width,
            layer=self.layer,
            net=self.net,
        )
        self.AddMember(seg)

        i = 0
        while i + 1 < len(r):

            # Arc
            arc = Arc(
                center=Point(0, 0),
                radius=r[i],
                start=angles[i],
                end=angles[i + 1],
                width=self._width,
                layer=self.layer,
                net=self.net,
            )
            self.AddMember(arc)

            # Line
            seg = Segment(
                Point(r[i] * np.cos(angles[i + 1]), r[i] * np.sin(angles[i + 1])),
                Point(
                    r[i + 1] * np.cos(angles[i + 1]), r[i + 1] * np.sin(angles[i + 1])
                ),
                width=self._width,
                layer=self.layer,
                net=self.net,
            )
            self.AddMember(seg)

            i += 1

        # Line to center of coil for via
        angle_1 = angles[i]
        angle_2 = angles[i : i + 2].mean()
        arc = Arc(
            center=Point(0, 0),
            radius=r[i],
            start=angle_1,
            end=angle_2,
            width=self._width,
            layer=self.layer,
            net=self.net,
        )
        self.AddMember(arc)

        # Reprocess geometry adding center offsets.
        self.Translate(self._center.x, self._center.y)


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

    def Translate(self, delta=Point):
        """
        Translates geometry by given offset [x,y].
        """

        # Process geometry list
        for g in self._geo:
            g.Translate(delta.x, delta.y)

    def ToKiCad(self, filename: str = None, to_stdout: bool = False):
        """
        Converts geometry to KiCAD PCB format.
        """

        if self._geo is None:
            self.GenerateGeo()

        eol = os.linesep
        s = ""

        # Process all geometry data
        for g in self._geo:
            s += g.ToKiCad() + eol
        s += eol

        # Add that geo to a group.
        g = Group(members=self._geo, name="PhA")
        s += g.ToKiCad()

        # Output options
        if filename is not None:
            with open(filename, "w") as fp:
                fp.write(s)

        if to_stdout:
            print(s)

        return s


#%%
if __name__ == "__main__":

    if False:  # Base geometry elements
        seg = Segment(Point(0, 0), Point(10, 10), width=0.25)
        seg.Translate(2, 3)
        print(seg.ToKiCad())

        arc = Arc(Point(0, 0), radius=5, start=0, end=np.pi / 2, width=0.123)
        arc.Rotate(np.pi / 2)
        print(arc.ToKiCad())

    if True:
        g = Group(name="Quadrants")

        c1 = SectorCoil()
        c1.name = "SE"
        c1.Generate()
        # c.Translate(100, 80)
        g.AddMember(c1)

        angles = [(1, "NE"), (2, "NW"), (3, "SW")]
        for angle in angles:
            rot = angle[0] * -np.pi / 2

            c = copy.deepcopy(c1)
            c.name = angle[1]
            c.net = angle[0] + 1
            c.Rotate(rot)
            g.AddMember(c)

        # c2 = SectorCoil()
        # c2.name = "NE"
        # c2.Generate()
        # c2.Rotate(-np.pi / 2)
        # g.AddMember(c2)
        g.Translate(120, 85)

        s = g.ToKiCad()

        print(s)

    # fn = "coil.config"
    # coil = Coil3Ph(cfgfile=fn)

    if False:  # 3phase coil
        coil = Coil3Ph()
        coil._replication = 1
        coil._od = 20
        coil._id = 4
        coil._width = 0.25
        coil._spacing = 0.25
        coil._layers = ["F.Cu"]
        coil._center = Point(120, 120)

        coil.GenerateGeo()
        # coil.Plot()
        coil.ToKiCad(to_stdout=True)
