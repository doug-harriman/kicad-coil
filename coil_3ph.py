# coil_3ph.py
# Three phase coil generation script.
# Inspired by: https://gist.github.com/JoanTheSpark/e5afd7081d9d1b9ad91a
#
# Tracks doc:
# URL: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/#_graphic_items_section:~:text=on%20the%20board.-,Tracks%20Section,-This%20section%20lists


import copy
import os
import uuid
import warnings
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from pint import UnitRegistry

# KiCAD Python
# pip install kicad_python
# URL https://github.com/pointhi/kicad-python

# Python netlist generator:
# URL: https://skidl.readthedocs.io/en/latest/readme.html

# TODO: Add optional mounting holes: count & radius
# Video link to custom hole geometries:
# URL: https://youtu.be/5Be7XOMmPQE?t=1592
# Video link to non-plated through holes:
# URL: https://youtu.be/5Be7XOMmPQE?t=1653
# TODO: Add optional center hole: radius
# TODO: Coil needs to capture number of turns in Generate().
# TODO: Estimate coil trace resistance.
#       * TraceLen implemented.
#       * Need to capture Copper thickness/weight
# TODO: Estimate coil inductance?
# TODO: Get Plotly plotting working again.
# TODO: Output code for FEMM model generation.
# TODO: Ability to read/write PCB file directly.
#       * Paren parsing.
#       * Delete elements by UUID or group name.
#       * Add elements to end.
# TODO: Completeness: Property getter/setters for SectorCoil
# TODO: Toml config files for reading coil configs.
#       >> pip install tomli
#       URL: https://github.com/hukkin/tomli


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

    def __str__(self):
        return f"({self.x:0.6f},{self.y:0.6f})"

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

    def __repr__(self) -> str:
        return f"Net:{self.net}, ID:{str(self.id)}"

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


class Via(Track):
    """
    Via object.
    """

    def __init__(
        self,
        position: Point = None,
        size: float = 0.8,
        drill: float = 0.4,
        layers: str = ["F.Cu", "B.Cu"],
        net: int = 1,
    ):
        """
        Creates a Via Track object.
        """

        super().__init__(net)

        self.layers = layers
        self.position = position
        self.size = size
        self.drill = drill

    def __repr__(self) -> str:

        s = (
            f"Via:({self.position.x},{self.position.y}),"
            f" Layers:{self.layers},"
            f" {super().__repr__()}"
        )

        return s

    @property
    def position(self) -> Point:
        """
        Via position property getter.

        Returns:
            Point: Via center position.
        """

        return self._position

    @position.setter
    def position(self, value: Point) -> None:
        """
        Via center position getter.

        Args:
            value (Point): Center position.

        Raises:
            ValueError: Invalid position value type.
        """

        if value is None:
            value = Point()

        if not isinstance(value, Point):
            return TypeError(f"Invalid position value type: {type(value)}")

        self._position = value

    @property
    def layers(self) -> list:
        """
        Returns Via layer list.

        Returns:
            list: List of layer names.
        """

        return self._layers

    @layers.setter
    def layers(self, value: list = None) -> None:
        """
        Via layers property setter.

        Args:
            value (list): List of layer names.
                          Note: Names are not checked for validity.
        """

        if value is None:
            raise ValueError("No layers specified.")

        if isinstance(value, str):
            value = [value]

        # Make sure all are strings
        for v in value:
            if not isinstance(v, str):
                raise TypeError(f"Invalid layer value type: {type(v)}:{v}")

        self._layers = value

    @property
    def size(self) -> float:
        """
        Via size property getter.

        Returns:
            float: Via size.
        """

        return self._size

    @size.setter
    def size(self, value: float = 0.8) -> None:
        """
        Via size setter.

        Args:
            value (float, optional): Via size. Defaults to 0.8.
        """

        size_orig = value
        size = float(value)
        if size <= 0.0:
            raise ValueError(f"Vias size must be positive: {size_orig}")

        self._size = size

    @property
    def drill(self) -> float:
        """
        Via drill property getter.

        Returns:
            float: Via drill.
        """

        return self._drill

    @drill.setter
    def drill(self, value: float = 0.4) -> None:
        """
        Via drill setter.

        Args:
            value (float, optional): Via drill. Defaults to 0.4.
        """

        drill_orig = value
        drill = float(value)
        if drill <= 0.0:
            raise ValueError(f"Vias drill must be positive: {drill_orig}")

        self._drill = drill

    def Translate(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the Via by the given distances.
        """
        self.position.Translate(x, y)

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the Via about the given x,y coordinates by the given angle in radians.
        """

        self.center.Rotate(angle, x, y)

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts Via to KiCAD string.
        """

        # Convert layer list to str
        layerstr = ""
        for layer in self.layers:
            layerstr += f' "{layer}"'

        s = (
            f"{indent}"
            f"(via "
            f"(at {self.position.ToKiCad()}) "
            f"(size {self.size}) "
            f"(drill {self.drill}) "
            f"(layers{layerstr})"
            f"{super().ToKiCad()}"
            f")"
            f"{os.linesep}"
        )
        return s


class Segment(Track):
    def __init__(
        self,
        start: Point = None,
        end: Point = None,
        width: float = 0.1,
        layer: str = "F.Cu",
        net: int = 1,
    ):
        """
        Creates a linear segment Track object.
        """

        super().__init__(net)

        self.start = start
        self.end = end
        self.width = width
        self.layer = layer

    def __repr__(self):
        s = (
            f"Segment:"
            f"{self._start},"
            f"{self._end},"
            f'Layer:"{self._layer}", '
            f"{super().__repr__()}"
        )
        return s

    @property
    def start(self) -> Point:
        """
        Segment start point property getter.

        Returns:
            Point: Start point.
        """
        return self._start

    @start.setter
    def start(self, value: Point = None) -> None:
        """
        Segment start point property setter.
        NOTE: Creates a point of the Point.

        Args:
            value (Point): Start point.

        Raises:
            TypeError: Incorrect type provided for value.
        """

        if value is None:
            value = Point(0, 1)
        if not isinstance(value, Point):
            raise TypeError(f"Expected value of type point, not: {type(value)}")
        self._start = value  # Have our own point.

    @property
    def end(self) -> Point:
        """
        Segment end point property getter.

        Returns:
            Point: end point.
        """
        return self._end

    @start.setter
    def end(self, value: Point = None) -> None:
        """
        Segment end point property setter.

        Args:
            value (Point): End point.

        Raises:
            TypeError: Incorrect type provided for value.
        """

        if value is None:
            value = Point(0, 1)
        if not isinstance(value, Point):
            raise TypeError(f"Expected value of type point, not: {type(value)}")
        self._end = value  # Have our own point.

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

    def ChangeSideFlip(self):
        """
        Flips geometry for placing on opposite side.
        """

        # Just flip start & end
        temp = copy.deepcopy(self._end)
        self._end = copy.deepcopy(self._start)
        self._start = temp

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
        radius: float = 10.0,
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

        s = (
            f"Arc:"
            f"{self._center},"
            f"Rad:{self._radius:0.3f}, "
            f"StartAng:{self._start:0.3f}, "
            f"EndAng:{self._end:0.3f}, "
            f'Layer:"{self._layer}", '
            f"{super().__repr__()}"
        )
        return s

    @property
    def center(self) -> Point:
        """
        Segment center point property getter.

        Returns:
            Point: center point.
        """
        return self._center

    @center.setter
    def center(self, value: Point = None) -> None:
        """
        Segment center point property setter.
        NOTE: Creates a point of the Point.

        Args:
            value (Point): center point.

        Raises:
            TypeError: Incorrect type provided for value.
        """

        if value is None:
            value = Point(0, 1)
        if not isinstance(value, Point):
            raise TypeError(f"Expected value of type point, not: {type(value)}")
        self._center = value  # Have our own point.

    @property
    def start(self) -> float:
        """
        Arc start point property getter.

        Returns:
            Point: Start point.
        """
        return self._start

    @start.setter
    def start(self, value: float = 0) -> None:
        """
        Arc start angle property setter.

        Args:
            value (Point): Start angle.

        Raises:
            TypeError: Incorrect type provided for value.
        """

        value = float(value)
        self._start = value

    @property
    def end(self) -> float:
        """
        Arc end point property getter.

        Returns:
            Point: end point.
        """
        return self._end

    @end.setter
    def end(self, value: float = 0) -> None:
        """
        Arc end angle property setter.

        Args:
            value (Point): end angle.

        Raises:
            TypeError: Incorrect type provided for value.
        """

        value = float(value)
        self._end = value

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
    def radius(self) -> float:
        """
        Arc track radius property getter.

        Returns:
            float: Arc track radius.
        """

        return self._radius

    @radius.setter
    def radius(self, value: float = 10) -> None:
        """
        Arc track radius setter.

        Args:
            value (float, optional): Track radius. Defaults to 10.
        """

        radius_orig = value
        radius = float(value)
        if radius <= 0.0:
            raise ValueError(f"Track radius must be positive: {radius_orig}")

        self._radius = radius

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
        if width <= 0.0:
            raise ValueError(f"Track width must be positive: {width_orig}")

        self._width = width

    @property
    def pointstart(self) -> Point:
        """
        Start Point for Arc. Read only.

        Returns:
            Point: 2D start point of Arc.
        """

        pt = Point(self._radius, 0)
        pt.Rotate(self._start)
        pt.Translate(self._center.x, self._center.y)

        return pt

    @property
    def pointmid(self) -> Point:
        """
        Mid Point for Arc. Read only.

        Returns:
            Point: 2D mid point of Arc.
        """

        pt = Point(self._radius, 0)
        pt.Rotate(self.mid)
        pt.Translate(self._center.x, self._center.y)

        return pt

    @property
    def pointend(self) -> Point:
        """
        End Point for Arc. Read only.

        Returns:
            Point: 2D end point of Arc.
        """

        pt = Point(self._radius, 0)
        pt.Rotate(self._end)
        pt.Translate(self._center.x, self._center.y)

        return pt

    @property
    def mid(self) -> float:
        """
        Arc mid point angle property getter.  Read only.

        Returns:
            float: Mid-point angle
        """

        return np.mean([self._start, self._end])

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

    def ChangeSideFlip(self):
        """
        Flips geometry for placing on opposite side.
        """

        # Just flip start & end
        temp = copy.deepcopy(self._end)
        self._end = copy.deepcopy(self._start)
        self._start = temp

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts Arc to KiCAD string.
        """

        s = (
            f"{indent}"
            f"(arc "
            f"(start {self.pointstart.ToKiCad()}) "
            f"(mid {self.pointmid.ToKiCad()}) "
            f"(end {self.pointend.ToKiCad()}) "
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


class GrText:
    """
    Text graphics object.
    """

    def __init__(
        self,
        position: Point = None,
        text: str = "",
        layer: str = "F.SilkS",
        angle: float = 0.0,
        size: float = 1.5,
        mirror: bool = False,
    ):

        self._id = uuid.uuid4()
        if position is None:
            self.position = Point()
        else:
            self.position = position
        self.layer = layer
        self.text = text
        self.angle = angle
        self.size = size
        self._mirror = bool(mirror)

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

    def __str__(self) -> str:

        return self.text

    def __repr__(self) -> str:

        mirrorstr = ""
        if self.mirror:
            mirrorstr = " Mirrored, "

        s = (
            f"GrText:{self.position}, "
            f'"{self.text}", '
            f"{mirrorstr}"
            f"Angle:{self.angle:0.3f}, "
            f'Layer:"{self.layer}", '
            f"ID:{str(self.id)}"
        )

        return s

    @property
    def text(self) -> str:
        """
        GrText text property getter.

        Returns:
            str: Text to display.
        """

        return self._text

    @text.setter
    def text(self, value: str = "") -> None:
        """
        GrText text property setter.

        Args:
            value (str): Text to display. Default="".
        """

        if not isinstance(value, str):
            raise TypeError(f"Invalid data type passed: {type(value)}")

        self._text = value

    @property
    def id(self) -> uuid:
        """
        Returns the UUID of the GrText.
        """
        return self._id

    @property
    def position(self) -> Point:
        """
        GrText position property getter.

        Returns:
            Point: GrText center position.
        """

        return self._position

    @position.setter
    def position(self, value: Point = Point()) -> None:
        """
        GrText center position getter.

        Args:
            value (Point): Center position.

        Raises:
            ValueError: Invalid position value type.
        """

        if not isinstance(value, Point):
            return TypeError(f"Invalid position value type: {type(value)}")

        self._position = value

    @property
    def angle(self) -> float:
        """
        GrText angle property getter.

        Returns:
            float: track angle.
        """

        return self._angle

    @angle.setter
    def angle(self, value: float = 0):
        """
        GrText angle getter.

        Args:
            value (float, optional): GrText rotation angle. Defaults to 0.
        """

        angle = float(value)
        self._angle = angle

    @property
    def layer(self) -> str:
        """
        Returns layer for GrText.

        Returns:
            str: Layer
        """

        return self._layer

    @layer.setter
    def layer(self, value: str = "F.SilkS"):
        """
        Sets layer for GrText.

        Args:
            value (str, optional): Layer name. Defaults to 'F.SilkS'.
        """

        value = str(value)
        self._layer = value

    @property
    def mirror(self) -> bool:
        """
        Mirror property getter.
        NOTE: Use ChangeSideFlip method to mirror text.

        Returns:
            bool: True if text is mirrored.
        """

        return self._mirror

    @property
    def size(self) -> float:
        """
        GrText font size property getter.
        Note: X & Y sizes are the same.

        Returns:
            float: GrText size.
        """

        return self._size

    @size.setter
    def size(self, value: float = 1.5) -> None:
        """
        GrText font size setter.

        Args:
            value (float, optional): GrText size. Defaults to 1.5.
        """

        size_orig = value
        size = float(value)
        if size <= 0.0:
            raise ValueError(f"GrText size must be positive: {size_orig}")

        self._size = size

    def Translate(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the GrText by the given distances.
        """
        self.position.Translate(x, y)

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the GrText about the given x,y coordinates by the given angle in radians.
        """
        self.angle -= angle
        self.position.Rotate(angle, x, y)

    def ChangeSideFlip(self):
        """
        Flips text for placing on opposite side.
        """

        self._mirror = not self._mirror

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts GrText to KiCAD string.
        """

        mirrorstr = ""
        if self.mirror:
            mirrorstr = f"(justify mirror)"

        s = (
            f"{indent}"
            f'(gr_text "{self.text}" '
            f"(at {self.position.ToKiCad()} {self.angle*180/np.pi:0.3f}) "  # GrText angle in deg
            f'(layer "{self._layer}") '
            f"(tstamp {str(self.id)})"
            f"{os.linesep}"
            f"{indent*2}"
            f"(effects (font (size {self.size:0.3f} {self.size:0.3f}) (thickness 0.3)) {mirrorstr})"
            f"{os.linesep}"
            f"{indent})"
            f"{os.linesep}"
        )
        return s


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

        result._id = uuid.uuid4()  # assign new uique ID.

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

    def ChangeSideFlip(self):
        """
        Flips Group memeber individual geometry for placing on opposite side.
        NOTE: Does not flip as a group.  I.e. each element will end up under
              where it was, not at the opposite side of the group.
        """

        # Process member list
        for g in self._members:
            g.ChangeSideFlip()

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

        tracklen = 0
        for m in self.members:
            if isinstance(m, Track):
                tracklen += m.TraceLen()

        return tracklen


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

        # If we have already generated geometry,
        # move all geometry elements to the new layer.
        if len(self.members) > 0:
            for el in self.members:
                try:
                    el.layer = self.layer
                except:
                    warnings.warn(
                        f'Unable to set layer "{self.layer}" for element: {el}'
                    )

    @property
    def net(self) -> int:
        """
        Net ID property getter.
        """
        return self._net

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
        if width <= 0.0:
            raise ValueError(f"Track width positive: {width_orig}")

        self._width = width

    @property
    def spacing(self) -> float:
        """
        Track spacing property getter.

        Returns:
            float: track spacing.
        """

        return self._spacing

    @spacing.setter
    def spacing(self, value: float = 0.2):
        """
        Track spacing getter.

        Args:
            value (float, optional): Track spacing. Defaults to 0.2.
        """

        spacing_orig = value
        spacing = float(value)
        if spacing <= 0.0:
            raise ValueError(f"Track spacing positive: {spacing_orig}")

        self._spacing = spacing

    @property
    def dia_outside(self) -> float:
        """
        Track dia_outside property getter.

        Returns:
            float: track dia_outside.
        """

        return self._dia_outside

    @dia_outside.setter
    def dia_outside(self, value: float = 30):
        """
        Track dia_outside getter.

        Args:
            value (float, optional): Track dia_outside. Defaults to 30.
        """

        dia_outside_orig = value
        dia_outside = float(value)
        if dia_outside <= 0.0:
            raise ValueError(f"Track dia_outside positive: {dia_outside_orig}")

        self._dia_outside = dia_outside

    @property
    def dia_inside(self) -> float:
        """
        Track dia_inside property getter.

        Returns:
            float: track dia_inside.
        """

        return self._dia_inside

    @dia_inside.setter
    def dia_inside(self, value: float = 10):
        """
        Track dia_inside getter.

        Args:
            value (float, optional): Track dia_intside. Defaults to 10.
        """

        dia_inside_orig = value
        dia_inside = float(value)
        if dia_inside <= 0.0:
            raise ValueError(f"Track dia_inside positive: {dia_inside_orig}")

        self._dia_inside = dia_inside

    @property
    def angle(self) -> float:
        """
        Track angle property getter.

        Returns:
            float: track angle.
        """

        return self._angle

    @angle.setter
    def angle(self, value: float = np.pi / 2):
        """
        Track angle getter.

        Args:
            value (float, optional): Track dia_intside. Defaults to pi/2.
        """

        angle_orig = value
        angle = float(value)
        if angle <= 0.0:
            raise ValueError(f"Track angle positive: {angle_orig}")

        self._angle = angle

    def Generate(self):
        """
        Generates coil geometry segments.

        All segments define the midpoints of the track.

        """

        self._members = []

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


class MultiPhaseCoil(Group):
    """
    Multiple phase coil geometry generator for KiCad.
    WARNING: All distance/length units assumed to be mm.
    """

    def __init__(self):
        """
        Constructor.
        """

        super().__init__()

        # Defaults
        self._layers = ["F.Cu"]
        self._nets = [1, 2, 3]
        self._multiplicity = 1

        # Base coil info.
        # We'll treat this like a pseudo-base class and store
        # coil parameters directly into the coil.
        self._coil = SectorCoil()

    @property
    def layers(self) -> list:
        """
        Returns coil layer list.

        Returns:
            list: List of layer names.
        """

        return self._layers

    @layers.setter
    def layers(self, value: list = None) -> None:
        """
        Layers property setter.

        Args:
            value (list): List of layer names.
                          Note: Names are not checked for validity.
        """

        if value is None:
            raise ValueError("No layers specified.")

        if isinstance(value, str):
            value = [value]

        # Make sure all are strings
        for v in value:
            if not isinstance(v, str):
                raise TypeError(f"Invalid layer value type: {type(v)}:{v}")

        # Support limited to 2-layer boards currently.
        # For layer count > 2, need to:
        # * Keep flipping the geometry.
        # * Place vias only between adjacent layers.
        # * Silkscreen text only on front and back layers.
        if len(value) > 2:
            raise ValueError("MultiPhaseCoil only supports 2 layer boards.")

        self._layers = value

    @property
    def nets(self) -> list:
        """
        Returns list of nets in multi-phase coil.

        Returns:
            list: List of net numeric ID's.
        """

        return self._nets

    @nets.setter
    def nets(self, value: list = None) -> None:
        """
        MultiPhaseCoil nets list setter.
        Phase count determined by length of nets list.

        Args:
            value (list): List of net ID's (ints)
        """

        if value is None:
            raise ValueError("No nets specified.")
        if not isinstance(value, list):
            value = list(value)
        if len(value) < 2:
            raise ValueError("Net list length must be > 1: {value}")
        for el in value:
            if not isinstance(el, int):
                raise ValueError(f"All net IDs must be ints: {el}")

        self._nets = value

    @property
    def multiplicity(self) -> int:
        """
        Returns multiplicity of coils in a layer

        Returns:
            int: Multiplicity
        """

        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int):
        """
        Multiplicity property setter.

        Args:
            value (int): Number of times to repeat each phase of coil.
                         Phase count set by number of nets in nets property.
        """

        value = int(value)
        if value < 1:
            raise ValueError(f"Value must be >0: {value}")

        self._multiplicity = value

    @property
    def width(self) -> float:
        """
        Coil track width property getter.

        Returns:
            float: Sector track width.
        """

        return self._coil._width

    @width.setter
    def width(self, value: float = 0.2) -> None:
        """
        Coil track width setter.

        Args:
            value (float, optional): Track width. Defaults to 0.2.
        """

        self._coil._width = value

    @property
    def spacing(self) -> float:
        """
        Coil track spacing property getter.

        Returns:
            float: track spacing.
        """

        return self._coil._spacing

    @spacing.setter
    def spacing(self, value: float = 0.2):
        """
        Coil track spacing getter.

        Args:
            value (float, optional): Track spacing. Defaults to 0.2.
        """
        self._coil._spacing = value

    @property
    def dia_outside(self) -> float:
        """
        Coil track dia_outside property getter.

        Returns:
            float: track dia_outside.
        """

        return self._coil._dia_outside

    @dia_outside.setter
    def dia_outside(self, value: float = 30):
        """
        Coil track dia_outside getter.

        Args:
            value (float, optional): Track dia_outside. Defaults to 30.
        """

        self._coil._dia_outside = value

    @property
    def dia_inside(self) -> float:
        """
        Coil rack dia_inside property getter.

        Returns:
            float: track dia_inside.
        """

        return self._coil._dia_inside

    @dia_inside.setter
    def dia_inside(self, value: float = 10):
        """
        Coil rack dia_inside getter.

        Args:
            value (float, optional): Track dia_intside. Defaults to 10.
        """

        self._coil._dia_inside = value

    def Generate(self):
        """
        Generates multi-phase coil geometry.

        Key parameters:
        * layers - A full multi-phase coil is generated for each layer,
                   flipped from the layer above it.
        * nets - Defines the number of coils generated.
                 If two nets are provided, a two phase coil is generated.
                 if three nets are provided, a three phase coil is generated.
        * multiplicity - Replicates the phases in the coil.

        """

        # Clear out existing geometry.
        self._members = []

        # Coil angle
        angle_deg = 360 / len(self.nets) / self.multiplicity
        angle_rad = angle_deg * np.pi / 180

        # Update the base coil.  Then we'll just copy and move.
        self._coil.angle = angle_rad
        self._coil.Generate()

        # Generte list of coil phase names.
        ph_chr = np.arange(0, len(self.nets), dtype=int) + 65
        ph_name = [f"Ph{chr(x)}" for x in ph_chr] * self.multiplicity
        if self.multiplicity > 1:
            for i, _ in enumerate(ph_name):
                ph_name[i] = ph_name[i] + str(1 + int(i / len(self.nets)))

        # Create first layer set of coils.
        # We can then just copy that to other layers
        layer_g = Group()
        for i_net, n in enumerate(self.nets * self.multiplicity):
            # Create individual coil
            c = copy.deepcopy(self._coil)
            name = ph_name[i_net]
            c.name = name
            c.layer = self.layers[0]
            c.net = n
            c.Rotate(angle_rad * i_net)

            # Create coil text
            t = GrText(text=name)

            # Orients text facing out.
            t.Rotate(-np.pi / 2)

            # Push text to outside of coil
            r = self._coil.dia_outside / 2
            r += 2  # Space from coil to text
            t.Translate(r, 0)

            # Rotate text with coil
            txt_angle = angle_rad * i_net + angle_rad / 2
            t.Rotate(txt_angle)

            # Add label to coil
            c.AddMember(t)

            # Add coil to layer
            layer_g.AddMember(c)

        # Create other layers.
        if len(self.layers) > 1:

            for i_layer, layer in enumerate(self.layers):
                if i_layer == 0:
                    # First layer, special case.
                    # Add it in
                    layer_cur = copy.deepcopy(layer_g)
                    self.AddMember(layer_cur)
                    continue

                # Copy the pervious layer to the current layer.
                layer_cur = copy.deepcopy(layer_cur)

                # Flip the layer.
                # Since we copied the previoius layer, multi-layer coils will
                # just keep getting flipped back and forth.
                layer_cur.ChangeSideFlip()

                # Move all top-level elements to new layer.
                for el in layer_cur.members:
                    if isinstance(el, SectorCoil):
                        el.layer = layer

                        # Text was added to the SectorCoil, so find it.
                        txt = [
                            child for child in el.members if isinstance(child, GrText)
                        ][0]

                        # If last layer, then set to bottom layer silk screen
                        # If not, then remove from the coil.
                        if layer == self.layers[-1:][0]:
                            txt.layer = "B.SilkS"
                        else:
                            el.members.remove(txt)

                self.AddMember(layer_cur)

                # Create vias at center of coils
                for coil in layer_cur.members:
                    # Skip non-coil members.
                    if not isinstance(coil, SectorCoil):
                        continue

                    # Via position is at the end of the last,
                    # half arc in the coil
                    arcs = [a for a in coil.members if isinstance(a, Arc)]
                    pos = arcs[-1:][0].pointstart

                    # Via layers are this layer and previous.
                    layers = [self.layers[i_layer - 1], layer]

                    v = Via(position=pos, layers=layers, net=coil.net)

                    self.AddMember(v)

        else:
            # Only 1 layer in the group, so overwrite that groups members with this group.
            # This avoids having an additional, unneeded level of grouping.
            self._members = layer_g._members


#%%
if __name__ == "__main__":

    if True:
        # Three phase test
        c = MultiPhaseCoil()
        c.nets = [1, 2]
        c.multiplicity = 2
        c.dia_inside = 2
        c.layers = ["F.Cu", "B.Cu"]
        c.Generate()
        # c.Translate(x=120, y=90)
        print(c.ToKiCad())

    if False:  # Base geometry elements
        seg = Segment(Point(0, 0), Point(10, 10), width=0.25)
        seg.Translate(2, 3)
        print(seg.ToKiCad())

        arc = Arc(Point(0, 0), radius=5, start=0, end=np.pi / 2, width=0.123)
        arc.Rotate(np.pi / 2)
        print(arc.ToKiCad())

    # 4-quadrant coils.
    if False:
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

    # Coil flip test.
    if False:
        g = Group("Flip Test")

        c1 = SectorCoil()
        c1.name = "Top"
        c1.Generate()
        g.AddMember(c1)

        c2 = copy.deepcopy(c1)
        c2.name = "Bottom"
        c2.layer = "B.Cu"
        c2.ChangeSideFlip()
        c2.Generate()
        g.AddMember(c2)

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
