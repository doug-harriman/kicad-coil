# coil_3ph.py
# Three phase coil generation script.
# Inspired by: https://gist.github.com/JoanTheSpark/e5afd7081d9d1b9ad91a
#
# Tracks doc:
# URL: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/#_graphic_items_section:~:text=on%20the%20board.-,Tracks%20Section,-This%20section%20lists


from __future__ import annotations

import copy
import logging
import os
import uuid
import warnings
from typing import Tuple

import numpy as np

# KiCAD Python
# pip install kicad_python
# URL https://github.com/pointhi/kicad-python

# Python netlist generator:
# URL: https://skidl.readthedocs.io/en/latest/readme.html

# TODO: Group: find by property
# TODO: Estimate coil trace resistance.
#       * TraceLen implemented.
#       * Need to capture Copper thickness/weight: 35μm=1oz, 70μm=2oz, 105μm=3oz.
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
# TODO: Use teardrop via to minimize annular ring size.
#       Update SectorCoil.Generate()
# TODO: Use rounded corners/fillets.
#       * Radius is from the radius on the innermost coil.
#         May be larger do to via size.
#       * Could go back and replace all sharp corners,
#         or add as we go.

# Logging configuration
# Logging to STDOUT

if False:
    import sys  # Used by logging

    logging.basicConfig(stream=sys.stdout, filemode="w", level=logging.DEBUG)


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

        logging.debug(
            f"Via.Translate:x={self.position.x},y={self.position.y},dx={x},dy={y},id={str(self.id)}"
        )

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

    def __repr__(self: Segment) -> str:
        s = (
            f"Segment:"
            f"{self._start},"
            f"{self._end},"
            f'Layer:"{self._layer}", '
            f"{super().__repr__()}"
        )
        return s

    @property
    def start(self: Segment) -> Point:
        """
        Segment start point property getter.

        Returns:
            Point: Start point.
        """
        return self._start

    @start.setter
    def start(self: Segment, value: Point = None) -> None:
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
    def end(self: Segment) -> Point:
        """
        Segment end point property getter.

        Returns:
            Point: end point.
        """
        return self._end

    @end.setter
    def end(self: Segment, value: Point = None) -> None:
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
    def layer(self: Segment) -> str:
        """
        Returns layer for Segment.

        Returns:
            str: Layer
        """

        return self._layer

    @layer.setter
    def layer(self: Segment, value: str = "F.Cu"):
        """
        Sets layer for Segment.

        Args:
            value (str, optional): Layer name. Defaults to 'F.Cu'.
        """

        value = str(value)
        self._layer = value

    @property
    def width(self: Segment) -> float:
        """
        Segment track width property getter.

        Returns:
            float: Segment track width.
        """

        return self._width

    @width.setter
    def width(self: Segment, value: float = 0.2) -> None:
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

    def IntersectionLine(self: Segment, other: Segment = None) -> Point:
        """
        Calculates the intersection point of this Segment an another Segment.
        Treats that Segments as if they were infinite lines.

        Args:
            other (Segment, optional): Other line. Defaults to None.

        Returns:
            Point: Intersection Point or None if parallel segments.
        """

        if other is None:
            raise ValueError("Second segment not defined.")

        # Formula from:
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        # "Given two points on each line"
        x1 = self.start.x
        y1 = self.start.y
        x2 = self.end.x
        y2 = self.end.y
        x3 = other.start.x
        y3 = other.start.y
        x4 = other.end.x
        y4 = other.end.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if np.isclose(den, 0):
            warnings.warn("Parallel segments, no intersection point.")
            return None

        x_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        y_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

        return Point(x_num / den, y_num / den)

    def Translate(self: Segment, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the Segment Track by the given distances.
        """
        self._start.Translate(x, y)
        self._end.Translate(x, y)

    def Rotate(self: Segment, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the Segment about the given x,y coordinates by the given angle
        in radians.
        """

        self._start.Rotate(angle, x, y)
        self._end.Rotate(angle, x, y)

    def ChangeSideFlip(self: Segment):
        """
        Flips geometry for placing on opposite side.
        """

        # Just flip start & end
        temp = copy.deepcopy(self._end)
        self._end = copy.deepcopy(self._start)
        self._start = temp

    def ToKiCad(self: Segment, indent: str = "") -> str:
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

    def ToNumpy(self: Segment) -> Tuple:
        """
        Returns Numpy arrays for Segment X & Y coordinates.

        Suitable for plotting.
        """

        x1, y1 = self._start.ToNumpy()
        x2, y2 = self._end.ToNumpy()

        return np.append(x1, x2), np.append(y1, y2)

    def TraceLen(self: Segment) -> float:
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


class GrCircle:
    """
    Circle graphics object.
    Defined as center and a point on the circumference.
    """

    def __init__(
        self,
        position: Point = None,
        diameter: float = 5.0,
        width: float = 0.1,
        layer: str = "F.SilkS",
    ):

        self._id = uuid.uuid4()
        if position is None:
            self.position = Point()
        else:
            self.position = position
        self.layer = layer
        self.diameter = diameter
        self.width = width

    def __deepcopy__(self, memo):
        """
        Deep copy of GrCircle class with new UUIO.
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

        s = (
            f"GrCircle:{self.position}, "
            f"Dia:{self.diameter:0.3f}, "
            f'Layer:"{self.layer}", '
            f"ID:{str(self.id)}"
        )

        return s

    @property
    def id(self) -> uuid:
        """
        Returns the UUID of the GrCircle.
        """
        return self._id

    @property
    def position(self) -> Point:
        """
        GrCircle position property getter.

        Returns:
            Point: GrCircle center position.
        """

        return self._position

    @position.setter
    def position(self, value: Point = Point()) -> None:
        """
        GrCircle center position getter.

        Args:
            value (Point): Center position.

        Raises:
            ValueError: Invalid position value type.
        """

        if not isinstance(value, Point):
            return TypeError(f"Invalid position value type: {type(value)}")

        self._position = value

    @property
    def diameter(self) -> float:
        """
        GrCircle diameter property getter.

        Returns:
            float: circle diameter
        """

        return self._diameter

    @diameter.setter
    def diameter(self, value: float = 5):
        """
        GrCircle angle getter.

        Args:
            value (float, optional): GrCircle diameter. Defaults to 5.
        """

        angle = float(value)
        self._diameter = angle

    @property
    def radius(self) -> float:
        """
        Returns GrCircle radius.

        Returns:
            float: GrCircle radius.
        """

        return self.diameter / 2

    @radius.setter
    def radius(self, value: float) -> None:
        """
        Sets GrCircle radius.

        Args:
            value (float): radius.
        """

        self.diameter = value * 2

    @property
    def layer(self) -> str:
        """
        Returns layer for GrCircle.

        Returns:
            str: Layer
        """

        return self._layer

    @layer.setter
    def layer(self, value: str = "F.SilkS"):
        """
        Sets layer for GrCircle.

        Args:
            value (str, optional): Layer name. Defaults to 'F.SilkS'.
        """

        value = str(value)
        self._layer = value

    @property
    def width(self) -> float:
        """
        GrCircle line width.

        Returns:
            float: GrCircle width.
        """

        return self._width

    @width.setter
    def width(self, value: float = 0.1) -> None:
        """
        GrCircle line width setter.

        Args:
            value (float, optional): GrCircle line width.  Defaults to 0.1.
        """

        size_orig = value
        size = float(value)
        if size <= 0.0:
            raise ValueError(f"GrCircle line width must be positive: {size_orig}")

        self._width = size

    def Translate(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Translates the GrText by the given distances.
        """
        self.position.Translate(x, y)

    def Rotate(self, angle: float, x: float = 0.0, y: float = 0.0) -> None:
        """
        Rotates the GrCircle about the given x,y coordinates by the given angle in radians.
        """

        # This is a no-op for a circle.
        # Provided so that calls do not error out.
        pass

    def ChangeSideFlip(self):
        """
        Flips circle for placing on opposite side.
        """

        # This is a no-op for a circle.
        # Provided so that calls do not error out.
        pass

    def ToKiCad(self, indent: str = "") -> str:
        """
        Converts GrText to KiCAD string.
        """

        s = (
            f"{indent}"
            f"(gr_circle  "
            f"(center {self.position.ToKiCad()}) "
            f"(end {self.position.x+self.diameter/2:0.6f} {self.position.y:0.6f}) "
            f'(layer "{self._layer}") '
            f"(tstamp {str(self.id)}) )"
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

    def FindByProperty(self, property, value) -> list:
        """_summary_

        Args:
            property (_type_): _description_
            value (_type_): _description_

        Returns:
            list: _description_
        """
        pass

    def FindByClass(self, cls=None) -> list:
        """
        Recursively searches members list for members of type class.

        Args:
            cls (class type): Class type to search for.

        Returns:
            list: List of all child members of type class.
        """

        matches = []
        if cls is None:
            return matches

        for member in self.members:
            if isinstance(member, cls):
                matches.append(member)
            elif isinstance(member, Group):
                matches.extend(member.FindByClass(cls))

        return matches

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
        self._dia_via = 0.8
        self._center = Point()
        self._angle = np.pi / 2

        self._pointstart = None
        self._pointend = None

        self._layer = "F.Cu"
        self._width = 0.2
        self._spacing = 0.2
        self._net = 1

        # Calculated properties
        self._turns = 0

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

    @property
    def dia_via(self) -> float:
        """
        Returns via diameter for center of coil.
        Used to determine innermost coil size.

        Returns:
            float: Via diameter.
        """

        return self._dia_via

    @dia_via.setter
    def dia_via(self, value: float = 0.8) -> None:
        """
        Sets via diameter to use to determine size of innermost coil.

        Args:
            value (float, optional): Via diameter. Defaults to 0.8.
        """

        value_orig = value
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"Via diameter must be >0: {value_orig}")

        self._dia_via = value

    @property
    def pointstart(self) -> Point:
        """
        Returns Sector coil Track start point.

        Returns:
            Point: Track start point
        """
        return self._pointstart

    @property
    def pointend(self) -> Point:
        """
        Returns Sector coil Track end point.

        Returns:
            Point: Track end point
        """
        return self._pointend

    @property
    def turns(self) -> int:
        """
        Number of turns in coil.

        Returns:
            int: Number of turns.
        """

        return self._turns

    def Generate(self):
        """
        Generates coil geometry segments.

        All segments define the midpoints of the track.

        """

        logging.debug("SC_ARC:\tType,\tRadius,\tAngle,\t  dX,\t  dY,\t")

        # Error checks
        if self.dia_inside >= self.dia_outside:
            raise ValueError(
                f"Inside dia >= outside dia: {self.dia_inside} >= {self.dia_outside}"
            )

        def arc_interect(radius: float, coordinate: float) -> float:
            """
            Determines the intersection point between a line
            and an arc.  Assuming:
            * Full first quadrant arc (0-90 deg)
            * Horizontal or vertical lines only.
            * Symmetric problem, so given one coordinate,
              calculates the other.
            * Starting with horizontal line, coming in from outside connection point.

            Args:
                radius (float): Radius of arc.
                coordinate (float): X coord for vertical line, y for horizontal.

            Returns:
                float: other coordinate for intersection point.
            """

            # Assume this is a horizontal line and we've been
            # given the Y-coordinate of the point.  The angle
            # of the intersection point is:
            theta = np.arcsin(coordinate / radius)

            # Then the other point is:
            return radius * np.cos(theta)

        # Since we use the AddMember method, need to clear out.
        self._members = []
        self._turns = 0
        inner_arc = True  # Denotes if we have room for inner arc

        # OD is started outside of coil.
        # First loop will have outside edge on the OD
        rad_out = self._dia_outside / 2 + 1.5 * self._width
        rad_pitch = self._width + self._spacing

        # Potenital radii
        # TODO: Rework with middle dia supporting min via dia.
        radii = np.arange(rad_out, 0, -rad_pitch)

        # Must be greater than inside radius
        radii = radii[np.where(radii > self._dia_inside / 2)]

        # Must be odd (for coil center via)
        if len(radii) % 2 == 0:
            radii = radii[:-1]

        # Reorder radii so we process outside->inside->outide, etc
        r = np.zeros((len(radii) * 2,))
        r[::2] = radii
        r[1::2] = np.flip(radii)
        r = r[: int(len(r) / 2)]
        radii = r

        # Offset each line by a delta position based on spacing & track size
        dp = self.width + self.spacing

        # Minimal radial distance needed to fit via
        # 1.25 is a hueristic for estimating the effective thickness increase
        # in the inner corner of two lines intersecting.
        # An exact value could be calculated based on line thicknesses however.
        radial_dist_min = self.dia_via + 2 * self.spacing + self.width * 1.25

        # Create a first horizontal line.
        offset = 0.0
        create_vertical = True
        seg_start = Point(arc_interect(radius=radii[0], coordinate=offset), offset)
        seg_end = Point(arc_interect(radius=radii[1], coordinate=offset), offset)
        seg = Segment(
            start=seg_start,
            end=seg_end,
            width=self.width,
            layer=self.layer,
            net=self.net,
        )
        self.AddMember(seg)
        self._pointstart = seg.start
        offset += dp

        # Basic sector coil creation is for a 90 deg sector.
        # For other angles, create the points, then rotate.
        rotation_needed = not np.isclose(self.angle, np.pi / 2)
        rotation_theta = self.angle - np.pi / 2

        # For Each pair of radii (0,1),(2,3) ...
        # intersect one line parallel to the horizontal axis.
        # Step through those, calculating intersection points.
        intersect_coord = np.zeros(len(radii))
        for i in range(2, len(radii)):

            # Calc start point for next segment.
            intersect_start = arc_interect(radius=radii[i - 1], coordinate=offset)
            intersect_end = arc_interect(radius=radii[i], coordinate=offset)

            if create_vertical:
                seg_start = Point(offset, intersect_start)

                # Rotate segment point if needed.
                if rotation_needed:
                    seg_start.Rotate(rotation_theta)

                # Every vertical maps to a turn.
                self._turns += 1
            else:
                seg_start = Point(intersect_start, offset)

            # Arc links previous end point to new starting point.
            # Angle to horizontal line end point.
            angle_start = np.arctan2(seg_end.y, seg_end.x)

            # Angle to vertical line start point
            angle_end = np.arctan2(seg_start.y, seg_start.x)

            dA = angle_start - angle_end

            # If angle is positive for an inside arc, then the lines cross
            # and we don't want the arc.
            # An inside arc is one created when create_vertical = true
            arc = None
            if create_vertical:
                log_txt = "in"

                if dA > 0:
                    # Denote that things are tight for inner arc.
                    inner_arc = False

                if inner_arc:
                    arc = Arc(
                        center=Point(),  # During generation, center always at (0,0)
                        radius=radii[i - 1],
                        start=angle_start,
                        end=angle_end,
                        width=self.width,
                        layer=self.layer,
                        net=self.net,
                    )
                    self.AddMember(arc)

            else:
                # Always create outside arcs.
                log_txt = "out"
                arc = Arc(
                    center=Point(),  # During generation, center always at (0,0)
                    radius=radii[i - 1],
                    start=angle_start,
                    end=angle_end,
                    width=self.width,
                    layer=self.layer,
                    net=self.net,
                )
                self.AddMember(arc)

                # If we're not adding inner arcs, then check for
                # termination due to inner coil size getting small.
                if not inner_arc:
                    radius_last_intersect = np.sqrt(seg.start.x**2 + seg.start.y**2)
                    radial_dist_left = arc.radius - radius_last_intersect
                    if radial_dist_left < 2 * radial_dist_min:
                        # Terminate coil generation
                        break

            # Debug logging
            if arc is not None:
                logging.debug(
                    f"SC_ARC:\t"
                    f"{log_txt},\t"
                    f"{arc.radius:02.3f},\t"
                    f"{arc.start-arc.end: .2f},\t"
                    f"{arc.pointstart.x-arc.pointend.x: .3f},\t"
                    f"{arc.pointstart.y-arc.pointend.y: .3f},"
                )

            # Calc end point for next segment.
            if create_vertical:
                seg_end = Point(offset, intersect_end)

                # Rotate segment point if needed.
                if rotation_needed:
                    seg_end.Rotate(rotation_theta)

            else:
                seg_end = Point(intersect_end, offset)
                offset += dp  # Update offset every other time

            # Segment
            seg_prev = seg
            seg = Segment(
                start=seg_start,
                end=seg_end,
                width=self.width,
                layer=self.layer,
                net=self.net,
            )
            self.AddMember(seg)

            # If we're not creating an arc, then the new segment
            # and the old segment intersect to form the coil.
            # Find the intersection point.
            if create_vertical and not inner_arc:
                intersect = seg.IntersectionLine(seg_prev)

                seg_prev.end = copy.deepcopy(intersect)
                seg.start = copy.deepcopy(intersect)

            # Toggle
            create_vertical = not create_vertical

        # Add in last half arc if we were adding inner arcs.
        if inner_arc:
            arc = Arc(
                center=Point(),  # During generation, center always at (0,0)
                radius=radii[-1],
                start=np.arctan2(seg_end.y, seg_end.x),  # Last end point.
                end=self._angle / 2,
                width=self.width,
                layer=self.layer,
                net=self.net,
            )
            self.AddMember(arc)
            self._pointend = arc.pointend
        else:
            # Approximation of middle if remaining triangle.
            # Centroid of triangle is 1/3 above base.
            r = radius_last_intersect + 2 * radial_dist_left / 3

            # Angle is offset slightly from sector angle.
            # Use angle to last intesect point.
            th = np.arctan2(intersect.y, intersect.x)

            # Point for the via.
            pt_via = Point(r * np.cos(th), r * np.sin(th))

            seg = Segment(
                start=copy.deepcopy(arc.pointend),
                end=pt_via,
                width=self.width,
                layer=self.layer,
                net=self.net,
            )
            self.AddMember(seg)
            self._pointend = seg.end

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
        self._via = Via()

        # Holes
        self._center_hole = None
        self._mount_hole = None
        self._mount_hole_pattern_diameter = 15
        self._mount_hole_pattern_angles = np.array([0, 120, 240]) * np.pi / 180

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

    @property
    def via(self) -> Via:
        """
        Returns the default via for the coil.
        Added vias will be made as copies of this via.

        Returns:
            Via: Default via.
        """

        return self._via

    @via.setter
    def via(self, value: Via):
        """
        Via property setter.
        All vias added will be copied from this via.
        Must be set before Geometry generation.

        Args:
            value (Via): _description_
        """

        if not isinstance(value, Via):
            raise TypeError(f"Expecting object of type Via. Got: {type(value)}")

        # TODO: Search all generated geometry and replace with new via type.
        #      Update documentation block.

        self._via = value

    @property
    def center_hole(self) -> GrCircle:
        """
        GrCircle to define center hole to cut.
        If None, then no center hole.

        Returns:
            GrCircle: Center hole circle.
        """

        return self._center_hole

    @center_hole.setter
    def center_hole(self, value: GrCircle) -> None:
        """
        GrCircle center for center hole.
        If None, then no center hole.

        Args:
            value (GrCircle): Center hole definition.
        """

        if value is None:
            self._center_hole = value
            return

        if not isinstance(value, GrCircle):
            raise TypeError(f"Type GrCircle expected, got: {type(value)}")

        self._center_hole = copy.deepcopy(value)
        self._center_hole.position = Point()  # Always at center
        self._center_hole.layer = "Edge.Cuts"
        self._center_hole.width = 0.1

    @property
    def mount_hole(self) -> GrCircle:
        """
        GrCircle to define mount hole to cut.
        If None, then no mount hole.

        Returns:
            GrCircle: mount hole circle.
        """

        return self._mount_hole

    @mount_hole.setter
    def mount_hole(self, value: GrCircle) -> None:
        """
        GrCircle mount for mount hole.
        If None, then no mount hole.

        Args:
            value (GrCircle): mount hole definition.
        """

        if value is None:
            self._mount_hole = value
            return

        if not isinstance(value, GrCircle):
            raise TypeError(f"Type GrCircle expected, got: {type(value)}")

        self._mount_hole = copy.deepcopy(value)
        self._mount_hole.layer = "Edge.Cuts"
        self._mount_hole.width = 0.1

    @property
    def mount_hole_pattern_diameter(self) -> float:
        """
        Mounting hold radial placement pattern diameter.

        Returns:
            Float: Diameter to center of holes.
        """

        return self._mount_hole_pattern_diameter

    @mount_hole_pattern_diameter.setter
    def mount_hole_pattern_diameter(self, value: float) -> None:
        """
        Mounting hold radial placement pattern diameter.
        Pattern diameter - hole radius > coil outside diameter.

        Args:
            value (float): Mounting hole radial pattern diameter.
        """

        value = float(value)
        if value <= 0:
            raise ValueError(f"Mounting hole pattern diameteter must be >0: {value}")

        self._mount_hole_pattern_diameter = value

    @property
    def mount_hole_pattern_angles(self) -> list:
        """
        List of angles at which to place mounting holes.
        Angle in radians.

        Returns:
            list: List of angles for mounting holes.
        """

        return self._mount_hole_pattern_angles

    @mount_hole_pattern_angles.setter
    def mount_hole_pattern_angles(self, value: list) -> None:
        """
        List of angles at which to place mounting holes.
        Angle in radians.

        Args:
            value (list): List of angles.
        """

        if isinstance(value, float):
            value = list(value)

        # Check all values.
        for v in value:
            if isinstance(v, int):
                v = float(v)
            if not isinstance(v, float):
                raise TypeError(f"Angle list contains non-float value: {v}")

        self._mount_hole_pattern_angles = value

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

        # Error checks
        if self.center_hole is not None:
            if self.center_hole.diameter >= self.dia_inside:
                raise ValueError(
                    f"Center hole dia >= coil inner dia: {self.center_hole.diameter } >= {self.dia_inside}"
                )

        if self.mount_hole is not None:
            if (
                self.mount_hole_pattern_diameter - self.mount_hole.diameter / 2
                <= self.dia_outside
            ):
                raise ValueError(
                    f"Mounting holes touch coil.  Increase mounting hole pattern diameter."
                )

        # Clear out existing geometry.
        self._members = []

        # Coil angle
        angle_deg = 360 / len(self.nets) / self.multiplicity
        angle_rad = angle_deg * np.pi / 180

        # Update the base coil.  Then we'll just copy and move.
        self._coil.angle = angle_rad
        self._coil.dia_via = self.via.size
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
                        # TODO: Replace with FindByClass method
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
                # TODO: Replace with FindByClass method
                for coil in layer_cur.members:
                    # Skip non-coil members.
                    if not isinstance(coil, SectorCoil):
                        continue

                    # Create a copy of the base via
                    # Via position is at the end of the last Track element
                    v = copy.deepcopy(self._via)
                    v.position = copy.deepcopy(coil.pointend)
                    v.layers = [self.layers[i_layer - 1], layer]
                    v.net = coil.net

                    self.AddMember(v)

        else:
            # Only 1 layer in the group, so overwrite that groups members with this group.
            # This avoids having an additional, unneeded level of grouping.
            self._members = layer_g._members

        # Center hole
        if self.center_hole is not None:
            self.AddMember(self.center_hole)

        # Mounting holes
        if self.mount_hole is not None:
            g = Group(name="Mounting Holes")
            r = self.mount_hole_pattern_diameter / 2
            for angle in self.mount_hole_pattern_angles:
                hole = copy.deepcopy(self.mount_hole)
                hole.position.x = r * np.cos(angle)
                hole.position.y = r * np.sin(angle)
                g.AddMember(hole)
            self.AddMember(g)


#%%
if __name__ == "__main__":

    from pint import Quantity as Q

    # Parameters
    # Base via
    drill = 0.3
    via_size = 0.61
    via = Via(size=via_size, drill=drill)

    # Tracks
    width = Q(8e-3, "in").to("mm").magnitude
    spacing = Q(6e-3, "in").to("mm").magnitude  # PCBWay min spacing for cheap boards

    # 3-phase test, multiplicity 1
    if False:
        c = MultiPhaseCoil()
        c.nets = [1, 2, 3]
        c.multiplicity = 1
        c.dia_inside = 5
        c.dia_outside = 20
        c.layers = ["F.Cu", "B.Cu"]
        c.width = Q(8e-3, "in").to("mm").magnitude
        c.spacing = spacing
        c.via = via

        # Center hole
        hole = GrCircle(Point(), diameter=3)
        # c.center_hole = hole
        c.mount_hole = hole
        c.mount_hole_pattern_diameter = c.dia_outside + hole.radius + 4
        c.mount_hole_pattern_angles = (np.array([0, 120, 240]) + 30) * np.pi / 180

        c.Generate()
        c.Translate(x=120, y=90)
        print(c.ToKiCad())

    # 3-phase test, multiplicity 2
    if True:
        c = MultiPhaseCoil()
        c.nets = [1, 2, 3]
        c.multiplicity = 2
        c.dia_inside = 6
        c.dia_outside = 30
        c.layers = ["F.Cu", "B.Cu"]
        c.width = width
        c.spacing = spacing
        c.via = via

        # Center hole
        hole = GrCircle(Point(), diameter=3)
        # c.center_hole = hole
        c.mount_hole = hole
        c.mount_hole_pattern_diameter = c.dia_outside + hole.radius + 4
        c.mount_hole_pattern_angles = (np.array([0, 120, 240]) + 30) * np.pi / 180

        c.Generate()
        c.Translate(x=120, y=90)
        with open("tmp.txt", "w") as fp:
            fp.write(c.ToKiCad())
        # print(c.ToKiCad())

        matches = c.FindByClass(Via)
        print(matches)
        print(f"Count: {len(matches)}")

    # 2-phase test, multiplicity 2
    if False:
        c = MultiPhaseCoil()
        c.nets = [1, 2]
        c.multiplicity = 2
        c.dia_inside = 5
        c.dia_outside = 20
        c.layers = ["F.Cu", "B.Cu"]
        c.width = width
        c.spacing = spacing
        c.via = via

        c.Generate()
        c.Translate(x=120, y=90)
        print(c.ToKiCad())

    # Sector coil test
    if False:
        c = SectorCoil()
        c.width = width
        c.spacing = spacing
        c.angle = 60 * np.pi / 180
        c.dia_inside = 6
        c.dia_outside = 30

        c.Generate()
        print(c.ToKiCad())

    # Base geometry elements
    if False:
        seg = Segment(Point(0, 0), Point(10, 10), width=0.25)
        seg.Translate(2, 3)
        print(seg.ToKiCad())

        arc = Arc(Point(0, 0), radius=5, start=0, end=np.pi / 2, width=0.123)
        arc.Rotate(np.pi / 2)
        print(arc.ToKiCad())


# %%
