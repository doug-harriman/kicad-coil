import femm
from pint import Quantity as Q
import numpy as np


class Femm:
    def __init__(self):

        self._circuits = []
        self._next_group_id = 0

        # Open Femm and a model file.
        femm.openfemm(1)  # 0=Show GUI, 1=Hide GUI
        femm.newdocument(0)  # 0=Magnetics problem

        # Default object creation location.
        # Because FEMM requires selection of objects by position, you want to
        # create things in a position away from where you'll be working.
        self._default_create_x = -100
        self._default_create_y = -100

        # Problem definition
        frequency = 0  # static
        self._units = 'millimeters'
        prob_type = 'planar'
        precision = 1e-8  # standard value
        depth = 1  # in "units".  Using unit value lets us easily scale later.
        femm.mi_probdef(frequency, self._units, prob_type, precision, depth)

        femm.main_resize(500, 1000)  # Set FEMM window size in screen pixels

        # Define materials in use in the problem.
        matl_mag = "N52"
        # Solid copper, must define individual trace wires.
        matl_coil = "Copper"
        matl_air = "Air"
        matl_list = [matl_mag, matl_coil, matl_air]
        for matl in matl_list:
            femm.mi_getmaterial(matl)

    @property
    def units(self) -> str:

        return self._units

    def new_group_id(self) -> int:
        """
        Returns next available group ID.
        """

        self._next_group_id += 1
        return self._next_group_id

    def _circuit_add(self, circuit=None):
        """Adds name of circuit to list for tracking.
            Should not be called directly."""

        self._circuits.append(circuit.name)

        # Register with femm
        femm.mi_addcircprop(circuit.name,
                            circuit.current.to('A').magnitude,
                            1)  # series -> same currents for all elements.


class Circuit:

    def __init__(self,
                 model: Femm = None,
                 name: str = 'circuit1',
                 current: Q = Q(1, 'A')):

        if model is None:
            raise ValueError("Model not specified.")

        self._name = name
        self._current = current

        # Register with the model
        model._circuit_add(self)

    @property
    def name(self):
        return self._name

    @property
    def current(self):
        return self._current


class Track:
    def __init__(self,
                 model: Femm = None,
                 circuit: Circuit = None,
                 width: Q = Q(1, 'mm'),
                 height: Q = Q(1, 'mm'),
                 x: Q = None,
                 y: Q = None):

        if model is None:
            raise ValueError("Model not specified.")

        if x is None:
            x = Q(model._default_create_x, model.units)
        if y is None:
            y = Q(model._default_create_y, model.units)

        self._x1 = x.to(model.units).magnitude
        self._x2 = self._x1 + width.to(model.units).magnitude
        self._y1 = y.to(model.units).magnitude
        self._y2 = self._y1 + height.to(model.units).magnitude
        self._group = None

        femm.mi_clearselected()
        femm.mi_drawrectangle(self._x1, self._y1, self._x2, self._y2)
        self.select_by_rect()
        self.group = model.new_group_id()

        self._model = model

    def select_by_rect(self):
        """
        Selects object by its defining rectangle so that it is active for operations.
        """
        mode = 4  # All entity types selected
        femm.mi_selectrectangle(
            self._x1, self._y1, self._x2, self._y2, mode)

    def select_by_id(self):
        """
        Selects object by its group ID.
        Object is given an ID upon creation for selection.
        If group ID is changed, this method will select everything in the group.
        """

    @property
    def group(self):
        """
        Group number associated with object.
        """
        return self._group

    @group.setter
    def group(self, value: int = None):
        """
        Set group number associated with object
        """
        if value is None:
            raise ValueError("No group number provided")

        self.select_by_rect()
        femm.mi_setgroup(value)
        self._group = value

    @property
    def center(self) -> tuple:
        """
        Tuple of center coordinates.
        """

        x_c = np.mean([self._x1, self._x2])
        y_c = np.mean([self._y1, self._y2])

        return (Q(x_c, self._model.units), Q(y_c, self._model.units))

    def position_ll(self, x: Q = None, y: Q = None):
        """
        Sets object lower left corner to specified positions.
        """

        # Unit conversions.
        dx = 0
        dy = 0
        if x is not None:
            x = x.to(self._model.units).magnitude
            dx = x - self._x1

        if y is not None:
            y = y.to(self._model.units).magnitude
            dy = y - self._y1

        dx = Q(dx, self._model.units)
        dy = Q(dy, self._model.units)

        self.translate(dx, dy)

    def translate(self, dx: Q = Q(0, 'mm'), dy: Q = Q(0, 'mm')):
        """
        Translates object specified distances in X & Y.
        """

        dx = dx.to(self._model.units).to('mm').magnitude
        dy = dy.to(self._model.units).to('mm').magnitude

        self.select_by_rect()
        femm.mi_movetranslate(dx, dy)

        # Update internal data
        self._x1 += dx
        self._x2 += dx
        self._y1 += dy
        self._y2 += dy


class Coil:

    def __init__(self,
                 model: Femm = None,
                 track_width: Q = Q(1, 'mm'),
                 track_spacing: Q = Q(1, 'mm'),
                 track_thickness: Q = Q(1, 'mm'),
                 dia_inside: Q = Q(1, 'mm'),
                 x_center: Q = Q(0, 'mm'),
                 y_center: Q = Q(0, 'mm'),
                 turns: int = 3,
                 current: Q = Q(1, 'A')):

        if model is None:
            raise ValueError("Model not specified.")
        self._model = model

        self._group = model.new_group_id()
        print(f'Coil ID: {self._group}')

        # Name the coil, but it's not user settable for now.
        self._name = f'Coil_{self._group}'

        # Create a circuit for the coil.
        self._circuit = Circuit(model=model,
                                name=f'Circuit_{self._name}',
                                current=current)

        # Create an array of tracks.
        self._conductors = []

        # Left side, from inside out.
        offset = x_center - track_width - dia_inside/2
        for i in np.arange(0, turns):
            # Create and position track.
            # Created at default position, well outside where we want it
            # to avoid issues selecting geometry by position.
            track = Track(model=model,
                          width=track_width,
                          height=track_thickness)
            track.position_ll(x=offset, y=track_thickness/2)

            # Create FEMM label to mark material, circuit & turn count.
            (xc, yc) = track.center
            xc = xc.magnitude
            yc = yc.magnitude
            femm.mi_addblocklabel(xc, yc)
            femm.mi_selectlabel(xc, yc)
            femm.mi_setblockprop("Copper",  # Material to use.  Solid copper for a single trace.
                                 1,         # Auto mesh size
                                 .01,       # Mesh size.  Ignored for auto.
                                 # Magnetization direction, ignored.
                                 self._circuit.name,
                                 0,
                                 self._group,  # Belongs to coil group.
                                 1)          # Number of turns

            # Bookkeeping
            offset -= (track_width + track_spacing)
            self._conductors.append(track)

        # Right side, from inside out
        offset = x_center + dia_inside/2
        for i in np.arange(0, turns):
            # Create and position track.
            track = Track(model=model,
                          width=track_width,
                          height=track_thickness)
            track.position_ll(x=offset, y=track_thickness/2)

            # Create FEMM label to mark material, circuit & turn count.
            (xc, yc) = track.center
            xc = xc.magnitude
            yc = yc.magnitude
            femm.mi_addblocklabel(xc, yc)
            femm.mi_selectlabel(xc, yc)
            femm.mi_setblockprop("Copper",  # Material to use.  Solid copper for a single trace.
                                 1,         # Auto mesh size
                                 .01,       # Mesh size.  Ignored for auto.
                                 # Magnetization direction, ignored.
                                 self._circuit.name,
                                 0,
                                 self._group,  # Belongs to coil group.
                                 -1)          # Number of turns, opposite direction

            # Bookkeeping
            offset += track_width + track_spacing
            self._conductors.append(track)

    @property
    def group(self) -> int:
        """
        Coil group ID.
        """

        return self._group

    @property
    def name(self) -> str:
        """
        Coil name (read only)
        """
        return self._name


if __name__ == "__main__":

    model = Femm()

    # Test Coil
    track_thickness = Q(34.8, 'um')
    track_spacing = Q(6, 'milliinch')
    track_width = track_spacing
    dia_inside = 4*track_spacing

    coil = Coil(model=model,
                turns=12,
                track_thickness=track_thickness,
                track_width=track_width,
                track_spacing=track_spacing,
                dia_inside=dia_inside)

    femm.mi_saveas('test.FEM')

    # import time
    # time.sleep(10)
