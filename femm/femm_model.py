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

        # Boundary
        self._boundary_group = None

    @property
    def units(self) -> str:

        return self._units

    def new_group_id(self) -> int:
        """
        Returns next available group ID.
        """

        self._next_group_id += 1
        return self._next_group_id

    def boundary_generate(self,
                          xc: Q = Q(0, 'mm'),
                          yc: Q = Q(0, 'mm'),
                          radius: Q = Q(50, 'mm')) -> None:
        """
        Generates boundary conditions.
        NOTE: Can only be run once per model.

        Args:
            xc (Q, optional): X coord of center of circular boundary. Defaults to Q(0,'mm').
            yc (Q, optional): Y coord of center of circular boundary. Defaults to Q(0,'mm').
            radius (Q, optional): Radius of circular boundary. Defaults to Q(50,'mm').

        Raises:
            ValueError: Boundary condition already exists.
        """

        # TODO: Support deletion and re-creation of boundary group.
        if self._boundary_group is not None:
            raise ValueError("Boundary condition specification exists.")

        # Handle units if provided
        if isinstance(xc, Q):
            xc = xc.to(self._units).magnitude
        if isinstance(yc, Q):
            yc = yc.to(self._units).magnitude
        if isinstance(radius, Q):
            radius = radius.to(self._units)

        self._boundary_group = model.new_group_id()

        femm.mi_makeABC(7,       # Number of shells for boundary.  7 is standard.
                        radius,  # Radius of boundary shell circle
                        xc,      # Center of boundary shell circle
                        yc,
                        0)       # Boundary condition: 0=Dirichlet, 1=Neumann

        # Fill in boundary zone with Air
        y_l = yc+radius*0.95     # Place label just inside boundary.
        femm.mi_addblocklabel(xc, y_l)
        femm.mi_selectlabel(xc, y_l)
        femm.mi_setblockprop("Air",
                             1,     # Auto mesh
                             0.01,  # Mesh size, not used.
                             "",    # Circuit name.  "" => No circuit
                             0,     # Magnetization dir, ignored.
                             self._boundary_group)

    @property
    def boundary_group(self) -> int:
        """
        Boundary group ID.

        Returns:
            int: Group ID of boundary conditions.
        """

        return self._boundary_group

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


class Rect:
    """
    Base class for geometric manipulation.
    """

    def __init__(self,
                 model: Femm = None,
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

        femm.mi_clearselected()
        femm.mi_drawrectangle(self._x1, self._y1, self._x2, self._y2)

        self._group = None
        self.group = model.new_group_id()  # Use the method to take care of the details

        self._model = model

    def select_by_rect(self):
        """
        Selects object by its defining rectangle so that it is active for operations.
        """
        mode = 4  # All entity types selected
        femm.mi_selectrectangle(
            self._x1, self._y1, self._x2, self._y2, mode)

    def select_by_group(self):
        """
        Selects object by its group ID.
        Object is given an ID upon creation for selection.
        If group ID is changed, this method will select everything in the group.
        """
        femm.mi_selectgroup(self._group)

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

    @property
    def bbox(self) -> tuple:
        """
        Returns tuple of (x1,y1,x2,y2) where x1,y1 represents the 
        bounding box lower left corner and x2,y2 represents the 
        bounding box upper right corner.
        """

        return Q([self._x1, self._y1, self._x2, self._y2], self._model.units)

    def position_ll_set(self, x: Q = None, y: Q = None):
        """
        Sets object lower left corner to specified positions.
        If x or y position is not specified, that coordinate is not changed.
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

    # TODO: Add position_center_set()

    def translate(self, dx: Q = Q(0, 'mm'), dy: Q = Q(0, 'mm')):
        """
        Translates object specified distances in X & Y.
        If dx or dy is not specified, that coordinate is not changed.
        """

        dx = dx.to(self._model.units).to('mm').magnitude
        dy = dy.to(self._model.units).to('mm').magnitude

        self.select_by_group()
        femm.mi_movetranslate(dx, dy)

        # Update internal data
        self._x1 += dx
        self._x2 += dx
        self._y1 += dy
        self._y2 += dy


# TODO: Handled magnet material and magnetization direction
class Magnet(Rect):
    def __init__(self,
                 model: Femm = None,
                 width: Q = Q(1, 'mm'),
                 height: Q = Q(1, 'mm'),
                 x: Q = None,
                 y: Q = None):

        # Init Rect
        super().__init__(model, width, height, x, y)


# TODO: Handle circuit within the Track
class Track(Rect):
    def __init__(self,
                 model: Femm = None,
                 circuit: Circuit = None,
                 width: Q = Q(1, 'mm'),
                 height: Q = Q(1, 'mm'),
                 x: Q = None,
                 y: Q = None):

        # Init the Rect
        super().__init__(model, width, height, x, y)


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
        # TODO: How to deal with next coil, with opposite current.
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
            track.position_ll_set(x=offset, y=track_thickness/2)
            # TODO: Put track into the coil's group.

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
            track.position_ll_set(x=offset, y=track_thickness/2)

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

    @property
    def bbox(self) -> tuple:
        """
        Returns tuple of (x1,y1,x2,y2) where x1,y1 represents the 
        bounding box lower left corner and x2,y2 represents the 
        bounding box upper right corner.
        """

        x1 = Q(np.Inf, self._model.units)
        y1 = Q(np.Inf, self._model.units)
        x2 = -x1
        y2 = -y1
        for track in self._conductors:
            bb = track.bbox
            x1 = min([x1, bb[0]])
            y1 = min([y1, bb[1]])
            x2 = max([x2, bb[2]])
            y2 = max([y2, bb[3]])

        return [x1, y1, x2, y2]


if __name__ == "__main__":

    model = Femm()

    # Test Coil
    # PCB design limit info:
    # https://github.com/doug-harriman/kicad-coil/blob/main/design-info.md
    track_thickness = Q(34.8, 'um')
    track_spacing = Q(6, 'milliinch')
    track_width = track_spacing
    dia_inside = Q(1.6, 'mm')  # Measured from KiCAD PCB

    coil = Coil(model=model,
                turns=12,
                track_thickness=track_thickness,
                track_width=track_width,
                track_spacing=track_spacing,
                dia_inside=dia_inside)

    # -----------------------------------------------------------------
    # Generate the boundary conditions.
    # -----------------------------------------------------------------
    # TODO: if all children of the model can have a bbox prop,
    #       then we can create a boundary class that can find the
    #       boundary size automatically.
    bc_group = model.new_group_id()
    bb = [x.magnitude for x in coil.bbox]
    xc = np.mean([bb[0], bb[2]])
    yc = np.mean([bb[1], bb[3]])
    r = 2*np.linalg.norm([xc - bb[0], yc - bb[1]])
    model.boundary_generate(xc=xc, yc=yc, radius=r)

    # TODO: Add in magnet.

    femm.mi_saveas('test.FEM')

    # TODO: Run sim
    # TODO: Gather results

    # import time
    # time.sleep(10)
