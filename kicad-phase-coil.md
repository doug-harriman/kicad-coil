# KiCAD Multi Phase Coil Generator

# Common Traits
## Common Methods
All classes provide the following methods:
* `ToKiCad` - Generates a KiCAD PCB file string to instantiate the element.
* `Translate` - 2D translation of the .
* `Rotate` - Rotates the element about the given point, defaults to (0,0), by the given angle.
* `ChangeSideFlip` - Flips the element for placing on the opposite side of the board.  Note that this method does not modify the layer for the element.  That must be done manually.
* `Generate` - For complex elements, perform the calculations required to generate the geometry.  When present, this method should be called before calling `ToKiCad`.

## Common Properties
Most classes provide two important properties:
* `id` - A UUID uniquely identifiying the element.  This is a read only property.  When an element with an `id` property is copied (deepcopy) the new element is given it's own unique ID. 
* `net` - The schematic net to which the element belongs for electrical connectivity.

## Instantiation
Unless otherwise noted, classes are instantiated with an empty constructor, then properties are configured as needed.  Most constructors do not take arguments.

## Units
* All distances, lengths and positions are millimeters.
* All angles are in radians.

# Basic KiCAD Element Classes
## Point
2D Point class.

## Track
## Via
## Segment
## Arc
## GrText
## GrCircle
Used to draw circles on the silkscreen layers, or to create holes on the cuts layer.  Used by [MultiPhaseCoil](#MultiPhaseCoil) to create center and radial holes.
## Group
Container class to group elements together.  Groups can contain other groups createing a hierarchy of elements.

# Coil Classes
## SectorCoil
Creates an angular sector coil. 
Coil is constructed for basic KiCAD elements classes: [Segment](#segment) and [Arc](#arc).

The coil is created spanning the given angle between the inside and outside diameters.  The actual number of turns generated is depenedent upon the coil track width and spacing, as well as the diameter of the coil center via.

## MultiPhaseCoil
This class is the high level multi-phase coil generator.  

### Full Example

1. Instantiate the class, and set basic size parameters

    ````
    coil = MultiPhaseCoil()
    coil.dia_outside = 30
    coil.dia_inside  = 10
    coil.width       = 0.2
    coil.spacing     = 0.2
    ````
    * `dia_outside` - Diameter of the outside edge of the outermost coil arc track. Note that a small leader for each coil extends outside of this diameter.
    * `dia_inside` - Inside diameter of the inner edge of the innermost coil arc.
    * `width` - Width of coil tracks.
    * `spacing` - Spacing between coil tracks.

1. Set the number of coil phases to be created.

    ```
    coil.nets         = [1,2,3]
    coil.multiplicity = 2
    ```
    * `nets`
        * The nets to which the coils will be long.  
        * These nets should already be present in the PCB due to being imported from the schematic.
        * The number of nets listed determines the number of phases the overall multiphase coil will have. If three nets are provided, a three phase coil is generated, per standard brushless DC motors.  A two phase coil is standard for stepper motors.
    * `multiplicity` - The number of coils per phase.  A multiplicity of 1 and 3 nets yields a coil with three phases: A,B,C.  Using a multiplicity of 2 yields a three phase coil with phases of: A+, A-, B+, B-, C+ and C-.  A coil with 2 nets and multiplicity of two yields coils of: A+, A-, B+ and B-.

1. Set the PCB layers on which the coil exists.  

    ```
    coil.layers = ["F.Cu","B.Cu"]
    ```
    * `layers` - Layers for the coil.  For a 1 layer board, specify only the top layer.  For a two layer board, specify the top and bottom copper layers.  A via will be generated at the center of the coil to link the top and bottom coils of each phase.  Currently only 1 and 2 layer boards are supported.

1. If using a two layer boards, define the via linking the layers and set it.

    ```
    v = Via()
    v.size  = 0.7
    v.drill = 0.3
    coil.via = v
    ```
    * `via` - [Via](#via) prototype to use for creating vias.  All vias will be created as copies of this via.

1. If a center hole is desired, define a [GrCircle](#grcircle) for center hole.

    ```
    hole = GrCircle()
    hole.diameter = 5
    coil.center_hole = hole
    ```
    * `center_hole` - [GrCircle](#grcircle) element defining the center hole for the board.
        * The hole value defaults to `None`.  If set as `None`, no center hole is created.
        * The center hole diameter must be smaller than the inside diameter of the coil.
        * When set, a copy of the [GrCircle](#grcircle) is made, and several properties are set:
            * The circle is placed on the edge cuts board layer.
            * The circle is positioned in the center of the coil.
            * The circle trace width is set to 0.1

1. If exterior mounting holes are desired, these are defined and set.

    ```
    hole = GrCircle()
    hole.diameter = 3
    coil.mount_hole = hole
    coil.mount_hole_pattern_diameter = 35
    coil.mount_hole_pattern_angles = (np.array([0, 120, 240]) + 30) * np.pi / 180
    ```
    * `mount_hole` - [GrCircle](#grcircle) element defining the mounting hole.  Like the center hole, a copy of this element is made when set.  For each mounting hole created, a new copy of the circle element is made.  Like the center hole, the layer, width and positions are all set automatically.
    * `mount_hole_pattern_diameter` - Diameter of the mounting hole patter.  All mounting hole centers are placed at this diameter.  Note that the diameter must be large enough that the hole edges will not contact the outside diameter of the coil traces.
    * `mount_hole_pattern_angles` - Angles at which to place the mounting holes.  One hole will be created for each angle in the array.
    
    > **_NOTE_**: In this example, a Numpy array of angles is used to allow us to easily do math.  The holes are placed at 0, 120 and 240 degrees, then rotated another 30 degrees so that they won't align with the coil leaders.  Finally, the units are converted to radians by multiplying by `np.pi/180`.

1. Geneate the coil geometry.

    ```
    coil.Generate()
    ```
    * `Generate()` - Generates all geometry needed for the `MultiPhaseCoil`.

1. Move the geometry to the target coil center position as desired.  Note, this step can be performed manually by dragging the `MultiPhaseCoil` group object in the PCB editor like any other layout element.

    ```
    coil.Translate(100,50)
    ```
    * `Translate(x,y)` - Moves the coil from its current position, default of (0,0), by the values specified.  Note that this is a relative translation, not absolute.  

    > **_NOTE:_** The `MultiPhaseCoil` object also supports a `Rotate` method.  However, since coils are round, this is rarely needed.

1. Geneate the KiCAD text data.

    ```
    s = coil.ToKiCad()
    print(s)
    ```

    The generates KiCAD text to be copied into the `.kicad_pcb` file.  This text can be placed just before the last `)` in the file.  

    Note that coils contain many elements and may generate many kilobytes of text data.  It may be easier to dump the textual data into a file, then 'Copy All' from your favorite text editor before pasting in the the PCB file.

    ```
    with open("tmp.txt","w") as fp:
        fp.write(coil.ToKiCad())
    ```