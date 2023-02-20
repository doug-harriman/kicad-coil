# General Design Data

## Magnets
6mm dia x 2mm thick disk magnet, axially magnetized.

Max field (at surface, center)

|Grade| Field Strength [GS] | Relative to N52 |
| :-: | :-: | :-: |
| N35 | 3355 | 82% |
| N42 | 3660 | 89% |
| N52 | 4100 |  - |

### Hall Effect Sensors

| Sensor | Sensitivity [mV/GS] | Sensitivity [mV/mT] |
| :-: | :-: | :-: |
| A1302| 1.3 | 13 |
| 49E| 1.6 | 16 |

> **Note** 
>
> Units: 1 mT = 10 G

Voltage measurement range: ±2.5V off of a 2.5V nominal output with no field.

## Coil
- FEMM estimates ~3.5 mT for a single coil, 1 mm deep, 1A current 
- Center coil dia is ~1.4 mm across
- Was running stepper driver at 600 mA
- MagPyLib was estimating 2x coils (top + bottom) => ~2x the field.

### Trace Width Limits

Copper weight an min trace thickness & spacing (same value) per [PCB Way](https://www.pcbway.com/).

|Copper Weight (oz)| Minimum Space [mils]| Copper Thickness [um]|
| :-: | :-: | :-: | 
|0.5|	3| |
|1|	3| 34.80 |
|2|	6.5| 69.60|
|3|	10| 104.39
|4|	13| 139.19|
|5|18| 179.99 |
|6|18| 208.79|
|7|23| 243.59|
|8|24| 278.38|
|9|25| 313.18
|10|31.5||

There are various PCB trace temperature rise calculators available that take track width, copper thickness and trace current as inputs.  Good information from Proto Express:

- [Overivew](https://www.protoexpress.com/blog/trace-current-capacity-pcb-design/)
- [IPC-2152 Standard](https://www.protoexpress.com/blog/how-to-optimize-your-pcb-trace-using-ipc-2152-standard/)
- [Thermal Properties of Substrates](https://www.protoexpress.com/blog/comparing-the-manufacturability-of-pcb-laminates/)

FR4 thermal properties
* Conductivity (X&Y axes): 0.9 W/(m-K)
* Conductivity (Z axes): 0.3 W/(m-K)
* Expansion (X&Y axes): 13 ppm/K
* Expansion (Z axes): 70 ppm/K
* Glass transition temperature: 135 to 170 deg C
* Specific heat capacity: 1110 J/(kg-deg C)

## Design Questions

- [ ] FEMM model prediction for force for unit depth rectangular circuit.
- [ ] Trade curves for different PCB geometry (id, od, trace width/spacing) based on available magnet sizes.
- [ ] Impact of Iron on top of magnets.  Trade of magnet/iron thickness ratio vs. overall thickness.
- [ ] Km vs. trace width/spacing and vs. PCB layer count for a given ID/OD, and for viable OD's based on available magnets.
- [ ] FEMM model field vs. Hall measured field -> first pass geometric field correction factor.
    - Geometric errors
        - Magnets are round
        - Coil conductors are parallel to each other for maximum packing factor, but do not intersect the center of rotation.
        - Could update FEMM model to look at tangential slice => each coil would have different width and spacing.
    - Future: ELMER for true geometric impact assessment.
- [ ] Hall measured field for magnets, estimate magnet type based on field & distance.
- [ ] Optimal magnet spacing (tangentially) for rotor to inform next design.
- [ ] What are limits on copper thickness vs. PCB thickness vs. trace width.
