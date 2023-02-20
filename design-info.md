
# Magnets
6mm dia x 2mm thick disk magnet, axially magnetized.

Max field (at surface, center)
- N35 - 3355 GS (82% of N52)
- N42 - 3660 GS (89% of N52)
- N52 - 4100 GS

## Hall Effect Sensors

Sensitivity
- A1302: 1.3 mV/GS
- 49E: 1.6 mV/GS

> **Note** 
>
> Units: 1 mT = 10 G

Voltage measurement range: ±2.5V off of a 2.5V nominal output with no field.

# Coil
- FEMM estimates ~3.5 mT for a single coil, 1 mm deep, 1A current 
- Center coil dia is ~1.4 mm across
- Was running stepper driver at 600 mA
- MagPyLib was estimating 2x coils (top + bottom) => ~2x the field.

# Design Questions

- [ ] FEMM model prediction for force.
- [ ] Trade curves for different PCB geometry (id, od, trace width/spacing) based on available magnet sizes.
- [ ] FEMM model field vs. Hall measured field -> first pass geometric field correction factor.
- [ ] Hall measured field for magnets, estimate magnet type based on field & distance.
- [ ] Optimal magnet spacing (tangentially) for rotor to inform next design.
