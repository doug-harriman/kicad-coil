# Power

|Signal| AD2 | Stepper Driver |
| ---  | --- | -------------- | 
|  5V  |  V+ | PUL+, DIR+, ENA+ |
| Step |  W1 | PUL- |
| Enable| W2 | ENA- |

Signals are active low.

# Waveforms Config
- Activate master power
- V+ to 5V
- W1 Square wave, 2.5V amplitude, 2.5V offset
- W2 DC, 0V to activate, 5V to disable


# Stepper Drive Config
* 1/16 step
* 0.6 A (runs cooler, less power, ran up teo 3.8A)
* Ran sweeps up to 10kHz drive.  Starting at ~500 Hz

# Magnet Config
* 4x axial magnetized magnets in 2x2 pattern so had 2x N and 2xS permanent poles interacting with the 2ph coils.
* Radially magnetized permantent magnets did not work.


