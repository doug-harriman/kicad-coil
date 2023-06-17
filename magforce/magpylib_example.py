# https://magpylib.readthedocs.io/en/latest/

#from telnetlib import XASCII
import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt

coil1 = magpy.Collection()
for z in np.linspace(-8, 8, 16):
    winding = magpy.current.Loop(
        current=100,
        diameter=10,
        position=(0,0,z),
    )
    coil1.add(winding)

#coil1.show()

# Define geometry for PCB trace
SIDE_LENGTH = 25
SPACER_MULT = 5
line_vertices = []
for i in range(SIDE_LENGTH):
    if(i%2)==0:
        a = [i*SPACER_MULT, 0, 0]
        b = [i*SPACER_MULT, 125, 0] 
    else:
        a = [i*SPACER_MULT, 125, 0]
        b = [i*SPACER_MULT, 0, 0]
    line_vertices.append(a)
    line_vertices.append(b)
    
trace = magpy.current.Line(
    current=2.2,
    vertices=line_vertices,
    position=(-57.5,-62.5,0)
)

trace.show()

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(13, 5))

# create grid
#ts = np.linspace(-20, 20, 20)
ts = np.linspace(-100, 100, 270)
xs  = np.linspace(-100, 100, 270)
zs = np.linspace(-100, 100, 270)
grid = np.array([[(x,0,z) for x in ts] for z in ts])
gridTop = np.array([[(x,y,0) for x in ts] for y in ts])

# compute and plot side-view field of trace
B = magpy.getB(trace, grid)
Bamp = np.linalg.norm(B, axis=2)
Bamp /= np.amax(Bamp)

# compute and plot top-view field of trace
B2 = magpy.getB(trace, gridTop)
Bamp2 = np.linalg.norm(B2, axis=2)
Bamp2 /= np.amax(Bamp2)

#sp = ax1.streamplot(
#    grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
#    density=2,
#    color=Bamp,
#    linewidth=np.sqrt(Bamp)*3,
#    cmap='coolwarm',
#)

# Side view
cp = ax2.contourf(
    grid[:,:,0], grid[:,:,2], Bamp,
    levels=100,
    cmap='jet',
)

X,Y = np.meshgrid(xs,zs)
U,V = B2[:,:,0], B2[:,:,2]
amp = np.sqrt(U**2+V**2)
# Top view
sp = ax1.contourf(
    #grid[:,:,0], grid[:,:,2], Bamp2,
    X, Y, amp,
    np.linspace(-100,100,100),
    levels=100,
    cmap='jet',
)
#ax1.streamplot(
#    grid[:,:,0], grid[:,:,2], B2[:,:,0], B2[:,:,2],
#    density=3,
#    color=Bamp2
#)

ax2.streamplot(
    grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=3,
    color=Bamp,
)

# figure styling
ax1.set(
    title='Magnetic field of current-carrying PCB trace',
    xlabel='X-position [mm]',
    ylabel='Y-position [mm]',
    aspect=1,
)
ax2.set(
    title='Cross-section View',
    xlabel='X-position [mm]',
    ylabel='Z-position [mm]',
    aspect=1,
)

#plt.colorbar(sp.lines, ax=ax1, label='[mT]')
plt.colorbar(sp, ax=ax1, label='[mT]')
plt.colorbar(cp, ax=ax2, label='[mT]')

plt.tight_layout()
# plt.show()
plt.savefig('magplotlib_example.png')