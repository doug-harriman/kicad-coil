# Circuit Properties

* Top and bottom coil should have:
    * Same current
    * Same current direction (sign)

# Force calcs

* Go to analysis window: "eyeglasses" icon button
    * This has to be done after every analysis.  Tab at bottom shows previous analysis results.
* Click the boundary for which you want the force calculation.
    * Ex: For the force on the magnet, click inside the magnet.
    * This should turn the inside of that boundary item green.
* Select the "Integrate" menu item.
* Select "Force via Weighted Stress Tensor"
* Click OK.
* Returns forces acting on the boundary in X & Y directions.
* Note: Force is returned for the geometry with the Z direction (into screen) depth as specified in the problem definition (accessed via the "Problem" menu in the setup window).
    * Ex: if the depth is 1 mm, but the real parts are 6mm deep, multiply the force value by 6.

# Multi-Sim via Python

See Meindert's report: https://simplexity.atlassian.net/wiki/spaces/SE/pages/2520711717/FEMM+Magnetic+FEA+Analysis