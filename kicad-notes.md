# KiCAD Notes

# Guides

NOTE: All guides KiCAD version 6+ unless otherwise noted.

## [Beginner’s Tutorial to Schematic and PCB Design](https://circuitstate.com/tutorials/getting-started-with-kicad-version-6-beginners-tutorial-to-schematic-and-pcb-design/)

General overview of PCB elements and how to create them with KiCAD.

## [How to Install KiCad Version 6 and Organize Part Libraries](https://circuitstate.com/tutorials/how-to-install-kicad-version-6-and-organize-part-libraries/)

Have not read as of 12-Jun-2022.

## [Get Your PCB Ready for Fab](https://circuitstate.com/tutorials/how-to-get-your-kicad-pcb-design-ready-for-fabrication-kicad-version-6-tutorial/) 

Overview of which files need to be generated, and rules checking tools within KiCAD to make sure there are no PCB errors.  A few manufacturer specific rules and hints.

## [Digikey Intro to KiCAD]()https://www.youtube.com/playlist?list=PLEBQazB0HUyR24ckSZ5u05TZHV9khgA1O

VERSION5.  Full video series on producing a PCBA with KiCAD.

# PCB/PCBA Manufacturers

## [PCBWay](https://www.pcbway.com/)

* [Design Rules](https://www.pcbway.com/capabilities.html#:~:text=PCB%20Capabilities%20%2D%20Quick%2Dturn%20PCB)
* [Only accepts RS-274D format Gerbers](https://circuitstate.com/tutorials/how-to-get-your-kicad-pcb-design-ready-for-fabrication-kicad-version-6-tutorial/)

## JLPCB
## OSHPARK

# KiCAD Plugins

[Curated list of Plugins](https://github.com/joanbono/awesome-kicad)

* [Interactive HTML BOM](https://github.com/openscopeproject/InteractiveHtmlBom) - BOM list _and_ board view, with highlighting of the PBC comonents on the board when the BOM item is selected in the list.
* [Board2Pdf](https://gitlab.com/dennevi/Board2Pdf/) - Finer control of Board (i.e. layout, not schematic) PDF generation.
* [KiCAD Teardrops](https://github.com/stimulu/kicad-teardrops)
    * Converts vias to teardrop vias.  
    * Provides a GUI.
    * Might also be a good source for understanding how someone else generated code.