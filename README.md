# `pygamry`: programmatic control of Gamry potentiostats

`pygamry` is a Python package for interfacing with Gamry potentiostats, allowing customization and automation of electrochemical measurements.

## Examples
The `measure_scripts` folder provides an example of how one might use `pygamry` to control a potentiostat programmatically. In this example, various Python scripts define different kinds of measurements, which can then be executed from the command line or by another program like LabVIEW. I hope to add more documentation soon.

## Limitations
To the best of my knowledge, there is no way within the Gamry Electrochemistry Toolkit (on which `pygamry` is based) to adjust certain settings that are available in the Framework GUI, such as  full cell/half cell configuration. In addition, the counter electrode potential is always measured at the reference (white) lead, rather than the counter sense (orange) lead. `pygamry`has been tested with Interface 1000 and 5000 models.

## Installation
See `installation.txt` for installation instructions with pip or conda.
