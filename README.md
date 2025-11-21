# ECE449-project

### Group:
| ***Name** |
| -------- |
| Jeremy Carefoot |
| Raashid Hamdan |
| Zeeshan Haque |

## Quick-Start Guide

To setup project and run program:

Install dependencies with
```bash
make
``` 

Run program with
```bash
./run.sh [args]
```

Note: Python version >=3.10 required. Ensure that tkinter is installed with your base python installation, otherwise you may run into dependency errors.
*This comes pre-installed by default unless python was installed with a package manager like Homebrew on macOS*

## Development Information

### Controllers

Controllers can be found in the `/code/controllers` directory. These python files are the fuzzy-system implementations that will actually control the game. (see the KesslerGame guide for more information on how controllers work). 

**The goal of this project is to implement our own custom controller**

### Scenarios

Scenarios can be found in the `/code/scenarios` directory. These python files are the **drivers** for the KesslerGame. A scenario is essentially a definition of initial conditions for the game to run, as well as game settings and controllers to use.

### main.py

This is the entrypoint for this program. Import and call whichever scenario you would like to run from here. (*scenarios should be implemented outside of this file*)
