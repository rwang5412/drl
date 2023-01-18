# Digit AR Sim
### This module contains the required files to execute `ar-control`, including
- `ar-control` binary file
- Shared library for LLAPI `llapi.so`
- Ctypes wrapper for LLAPI `llapictypes.py`

Together, these files provide an interface to Agility's control stack and API via Agility's Python SDK. `/llapi` allows user to read basic robot observations and send low level commands. However, when using LLAPI mode, most of the API calls will not be available to use, since the user commands handle the robot entirely. 