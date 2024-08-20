import copy
from typing import Any, Dict

import numpy as np


class Command:
    """A single command, with a value, bounds, and reset value."""
    # TODO maybe inherit from float to simplify indexing commands class?
    # TODO differentiate setting desired value and actual value, some setters maybe idk

    def __init__(self, name, value, bounds, reset_value):
        self.name = name
        self._value = value # One should never need to access this directly
        self.bounds = bounds
        self.reset_value = reset_value

        # Interactive eval smoothing
        self.user_desired_value = 0.0 # ONLY USED BY INTERACTIVE EVAL # TODO rename this
        self.max_delta = 0.025 # TODO where set this? where do this?

    def reset(self):
        self._value = copy.deepcopy(self.reset_value)

    def reset_user_command(self):
        self.user_desired_value = copy.deepcopy(self.reset_value)

    def update(self, smooth=True):
        """Update the command value based on the user desired value and max delta."""
        if self.user_desired_value is not None and smooth:
            self._value += np.clip((self.user_desired_value - self._value), -self.max_delta, self.max_delta)
        else:
            self._value = copy.deepcopy(self.user_desired_value)

    def __repr__(self):
        return f"Command(name={self.name}, value={self._value}, bounds={self.bounds}, reset_value={self.reset_value}, user_desired_value={self.user_desired_value}"


class Commands:
    """A collection of commands, each with a value, bounds, and reset value."""

    _commands: Dict[str, Command] = {}

    def reset_commands(self):
        for command in self._commands.values():
            command.reset()

    def reset_user_commands(self):
        for command in self._commands.values():
            command.reset_user_command()

    def add(self, name, reset_value, bounds):
        """Add a command to the collection."""

        assert len(bounds) == 2, f"Bounds must be a list of length 2, got {bounds}"
        self._commands[name] = Command(name, reset_value, bounds, reset_value)
        self._commands[name].reset()

    def __getattr__(self, name):
        """Allows getting command values with dot notation, e.g., commands.x_velocity

        Will return the actual VALUE
        """
        command = self._commands.get(name)
        if command is not None:
            return command._value
        raise AttributeError(f"'Commands' object has no attribute '{name}'")

    def __getitem__(self, name):
        """Allows getting command values with dict notation, e.g., commands['x_velocity']

        Will return the WHOLE COMMAND OBJECT
        """
        command = self._commands.get(name)
        if command is not None:
            return command
        raise AttributeError(f"'Commands' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Allows setting command values with dot notation, e.g., commands.x_velocity = 0.5"""
        if name in self._commands:
            self._commands[name]._value = value
        else:
            raise AttributeError(f"'Commands' object has no attribute '{name}'")

    def __len__(self):
        return len(self._commands)

    def __iter__(self):
        return iter(self._commands.values())

    def items(self):
        return self._commands.items()

    def __repr__(self):
        return '\n'.join([f"{command}: {self._commands[command]}" for command in self._commands])
