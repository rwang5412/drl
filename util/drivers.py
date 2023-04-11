from pynput import keyboard
import time 
import queue

class Keyboard():
    def __init__(self) -> None:
        # make a thread to listen to keyboard and register our callback functions
        self.listener = keyboard.Listener(on_press=self.on_press)

        # start listening
        self.listener.start()
        # queue to store keyboard commands
        self.command_queue = queue.Queue(maxsize=1)

    def on_press(self, key):
        """
        Adds the last keyboard input to the length-1 command queue 
        """
        self.command_queue.put(key.char)
    
    def get_input(self,):
        if self.command_queue.empty:
            return None
        else:
            command = self.command_queue.get()
            return command