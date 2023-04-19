from util.colors import BLUE, WHITE, ORANGE, FAIL, ENDC
class GenericEnv(object):

    """
    Define generic environment functions that are needed for RL. Should define (not implement) all 
    of the functions that sampling uses.
    """

    def __init__(self):
        self.observation_size = None
        self.action_size = None
        self.input_keys_dict = {} 
        self.control_commands_dict = {}

    def reset_simulation(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step_simulation(self, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError
    
    def get_action_mirror_indices(self):
        raise NotImplementedError

    def get_observation_mirror_indices(self):
        raise NotImplementedError

    def display_controls_menu(self,):
        """
        Method to pretty print menu of available controls.
        """
        def print_command(char, info, color=ENDC):
            char += " " * (10 - len(char))
            print(f"{color}{char}\t{info}{ENDC}")

        if ((type(self.input_keys_dict) is dict) and (len(self.input_keys_dict) > 0)):
            print("")
            print_command("Key", "Function", color=BLUE)
            for key, value in self.input_keys_dict.items():
                assert isinstance(key, str) and len(key) == 1, (
                    f"{FAIL}input_keys_dict key must be a length-1 string corresponding \
                    to the desired keybind{ENDC}")
                assert isinstance(value, str), (
                    f"{FAIL}control command description must be of type string{ENDC}")
                print_command(key, value, color=WHITE)
            print("")

    def display_control_commands(self,):
        """
        Method to pretty print menu of current commands.
        """
        def print_command(char, info, color=ENDC):
            char += " " * (10 - len(char))
            print(f"{color}{char}\t{info}{ENDC}")

        if ((type(self.input_keys_dict) is dict) and (len(self.input_keys_dict)>0)):
            print("")
            print_command("Control Input", "Commanded value",color=BLUE)
            for key, value in self.control_commands_dict.items():
                assert type(key) is str, (
                    f"{FAIL}ctrl_dict key must be of type string{ENDC}")
                print_command(key, value, color=WHITE)
            print("")
            # backspace the number of lines used to print the commanded value table
            # in order to update values without printing a new table to terminal at every step
            # equal to the length of ctrl_dict plus all other prints for the table, i.e table header
            print(f"\033[{len(self.control_commands_dict)+3}A\033[K", end='\r') 