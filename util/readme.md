# Util
Repo wide utility functions are contained here. A few useful utilities are highlighted here.

## check_number
[`check_number.py`](check_number.py) contains the `is_variable_valid` function that will check if the data is/contains any NaN, Inf, or None values. This can be useful for checking and debugging during optimization, when you might get invalid numbers after taking a bad gradient step. This file also has the `unpack_training_error` function that you can use to examine the `training_error.pt` file that `ppo.py` generates when it encounters such bad values in the optimization.

## colors
[`colors.py`](colors.py) contains unicode color constants to print color text. This is mostly used for test and check statements to print warning, error, and passing statements.

## env_factory
[`env_factory.py`](env_factory.py) implements the `env_factory` function which handles creation of the environment objects. It takes in a string name of the environment to create along with a list, namespace, or `argparser` object of environment arguments and returns a partial function that will create the corresponding environment object. For example, it will return the function `env_fn`, and then you new env objects (an arbitrary amount, all with the exact same arguments) can be made by just calling `env_fn()`. The main purpose of this is for use in parallel settings, where the same single env partial function can be passed to multiple workers so each can create their own env object with the exact same intialization paramemters.

However, even in non parallel settings where you only need to make a single env we still recommend using `env_factory`, as it handles all of the imports for you as well. In order to handle loading any env, all your script needs to import is the `env_factory` function. Importing of the necessary env files will happen dynamically and automatically at run time in the `env_factory` function. This is how we handle environment creation and this is what should be used anytime you need to make and env.

## evaluation_factory
[`evaluation_factory.py`](evaluation_factory.py) implements the simulation policy evaluation/visualization functions. There are two types of eval, `simple_eval` and `interactive_eval`. `simple_eval` will just open a visualization window and does not accept any user input, while `interactive_eval` will take in user input as defined by the interactive eval functions defined by the env (see the env [`readme`](../env/README.md) for more details). There is also `simple_eval_offscreen`, which is the same as `simple_eval` without any visualization, only episode length and average reward values will be printed out. This can be useful for evaluating a policy in a remote setting when no visualization is available, like on vLab.

`simple_eval` is also used the env `test` eval option (see [`eval.py`](../eval.py)). You can "evaluate" an env without a policy (it'll just send random actions) with
```
python eval.py test --env-name MyEnv
```
This can be useful for testing to make sure your env works and at least runs before strating any training. You can also specify any env arguments as well, so you can also use it test your new reward functions with
```
python eval.py test --env-name MyEnv --reward-name myreward
```

## nn_factory
[`nn_factory.py`](nn_factory.py) acts very similarly as `env_factory.py` but for NNs instead. You give it a Namespace of args (so args have to already be parsed from the command line) and it will construct and return the specified actor and critic objects. Whenever you need to make an actor/critic should use this function, that way you only have import the single function `nn_factory`. This file also contains the functions for saving and loading policy checkpoint `.pt` files.