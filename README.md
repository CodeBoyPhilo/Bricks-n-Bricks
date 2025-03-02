# Bricks'n Bricks (砖了个砖)

This is a replicate of the game 砖了个砖 (direct translate to 'Bricks and Bricks') using python and plays within the CLI.
The original game is held as a _mini program_ on WeChat, so you would need a WeChat account to experience it.
This project is only infilled with the minimal features of the original game but still respects the rules of the original game.
The purpose of this project is actually to help myself developing a quick solver for the original game.

# Requirements

- `python 3.11`
- `numpy 1.24.1`
- `pandas 2.1.4`
- `opencv-python 4.6.0.66`

# How to play

The game has the following parameters:

- `-p`, `--path`: the path of a screenshot of any game of the official game. You would need to capture the screenshot from the official game on WeChat. Otherwise, you could use the `test1.jpg` contained in this repo.
- `-r`, `--resume`: whether to resume a game or not. Useful when you unexpectedly quit a game and want to resume it.
- `--history`: the path of a **.csv** file that stores the moves of the game you would like to resume. The game automatically generates one once you play and safely terminates it with `e` (for 'exit'). You could follow the format specified in the `test1_history.csv` file to hand-craft one yourself.

To start the game:

```python
python bricksnbricks.py -p test1.jpg
```

To resume the game:

```python
python bricksnbricks.py -p test1.jpg -r --history test1_history.csv
```

# More to come
