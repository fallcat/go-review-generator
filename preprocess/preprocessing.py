"""Show the position from an SGF file.
This demonstrates the sgf and ascii_boards modules.
"""

import sys
from optparse import OptionParser

from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves

import numpy as np
import pickle

color_to_num = { None: 0,
                 'b': 1,
                 'w': -1}

def show_sgf_file(pathname, save_filename):


    try:
        input_file = open(pathname, "rb")
        sgf_src = input_file.read()
        input_file.close()
        sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
    except ValueError:
        try:
            input_file = open(pathname, "rt")
            sgf_src = input_file.read()
            input_file.close()
            sgf_game = sgf.Sgf_game.from_string(sgf_src)
        except ValueError:
            raise Exception("bad sgf file")

    board_size = sgf_game.get_size()
    if board_size != 19:
        raise Exception(f"board size not 19. board size is {board_size}")
    board_array = np.zeros((board_size, board_size))
    board_array_list = []
    comments_list = []
    steps = []

    winner = sgf_game.get_winner()
    board_size = sgf_game.get_size()
    root_node = sgf_game.get_root()
    try:
        b_player = root_node.get("PB")
    except:
        b_player = ""
    try:
        w_player = root_node.get("PW")
    except:
        w_player = ""
    for i, node in enumerate(sgf_game.get_main_sequence()):
        try:
            color, move = node.get_move()
            row, col = move
            board_array[row, col] = color_to_num[color]
            try:
                comment = node.get('C')
                comments_list.append(comment)
                board_array_list.append((row, col, color, board_array.copy()))
                steps.append(i)
            except:
                pass
        except:
            continue

    data = {'boards': board_array_list, 'comments': comments_list, 'steps': steps}

    if save_filename is not None:
        with open(save_filename, 'wb') as output_file:
            pickle.dump(data, output_file)

_description = """\
Show the position from an SGF file. If a move number is specified, the position
before that move is shown (this is to match the behaviour of GTP loadsgf).
"""

def main(argv):
    parser = OptionParser(usage="%prog <filename> [move number]",
                          description=_description)
    opts, args = parser.parse_args(argv)
    if not args:
        parser.error("not enough arguments")
    pathname = args[0]
    if len(args) > 2:
        parser.error("too many arguments")
    if len(args) == 2:
        try:
            save_filename = args[1]
        except ValueError:
            parser.error("invalid save_filename: %s" % args[1])
    else:
        save_filename = None
    try:
        show_sgf_file(pathname, save_filename)
    except Exception as e:
        print("show_sgf:", str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
