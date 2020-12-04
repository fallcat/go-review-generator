#!/usr/bin/python3
import sys
sys.path.append('.')
sys.path.append('..')

import math
import json
import tensorflow as tf
import numpy as np

from katago.board import Board
from katago.model import Model
from katago import common

import time

def get_model(saved_model_dir):

  (model_variables_prefix, model_config_json) = common.load_model_paths({"saved_model_dir": saved_model_dir,
                                                                         "model_variables_prefix": None,
                                                                         "model_config_json": None})
  name_scope = "swa_model"

  #Hardcoded max board size
  pos_len = 19

  # Model ----------------------------------------------------------------

  with open(model_config_json) as f:
    model_config = json.load(f)

  if name_scope is not None:
    with tf.compat.v1.variable_scope(name_scope):
      model = Model(model_config,pos_len,{})
  else:
    model = Model(model_config,pos_len,{})

  return model, model_variables_prefix, model_config_json


class GameState:
  def __init__(self,board_size):
    self.board_size = board_size
    self.board = Board(size=board_size)
    self.moves = []
    self.boards = [self.board.copy()]



# Moves ----------------------------------------------------------------

def fetch_output(session, model, gs, rules, fetches):
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = gs.board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
  # print("bin_input_data", bin_input_data.shape)
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return [output[0] for output in outputs]

def get_outputs(session, model, gs, rules):
  trunk_output = model.trunk_output

  [
   trunk
  ] = fetch_output(session, model, gs,rules,[
    trunk_output
  ])


  return trunk



def fetch_output_batch(session, model, gss, rules, fetches):
  bin_input_datas = np.zeros(shape=[len(gss)] + model.bin_input_shape, dtype=np.float32)
  global_input_datas = np.zeros(shape=[len(gss)] + model.global_input_shape, dtype=np.float32)
  # start = time.time()
  for i in range(len(gss)):
    # bin_input_data = np.zeros(shape=[1] + model.bin_input_shape, dtype=np.float32)
    # global_input_data = np.zeros(shape=[1] + model.global_input_shape, dtype=np.float32)
    pla = gss[i].board.pla
    opp = Board.get_opp(pla)
    move_idx = len(gss[i].moves)
    model.fill_row_features(gss[i].board,pla,opp,gss[i].boards,gss[i].moves,move_idx,rules,bin_input_datas,global_input_datas,idx=i)
    # bin_input_datas[i] = bin_input_data
    # global_input_datas[i] = global_input_data
  # print("time fill", time.time() - start)
  # start = time.time()
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_datas,
    model.global_inputs: global_input_datas,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  # print("time model", time.time() - start)
  return [output for output in outputs]

def get_outputs_batch(session, model, gss, rules):
  trunk_output = model.trunk_output

  [
   trunk
  ] = fetch_output_batch(session, model, gss,rules,[
    trunk_output
  ])


  return trunk


def extract_features(session, model, board_arr, color):
  """
  Extract features from board (19x19 numpy array) using katago
  :param board_arr: numpy array (n x n)
  :return:
  """
  # start = time.time()
  board_size = 19
  gs = GameState(board_size)

  rules = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5
  }


  def add_board_arr(board_arr):
    assert board_arr.shape == (19, 19)
    for i in range(19):
      for j in range(19):
        if board_arr[i,j] != 0:
          pla = (Board.BLACK if board_arr[i, j] == 1 else Board.WHITE)
          loc = gs.board.loc(i,j)
          gs.board.play(pla, loc)
          gs.moves.append((pla, loc))
          gs.boards.append(gs.board.copy())

  add_board_arr(board_arr)

  # swap because the color we enter is color of the last move instead of the current move
  gs.board.pla = Board.WHITE if color == "b" or color == "B" else Board.BLACK
  # print("time add board", time.time() - start)
  # start = time.time()


  # "genmove"
  outputs = get_outputs(session, model, gs, rules)
  # print("time get outputs", time.time() - start)

  return outputs


def extract_features_batch(session, model, board_arr, color):
  """
  Extract features from board (19x19 numpy array) using katago
  :param board_arr: numpy array (n x n)
  :return:
  """
  # start = time.time()
  rules = {
    "koRule": "KO_POSITIONAL",
    "scoringRule": "SCORING_AREA",
    "taxRule": "TAX_NONE",
    "multiStoneSuicideLegal": True,
    "hasButton": False,
    "encorePhase": 0,
    "passWouldEndPhase": False,
    "whiteKomi": 7.5
  }

  def get_gss(board_arr_, color_):

    board_size = 19
    gs = GameState(board_size)

    def add_board_arr(board_arr_, gs):
      assert board_arr_.shape == (19, 19)
      for i in range(19):
        for j in range(19):
          if board_arr_[i,j] != 0:
            pla = (Board.BLACK if board_arr_[i, j] == 1 else Board.WHITE)
            loc = gs.board.loc(i,j)
            gs.board.play(pla, loc)
            gs.moves.append((pla, loc))
            gs.boards.append(gs.board.copy())
      return gs

    gs = add_board_arr(board_arr_, gs)

    # swap because the color we enter is color of the last move instead of the current move
    gs.board.pla = Board.WHITE if color_ == "b" or color_ == "B" else Board.BLACK
    return gs

  gss = []
  for board_arr_, color_ in zip(board_arr, color):

    gss.append(get_gss(board_arr_, color_))

  # print("time add board", time.time() - start)
  # start = time.time()

  # "genmove"
  outputs = get_outputs_batch(session, model, gss, rules)
  # print("time get outputs", time.time() - start)

  return outputs


def get_xy(loc):
  x = (loc % 19) - 1
  y = (loc // 19) - 1
  return x, y

if __name__ == "__main__":
  board_arr1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  board_arr1 = np.array(board_arr1)
  color1 = 'b'

  board_arr2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  board_arr2 = np.array(board_arr2)
  color2 = 'w'

  saved_model_dir = "katago/trained_models/g170e-b20c256x2-s5303129600-d1228401921/saved_model/"
  model, model_variables_prefix, model_config_json = get_model(saved_model_dir)

  saver = tf.train.Saver(
    max_to_keep=10000,
    save_relative_paths=True,
  )

  with tf.Session() as session:
    saver.restore(session, model_variables_prefix)
    features1 = extract_features(session, model, board_arr1, color1)
    features2 = extract_features(session, model, board_arr2, color2)