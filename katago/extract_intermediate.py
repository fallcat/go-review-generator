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

saved_model_dir = "katago/trained_models/g170e-b20c256x2-s5303129600-d1228401921/saved_model/"
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
policy0_output = tf.nn.softmax(model.policy_output[:,:,0])
policy1_output = tf.nn.softmax(model.policy_output[:,:,1])
value_output = tf.nn.softmax(model.value_output)
scoremean_output = 20.0 * model.miscvalues_output[:,0]
scorestdev_output = 20.0 * tf.math.softplus(model.miscvalues_output[:,1])
lead_output = 20.0 * model.miscvalues_output[:,2]
vtime_output = 150.0 * tf.math.softplus(model.miscvalues_output[:,3])
ownership_output = tf.tanh(model.ownership_output)
scoring_output = model.scoring_output
futurepos_output = tf.tanh(model.futurepos_output)
seki_output = tf.nn.softmax(model.seki_output[:,:,:,0:3])
seki_output = seki_output[:,:,:,1] - seki_output[:,:,:,2]
seki_output2 = tf.sigmoid(model.seki_output[:,:,:,3])
scorebelief_output = tf.nn.softmax(model.scorebelief_output)
sbscale_output = model.sbscale3_layer
outputs_by_layer = model.outputs_by_layer

class GameState:
  def __init__(self,board_size):
    self.board_size = board_size
    self.board = Board(size=board_size)
    self.moves = []
    self.boards = [self.board.copy()]



# Moves ----------------------------------------------------------------

def fetch_output(session, gs, rules, fetches):
  bin_input_data = np.zeros(shape=[1]+model.bin_input_shape, dtype=np.float32)
  global_input_data = np.zeros(shape=[1]+model.global_input_shape, dtype=np.float32)
  pla = gs.board.pla
  opp = Board.get_opp(pla)
  move_idx = len(gs.moves)
  model.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=0)
  outputs = session.run(fetches, feed_dict={
    model.bin_inputs: bin_input_data,
    model.global_inputs: global_input_data,
    model.symmetries: [False,False,False],
    model.include_history: [[1.0,1.0,1.0,1.0,1.0]]
  })
  return [output[0] for output in outputs]

def get_outputs(session, gs, rules):
  [policy0,
   policy1,
   value,
   scoremean,
   scorestdev,
   lead,
   vtime,
   ownership,
   scoring,
   futurepos,
   seki,
   seki2,
   scorebelief,
   sbscale,
   layers
  ] = fetch_output(session,gs,rules,[
    policy0_output,
    policy1_output,
    value_output,
    scoremean_output,
    scorestdev_output,
    lead_output,
    vtime_output,
    ownership_output,
    scoring_output,
    futurepos_output,
    seki_output,
    seki_output2,
    scorebelief_output,
    sbscale_output,
    outputs_by_layer
  ])
  board = gs.board

  moves_and_probs0 = []
  for i in range(len(policy0)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy0)-1:
      moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs0.append((move,policy0[i]))

  moves_and_probs1 = []
  for i in range(len(policy1)):
    move = model.tensor_pos_to_loc(i,board)
    if i == len(policy1)-1:
      moves_and_probs1.append((Board.PASS_LOC,policy1[i]))
    elif board.would_be_legal(board.pla,move):
      moves_and_probs1.append((move,policy1[i]))

  ownership_flat = ownership.reshape([model.pos_len * model.pos_len])
  ownership_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        ownership_by_loc.append((loc,ownership_flat[pos]))
      else:
        ownership_by_loc.append((loc,-ownership_flat[pos]))

  scoring_flat = scoring.reshape([model.pos_len * model.pos_len])
  scoring_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        scoring_by_loc.append((loc,scoring_flat[pos]))
      else:
        scoring_by_loc.append((loc,-scoring_flat[pos]))

  futurepos0_flat = futurepos[:,:,0].reshape([model.pos_len * model.pos_len])
  futurepos0_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        futurepos0_by_loc.append((loc,futurepos0_flat[pos]))
      else:
        futurepos0_by_loc.append((loc,-futurepos0_flat[pos]))

  futurepos1_flat = futurepos[:,:,1].reshape([model.pos_len * model.pos_len])
  futurepos1_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        futurepos1_by_loc.append((loc,futurepos1_flat[pos]))
      else:
        futurepos1_by_loc.append((loc,-futurepos1_flat[pos]))

  seki_flat = seki.reshape([model.pos_len * model.pos_len])
  seki_by_loc = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      if board.pla == Board.WHITE:
        seki_by_loc.append((loc,seki_flat[pos]))
      else:
        seki_by_loc.append((loc,-seki_flat[pos]))

  seki_flat2 = seki2.reshape([model.pos_len * model.pos_len])
  seki_by_loc2 = []
  board = gs.board
  for y in range(board.size):
    for x in range(board.size):
      loc = board.loc(x,y)
      pos = model.loc_to_tensor_pos(loc,board)
      seki_by_loc2.append((loc,seki_flat2[pos]))

  moves_and_probs = sorted(moves_and_probs0, key=lambda moveandprob: moveandprob[1], reverse=True)
  #Generate a random number biased small and then find the appropriate move to make
  #Interpolate from moving uniformly to choosing from the triangular distribution
  alpha = 1
  beta = 1 + math.sqrt(max(0,len(gs.moves)-20))
  r = np.random.beta(alpha,beta)
  probsum = 0.0
  i = 0
  genmove_result = Board.PASS_LOC
  while True:
    (move,prob) = moves_and_probs[i]
    probsum += prob
    if i >= len(moves_and_probs)-1 or probsum > r:
      genmove_result = move
      break
    i += 1

  return {
    "policy0": policy0,
    "policy1": policy1,
    "moves_and_probs0": moves_and_probs0,
    "moves_and_probs1": moves_and_probs1,
    "value": value,
    "scoremean": scoremean,
    "scorestdev": scorestdev,
    "lead": lead,
    "vtime": vtime,
    "ownership": ownership,
    "ownership_by_loc": ownership_by_loc,
    "scoring": scoring,
    "scoring_by_loc": scoring_by_loc,
    "futurepos": futurepos,
    "futurepos0_by_loc": futurepos0_by_loc,
    "futurepos1_by_loc": futurepos1_by_loc,
    "seki": seki,
    "seki_by_loc": seki_by_loc,
    "seki2": seki2,
    "seki_by_loc2": seki_by_loc2,
    "scorebelief": scorebelief,
    "sbscale": sbscale,
    "genmove_result": genmove_result,
    "layers": layers
  }


def extract_features(board_arr, color):
  """
  Extract features from board (19x19 numpy array) using katago
  :param board_arr: numpy array (n x n)
  :return:
  """
  saver = tf.train.Saver(
    max_to_keep=10000,
    save_relative_paths=True,
  )
  with tf.Session() as session:
    saver.restore(session, model_variables_prefix)

    known_commands = [
      'boardsize',
      'clear_board',
      'showboard',
      'komi',
      'play',
      'genmove',
      'quit',
      'name',
      'version',
      'known_command',
      'list_commands',
      'protocol_version',
      'gogui-analyze_commands',
      'setrule',
      'policy',
      'policy1',
      'logpolicy',
      'ownership',
      'scoring',
      'futurepos0',
      'futurepos1',
      'seki',
      'seki2',
      'scorebelief',
      'passalive',
    ]
    known_analyze_commands = [
      'gfx/Policy/policy',
      'gfx/Policy1/policy1',
      'gfx/LogPolicy/logpolicy',
      'gfx/Ownership/ownership',
      'gfx/Scoring/scoring',
      'gfx/FuturePos0/futurepos0',
      'gfx/FuturePos1/futurepos1',
      'gfx/Seki/seki',
      'gfx/Seki2/seki2',
      'gfx/ScoreBelief/scorebelief',
      'gfx/PassAlive/passalive',
    ]

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

    layerdict = dict(model.outputs_by_layer)
    weightdict = dict()
    for v in tf.trainable_variables():
      weightdict[v.name] = v

    layer_command_lookup = dict()

    def add_extra_board_size_visualizations(layer_name, layer, normalization_div):
      assert (layer.shape[1].value == board_size)
      assert (layer.shape[2].value == board_size)
      num_channels = layer.shape[3].value
      for i in range(num_channels):
        command_name = layer_name + "-" + str(i)
        command_name = command_name.replace("/", ":")
        known_commands.append(command_name)
        known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
        layer_command_lookup[command_name.lower()] = (layer, i, normalization_div)

    def add_layer_visualizations(layer_name, normalization_div):
      if layer_name in layerdict:
        layer = layerdict[layer_name]
        add_extra_board_size_visualizations(layer_name, layer, normalization_div)

    add_layer_visualizations("conv1", normalization_div=6)
    add_layer_visualizations("rconv1", normalization_div=14)
    add_layer_visualizations("rconv2", normalization_div=20)
    add_layer_visualizations("rconv3", normalization_div=26)
    add_layer_visualizations("rconv4", normalization_div=36)
    add_layer_visualizations("rconv5", normalization_div=40)
    add_layer_visualizations("rconv6", normalization_div=40)
    add_layer_visualizations("rconv7", normalization_div=44)
    add_layer_visualizations("rconv7/conv1a", normalization_div=12)
    add_layer_visualizations("rconv7/conv1b", normalization_div=12)
    add_layer_visualizations("rconv8", normalization_div=48)
    add_layer_visualizations("rconv9", normalization_div=52)
    add_layer_visualizations("rconv10", normalization_div=55)
    add_layer_visualizations("rconv11", normalization_div=58)
    add_layer_visualizations("rconv11/conv1a", normalization_div=12)
    add_layer_visualizations("rconv11/conv1b", normalization_div=12)
    add_layer_visualizations("rconv12", normalization_div=58)
    add_layer_visualizations("rconv13", normalization_div=64)
    add_layer_visualizations("rconv14", normalization_div=66)
    add_layer_visualizations("g1", normalization_div=6)
    add_layer_visualizations("p1", normalization_div=2)
    add_layer_visualizations("v1", normalization_div=4)

    input_feature_command_lookup = dict()

    def add_input_feature_visualizations(layer_name, feature_idx, normalization_div):
      command_name = layer_name
      command_name = command_name.replace("/", ":")
      known_commands.append(command_name)
      known_analyze_commands.append("gfx/" + command_name + "/" + command_name)
      input_feature_command_lookup[command_name] = (feature_idx, normalization_div)

    for i in range(model.bin_input_shape[1]):
      add_input_feature_visualizations("input-" + str(i), i, normalization_div=1)

    linear = tf.cumsum(tf.ones([19], dtype=tf.float32), axis=0, exclusive=True) / 18.0
    color_calibration = tf.stack(axis=0, values=[
      linear,
      linear * 0.5,
      linear * 0.2,
      linear * 0.1,
      linear * 0.05,
      linear * 0.02,
      linear * 0.01,
      -linear,
      -linear * 0.5,
      -linear * 0.2,
      -linear * 0.1,
      -linear * 0.05,
      -linear * 0.02,
      -linear * 0.01,
      linear * 2 - 1,
      tf.zeros([19], dtype=tf.float32),
      linear,
      -linear,
      tf.zeros([19], dtype=tf.float32)
    ])
    add_extra_board_size_visualizations("colorcalibration", tf.reshape(color_calibration, [1, 19, 19, 1]),
                                        normalization_div=None)

    # pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
    # loc = parse_coord(command[2], gs.board)
    # gs.board.play(pla, loc)
    # gs.moves.append((pla, loc))
    # gs.boards.append(gs.board.copy())

    def add_board_arr(board_arr):
      assert board_arr.shape == (19, 19)
      for i in range(19):
        for j in range(19):
          if board_arr[i,j] != 0:
            pla = (Board.BLACK if board_arr[i, j] == 1 else Board.WHITE)
            loc = gs.board.loc(i,j)


    add_board_arr(board_arr)

    gs.board.pla = Board.BLACK if color == "b" or color == "B" else Board.WHITE

    # "genmove"
    outputs = get_outputs(session, gs, rules)
    print(outputs.keys())
    for k in outputs.keys():
      if 'loc' not in k and 'moves_and_probs' not in k:
        try:
          print(k, outputs[k].shape)
        except:
          print(k, outputs[k])
    return outputs


if __name__ == "__main__":
  board_arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
  board_arr = np.array(board_arr)
  color = 'b'
  extract_features(board_arr, color)