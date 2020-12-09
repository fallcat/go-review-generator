import sys
sys.path.append('.')
sys.path.append('..')

import argparse

from tqdm import tqdm

import tensorflow as tf
import katago
from katago.board import IllegalMoveError

from pretrained_combine.go_dataset_pretrained_fast import GoDataset
from torch.utils.data import DataLoader

from transformer_encoder import get_comment_features
from transformer_encoder.model import *

from pretrained_combine.model import PretrainedCombineModel
from pretrained_combine.utils import restore


def parse_args():
    parser = argparse.ArgumentParser()

    # paths and info
    parser.add_argument('--project-name', type=str, default='go-review-matcher', help='Project name')
    parser.add_argument('-d', '--data-dir', type=str, default='data_splits_final', help='Directory of data')
    parser.add_argument('--katago-model-dir', type=str,
                        default='katago/trained_models/g170e-b10c128-s1141046784-d204142634/saved_model/',
                        help='Directory of katago model')
    parser.add_argument('--restore-dir', type=str, default='experiments/exp01',
                        help='Directory to restore the experiment')

    # testing config
    parser.add_argument('--split', type=str, default='test',
                        help='Which split to use to train. Could use val for debugging')
    parser.add_argument('--portion', type=float, default=1, help='Train on only a portion of data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for each step')
    parser.add_argument('--average-checkpoints', type=int, default=5, help='Max number of checkpoints to average')
    parser.add_argument('--seed', type=int, default=42, help='Manual seed for torch')

    # text config
    parser.add_argument('--emsize', type=int, default=200, help='embedding dimension for text')
    parser.add_argument('--nhid', type=int, default=100,
                        help='the dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout value')
    parser.add_argument('--sentence-len', type=int, default=100, help='sentence len')
    parser.add_argument('--text-hidden-dim', type=int, default=200, help='text hidden dim')

    # combine model config
    parser.add_argument('--combine', type=str, default='concat', choices=['concat', 'dot', 'attn'],
                        help='Hidden dim size for the combine model')
    parser.add_argument('--d-model', type=int, default=512, help='Hidden dim size for the combine model')
    parser.add_argument('--combine-num-heads', type=int, default=4,
                        help='Num heads in the multiheaded attention in combine if have chosen attn as combine type')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='Dropout rate for the combine model')
    parser.add_argument('--board-embed-size', type=int, default=128, help='Size of board embedding. '
                                                                          'This depends on which katago model you use.')

    return parser


def evaluate(session, combine_model, board_model, text_model, val_dataloader, batch_size, device):
    print('------ evaluate ------')
    combine_model.eval()
    batches = tqdm(enumerate(val_dataloader))
    num_batches = int(len(val_dataloader.dataset) / batch_size)
    total_correct = 0
    total_total = 0
    for i_batch, sampled_batched in batches:
        batches.set_description(f'Evaluate batch {i_batch}/{num_batches}')
        bin_input_datas = sampled_batched[1]['bin_input_datas'].numpy()
        global_input_datas = sampled_batched[1]['global_input_datas'].numpy()
        text = sampled_batched[1]['text']
        label = sampled_batched[1]['label'][:, None].to(device)

        board_features = torch.tensor(
            katago.extract_intermediate.fetch_output_batch_with_bin_input(session, combine_model,
                                                                          bin_input_datas,
                                                                          global_input_datas)).to(device)

        text_features = torch.tensor(get_comment_features.extract_comment_features(text_model, text.to(device), batch_size, device)).to(device)
        logits = combine_model(board_features, text_features)
        pred = logits >= 0.5
        correct = sum(pred == label)
        total = label.shape[0]
        total_correct += correct
        total_total += total
    accuracy = float(total_correct) / total_total
    combine_model.train()
    print(f'Validation accuracy: {accuracy}')
    return accuracy


def main():
    parser = parse_args()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    config = {'data_dir': args.data_dir,
              'device': device,
              'portion': args.portion,
              }
    test_set = GoDataset(config, split=args.split)

    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load pretrained katago model
    board_model, model_variables_prefix, model_config_json = katago.get_model(args.katago_model_dir)
    saver = tf.train.Saver(
        max_to_keep=10000,
        save_relative_paths=True,
    )

    # Load pretrained text Tranformer model
    ntokens = test_set.vocab_size # the size of vocabulary

    text_model = TransformerModel_extractFeature(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    if torch.cuda.is_available():
        text_model = text_model.cuda()
    for param in text_model.parameters():
        param.requires_grad = False  # freeze params in the pretrained model

    # Construct the model
    combine_model = PretrainedCombineModel(combine=args.combine,d_model=args.d_model, dropout_p=args.dropout_p,
                                           sentence_len=args.sentence_len, text_hidden_dim=args.text_hidden_dim,
                                           board_embed_size=args.board_embed_size, num_heads=args.combine_num_heads)
    if torch.cuda.is_available():
        combine_model = combine_model.cuda()

    restore_modules = {'combine_model': combine_model}

    epoch, step = restore(
        args.restore_dir,
        restore_modules,
        num_checkpoints=args.average_checkpoints,
        map_location=device.type,
        strict=True
    )

    with tf.Session() as session:
        saver.restore(session, model_variables_prefix)
        test_acc = evaluate(session, combine_model, board_model, text_model, test_dataloader,
                                    args.batch_size, device)

        print(f"Test accuracy: {test_acc}")


        print("----------Finished evaluating----------")


if __name__ == '__main__':
    main()
