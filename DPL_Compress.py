import time
import os
import argparse
import logging
import logging.config
import torch
import torch.backends.cudnn as cudnn

from operator import attrgetter
from configparser import ConfigParser

from utils.watcher import ActivationWatcher
from utils.reshape import reshape_weight, reshape_back_weight
from utils.utils import compute_size
from utils import dataloader
from train_and_eval import evaluate
import model
import DPL

def loggerWarpper(logger, func, *argv):
    formLevel = logger.level
    if logger.level is not logging.INFO:
        logger.level = logging.INFO
    results = func(*argv)
    logger.setLevel(formLevel)
    return results

parser = argparse.ArgumentParser(description='DPL_Compression')
parser.add_argument('--data-path', default='data/imagenet', 
                    help='Path to dataset', type=str)
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='Dataset evaluate')
parser.add_argument('--model', default='resnet18', type=str, 
                    help='Model to compress')
parser.add_argument('--model_path', default=None, type=str,
                    help='.pth file path to load pretrained model')
parser.add_argument('--n-workers', default=4, type=int,
                    help='Number of workers for data loading')
parser.add_argument('--batch-size', default=256, type=int,
                    help='Batch size for val data loading')
parser.add_argument('--distributed', default=False, type=bool,
                    help='For multiprocessing distributed')
# TODO: DPL config
parser.add_argument('--config', default="config/", type=str,
                    help='used configure')
parser.add_argument('--layer', default='all', type=str, choices=['all', 'conv', 'fc'],
                    help='Layers to compress: all, conv, fc')
# setting for saving and loading compressed model
parser.add_argument('--path-to-save', default='model_path/',
                    help='Path to save compressed layer weights', type=str)
parser.add_argument('--path-to-load', default=None,
                    help='Path to load compressed layer weights', type=str) # model_path/
# other settings
parser.add_argument('--pretest', default=False, action='store_true',
                    help='Evaluate model before compression')
parser.add_argument('--dbg', default=False, action='store_true',
                    help='Show processing information')

args = parser.parse_args()
logging.config.fileConfig(os.path.join(args.config, 'logger.config'))
logger = logging.getLogger()
if args.dbg:
    logger.setLevel(logging.DEBUG)
else: 
    logger.setLevel(logging.INFO)
logger.info(f"DPL Compression")

if __name__ == '__main__':
    # config for bloks and words
    logger.info(f"block and word config loading.")
    blockconfig = ConfigParser()
    wordconfig = ConfigParser()
    blockconfig.read(os.path.join(args.config, 'DPL_block.config'), encoding='UTF-8')
    wordconfig.read(os.path.join(args.config, 'DPL_word.config'), encoding='UTF-8')
    logger.info(f"Configuration: {args}")
    
    # ---- load dataset ----
    train_loader, test_loader, num_classes = dataloader(args.dataset, args.data_path, args.batch_size, args.n_workers, distributed=False)
    logger.info(f"Dataset loaded.")
    
    # ---- load pretrained model ---- 
    model = model.__dict__[args.model](pretrained=(args.dataset == 'imagenet'), num_classes=num_classes)
    if args.model_path: #load private pretrained model 
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    #device
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    #some variables for compute compression time and model size
    size_uncompressed = compute_size(model)
    size_reconstruct = 0.0
    size_other = size_uncompressed
    time_compress = 0.0
    logger.info(f"Model {args.model} loaded ( size: {size_uncompressed:.3f}MB )")
    
    #evaluating before compression
    if args.pretest:
        top_1_before, top_5_before = loggerWarpper(logger, evaluate, model, test_loader)
        logger.info(f'Top1: {top_1_before:.3f} - Top5: {top_5_before:.3f}')
    
    # extract layers from model
    layer_start = 1
    thres = 1.0
    watcher = ActivationWatcher(model)
    layers = [layer for layer in watcher.layers[layer_start: ]]
    #-------------------------------  DPL Compression   ------------------------------    
    logger.info(f'DPL Compression\n')
    for layer in layers:
        #get weight of layer
        M = attrgetter(layer + '.weight.data')(model).detach()
        size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
        size_other -= size_uncompressed_layer
        logger.debug(f'{layer} - {list(M.size())} - {size_uncompressed_layer:.5f}MB - blocksize: {blockconfig[args.model][layer]} - wordsize: {wordconfig[args.model][layer]}')

        #load compressed model
        if args.path_to_load:
            try:
                layer_weight_path = os.path.join(args.path_to_load, f'{layer}.pth')
                M = torch.load(layer_weight_path)
                attrgetter(layer + '.weight')(model).data = M
                if args.dataset != 'imagenet':
                    top_1, top_5 = loggerWarpper(logger, evaluate, model, test_loader)
                continue
            except FileNotFoundError:
                logger.info('layer weight path is not found: {}'.format(os.path.join(args.path_to_load, '{}.pth'.format(layer))))
        
        #initialization
        error = 0.0
        size_layer = 0.0
        M_dpl = []
        is_conv = len(M.shape) == 4

        #  get weight shape info
        #  Skip if don't compress specific layers
        if is_conv:
            if args.layer == 'fc': 
                size_reconstruct += size_uncompressed_layer
                continue
            else:    
                out_features, in_features, k, _ = M.size()
        else:
            if args.layer == 'conv':
                size_reconstruct += size_uncompressed_layer
                continue
            else:
                out_features, in_features = M.size()
                k = 1
        
        n_blocks = 1 if int(blockconfig[args.model][layer]) == 0 else in_features * k * k // int(blockconfig[args.model][layer])
        n_word = int(wordconfig[args.model][layer])
        #reshape and chunk weight matrix
        M = reshape_weight(M)
        logger.debug(f'\treshaped layer size {list(M.shape)}')
        assert M.size(0) % n_blocks == 0, f"layer {layer} - division error: M[0] ({M.size(0)}) %% n_blocks ({n_blocks})"
        M_blocks = M.chunk(n_blocks, dim=0)
        begin = time.time()
        #__________________________   DPL decomposition   ________________________
        for M_block in M_blocks:
            dpl = DPL.DPL(Data = M_block, DictSize=n_word, tau=0.05)
            dpl.Update(iterations=20, showFlag=False)
            M_dpl.append(torch.matmul(dpl.DictMat, torch.matmul(dpl.P_Mat, dpl.DataMat)))
            block_size = torch.matmul(dpl.P_Mat, dpl.DataMat).numel() * 2/1024/1024 + dpl.DictMat.numel() * 2/1024/1024
            size_layer += block_size
            error += dpl.evaluate()
        #_________________________________________________________________________
        end = time.time()
        time_cost = end - begin
        time_compress += time_cost
        size_reconstruct += size_layer
        # reconstruct
        M = torch.cat(M_dpl, dim=0).float()
        M = reshape_back_weight(M, k=k, conv=is_conv)
        torch.save(M, os.path.join(args.path_to_save, '{}.pth'.format(layer)))
        attrgetter(layer + '.weight')(model).data = M
        if args.pretest:
            top_1, top_5 = loggerWarpper(logger, evaluate, model, test_loader)
            logger.debug('\tTop1 after compression: {:.3f}, Top5 after compression: {:.3f}'.format(top_1, top_5))
            logger.debug('\tAccuracy Loss:{:.3f}'.format(top_1_before - top_1))
        logger.debug(f'\t#words {n_word} - #blocks {n_blocks} - compressed size {size_layer:.4f}MB - ratio {size_uncompressed_layer / size_layer} - reconstruct error {error / M.numel():.6f} - time {time_cost:.2f}s')
        if error / M.numel() > thres:
            logger.warn(F' @@@@@ RESET #WORDS {n_word} - {layer} @@@@@ ') 
        #--------------------------------------------------------------------------
    
    #result print
    logger.info(f'compressed size {size_reconstruct + size_other:.2f}MB - ratio {size_uncompressed / (size_reconstruct + size_other):.2f} - time {time_compress:.2f}s')
    logger.setLevel(logging.INFO)
    top_1, top_5 = loggerWarpper(logger, evaluate, model, test_loader)
    logger.info(f'Top1 after compression {top_1:.3f}% - Top5 after compression {top_5:.3f}%')
