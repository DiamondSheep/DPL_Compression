import os
import csv
import time
import argparse
import logging
import logging.config
import torch
import torch.backends.cudnn as cudnn

from operator import attrgetter
from configparser import ConfigParser
from pytorchcv.model_provider import get_model

from utils.utils import ActivationWatcher, compute_size, reshape_weight, reshape_back_weight
from utils import dataloader
from utils.eval import evaluate

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
parser.add_argument('--batch-size', default=128, type=int,
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
parser.add_argument('--shared', default=False, action='store_true',
                    help='For shared dictionary')
parser.add_argument('--auto_param', default=False, action='store_true',
                    help='auto parameter tuning')
parser.add_argument('--error_thres_up', default=0.000006, type=float, 
                    help='lower up threshold decrease the construction error')
parser.add_argument('--error_thres_down', default=None, type=float, 
                    help='higher down threshold lead to larger compression ratio')
parser.add_argument('--results', default='results/', type=str,
                    help='path to results')

args = parser.parse_args()
logging.config.fileConfig(os.path.join(args.config, 'logger.config'))
logger = logging.getLogger()
if args.dbg:
    logger.setLevel(logging.DEBUG)
else: 
    logger.setLevel(logging.INFO)
logger.info(f"DPL Compression Starting.")

if __name__ == '__main__':
    # config for blocks and words
    logger.info(f"block and word config loading.")
    blockconfig = ConfigParser()
    wordconfig = ConfigParser()
    block_config_file = os.path.join(args.config, 'DPL_block_cv.config')
    word_config_file = os.path.join(args.config, 'DPL_word_cv.config')
    blockconfig.read(block_config_file, encoding='UTF-8')
    wordconfig.read(word_config_file, encoding='UTF-8')
    if args.error_thres_down == None:
        # auto config the down thres
        args.error_thres_down = args.error_thres_up / 10
    logger.info(f"Configuration: {args}")
    
    # ---- load dataset ----
    train_loader, test_loader, num_classes = dataloader(args.dataset, args.data_path, args.batch_size, args.n_workers, distributed=False)
    logger.info(f"Dataset loaded.")
    
    # ---- load pretrained model ---- 
    args.model = f'{args.model}_{args.dataset}' if 'cifar' in args.dataset else args.model
    model = get_model(args.model, pretrained=True)
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
    logger.info(f'Compressing')
    last_layer = None
    for layer in layers:
        M = None
        M_last = None
        shared = False
        if args.shared and 'conv' in layer and (int(wordconfig[args.model + '_shared'][layer]) == 0 or last_layer is not None):
            shared = True
            #shared dictionary design for [conv1, conv2]
            if int(wordconfig[args.model + '_shared'][layer]) == 0:
                logger.debug(f'shared dictionary activated')
                last_layer = layer
                continue
            else:
                M_last = attrgetter(last_layer + '.weight.data')(model).detach()
                M = attrgetter(layer + '.weight.data')(model).detach()
                size_uncompressed_layer = (M.numel() + M_last.numel()) * 4 / 1024 / 1024 
                size_other -= size_uncompressed_layer
                block_size = int(blockconfig[args.model + '_shared'][layer])
                n_word = int(wordconfig[args.model + '_shared'][layer])
                logger.debug(f'{last_layer}+{layer} - {list(M_last.size())}+{list(M.size())} - {size_uncompressed_layer:.5f}MB - blocksize {block_size} - words {n_word}')

        else:
            #get weight of layer
            M = attrgetter(layer + '.weight.data')(model).detach()
            M_last = None
            size_uncompressed_layer = M.numel() * 4 / 1024 / 1024
            size_other -= size_uncompressed_layer
            block_size = int(blockconfig[args.model][layer])
            n_word = int(wordconfig[args.model][layer])
            logger.debug(f'{layer} - {list(M.size())} - {size_uncompressed_layer:.5f}MB - blocksize {block_size} - words {n_word}')
        #load compressed model (not implemented for shared dictionary)
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
        n_blocks = 1 if block_size == 0 else in_features * k * k // block_size
        
        #reshape and chunk weight matrix
        need_transpose = True 
        if shared:
            M_last, M = reshape_weight(M_last), reshape_weight(M)
            assert M_last.size()[0] == M.size()[0]
            
            last_C_out = M_last.size()[1]
            M = torch.cat([M_last, M], dim=1) # C_out dimension
        else:
            M = reshape_weight(M, need_transpose)
        word_ceiling = min(M.size(1), block_size) if block_size != 0 else M.size(1)
        logger.debug(f'\treshaped layer size {list(M.shape)} - #words ceiling {word_ceiling}')
        assert M.size(0) % n_blocks == 0, f"layer {layer} - division error: M[0] ({M.size(0)}) %% n_blocks ({n_blocks})"
        M_blocks = M.chunk(n_blocks, dim=0)
        begin = time.time()
        if args.auto_param:
            n_weights = M.numel()
            #__________________________   DPL decomposition   ________________________
            def DPL_loop(): 
                error = 0.0
                for M_block in M_blocks:
                    dpl = DPL.DPL(Data = M_block, DictSize=n_word, tau=0.05)
                    dpl.Update(iterations=20, showFlag=False)
                    block_size = torch.matmul(dpl.P_Mat, dpl.DataMat).numel() * 2/1024/1024 + dpl.DictMat.numel() * 2/1024/1024
                    error += dpl.evaluate()
                    return error / n_weights
            recon_error = DPL_loop()
            iter = 0
            # update #words
            if n_word > word_ceiling:
                n_word = word_ceiling
            while recon_error > args.error_thres_up or recon_error < args.error_thres_down:
                if iter >= 50:
                    break
                if n_word > word_ceiling:
                    n_word = word_ceiling
                    logger.debug(f"REACH CEILING! | recon_error={recon_error:.8f} ")
                    break
                if recon_error < args.error_thres_down:
                    n_word -= 1
                    logger.debug(f"recon_error={recon_error:.8f} | tuning: n_word {n_word + 1} -> {n_word}")
                else:
                    n_word += 1                
                    logger.debug(f"recon_error={recon_error:.8f} | tuning: n_word {n_word - 1} -> {n_word}")
                if n_word <= 1:
                    break
                recon_error = DPL_loop()
                iter += 1
            wordconfig[args.model][layer] = str(n_word)
            #_________________________________________________________________________
        error = 0.0
        size_layer = 0.0
        M_dpl = []
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
        if shared:
            M = torch.cat(M_dpl, dim=0).float()
            M_last, M = M[:, :last_C_out], M[:, last_C_out:]
            M_last, M = reshape_back_weight(M_last, k=k, conv=is_conv), reshape_back_weight(M, k=k, conv=is_conv)
            torch.save(M, os.path.join(args.path_to_save, f'{args.dataset}_{args.model}_{layer}.pth'))
            torch.save(M_last, os.path.join(args.path_to_save, f'{args.dataset}_{args.model}_{last_layer}.pth'))
            attrgetter(layer + '.weight')(model).data = M
            attrgetter(last_layer + '.weight')(model).data = M_last
        else:
            M = torch.cat(M_dpl, dim=0).float()
            M = reshape_back_weight(M, k=k, conv=is_conv, transpose=need_transpose)
            torch.save(M, os.path.join(args.path_to_save, f'{args.dataset}_{args.model}_{layer}.pth'))
            attrgetter(layer + '.weight')(model).data = M
        
        if args.pretest and args.dataset != 'imagenet':
            top_1, top_5 = loggerWarpper(logger, evaluate, model, test_loader)
            logger.debug('\tTop1 after compression: {:.3f}, Top5 after compression: {:.3f}'.format(top_1, top_5))
            logger.debug('\tAccuracy Loss:{:.3f}'.format(top_1_before - top_1))
        logger.debug(f'\t#words {n_word} - #blocks {n_blocks} - compressed size {size_layer:.4f}MB - ratio {size_uncompressed_layer / size_layer} - reconstruct error {error / M.numel():.6f} - time {time_cost:.2f}s')
        if error / M.numel() > thres:
            logger.warn(F' @@@@@ RESET #WORDS {n_word} - {layer} @@@@@ ') 
        
        last_layer = None
        #--------------------------------------------------------------------------
    # save configs to file 
    if args.auto_param:
        logger.debug(f"update word settings to {word_config_file}")
        with open(word_config_file, 'w', encoding='utf-8') as word_handler:
            wordconfig.write(word_handler)
    #result print
    logger.info(f'compressed size {size_reconstruct + size_other:.2f}MB - ratio {size_uncompressed / (size_reconstruct + size_other):.2f} - time {time_compress:.2f}s')
    logger.setLevel(logging.INFO)
    top_1, top_5 = loggerWarpper(logger, evaluate, model, test_loader)
    logger.info(f'Top1 after compression {top_1:.3f}% - Top5 after compression {top_5:.3f}%')
    
    csvfile = f'{args.dataset}_{args.model}_shared.csv' if args.shared else f'{args.dataset}_{args.model}.csv'
    logger.info(f'results recorded in {csvfile}')
    with open(os.path.join(args.results, csvfile), 'a') as csvloader:
        writer = csv.writer(csvloader)
        writer.writerow([args.error_thres_up, args.error_thres_down, size_uncompressed / (size_reconstruct + size_other), top_1, top_5])
