import argparse
import model
from configparser import ConfigParser
from operator import attrgetter
from utils.watcher import ActivationWatcher

parser = argparse.ArgumentParser()

# setup for model
parser.add_argument('--model', default='resnet18', help='model', type=str)
parser.add_argument('--n_word', default=None, type=int)
parser.add_argument('--block_size', default=None, type=int)

args = parser.parse_args() 

block_config_file = 'config/DPL_block.config'
word_config_file = 'config/DPL_word.config'

word_num = args.n_word
block_size = args.block_size

blockconfig = ConfigParser()
wordconfig = ConfigParser()
blockconfig.read(block_config_file, encoding='UTF-8')
wordconfig.read(word_config_file, encoding='UTF-8')

model = model.__dict__[args.model](pretrained=True, num_classes=1000)
watcher = ActivationWatcher(model)
layers = [layer for layer in watcher.layers[:]]
for layer in layers:
    # word
    if (layer in wordconfig[args.model]):
        current_words = int(wordconfig[args.model][layer])
        wordconfig[args.model][layer] = str(word_num)#(int(current_words * 1.05))
    else:
        wordconfig.set(args.model, layer, str(word_num))
    
    with open(word_config_file, 'w', encoding='utf-8') as wordfile:
        wordconfig.write(wordfile)
    
    # block
    if (layer in blockconfig[args.model]):
        if block_size == None:
            continue
        else:
            blockconfig[args.model][layer] = str(block_size)
    
    else:
        blockconfig.set(args.model, layer, str(block_size))
    
    with open(block_config_file, 'w', encoding='utf-8') as blockfile:
            blockconfig.write(blockfile)
