
model=resnet18
python DPL_Compress.py --dataset imagenet --model $model --layer all #--dbg # for debug
