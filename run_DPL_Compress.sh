
model=resnet18
python Quantize_DPL.py --dataset imagenet --model $model --layer all #--dbg
