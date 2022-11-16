dataset=cifar10
model=resnet20
mkdir results
mkdir model_path
python DPL_Compress.py --dataset ${dataset} --data-path data/${dataset} --model $model --layer=all --auto_param # --dbg # for debug