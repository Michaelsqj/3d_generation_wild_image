script_path=$(dirname $(realpath $0))
code_path=$(dirname $(dirname $script_path))
. ${script_path}/data_config.sh
cd ${code_path}/src
export TORCH_EXTENSIONS_DIR="${code_path}/torch_extensions"
export CUDA_VISIBLE_DEVICES=0,4,5,6,7
CMD="python train.py \
--outdir=${exps_path}/imagenet_128/eg3d_depth_imagenet_14_02_24/ \
--cfg=shapenet \
--gamma=0.3 \
--data=${imagenet_path} \
--data_class=training.dataset.ImageFolderDatasetImagenet \
--gpus=5 \
--batch=40 \
--gen_pose_cond=True \
--aug=ada \
--workers=8 \
--glr=0.0025 \
--dlr=0.002 \
--loss_class=training.loss.StyleGAN2DepthLoss \
--depth_loss_weight=0.1 \
&> ${code_path}/logs/eg3d_depth_imagenet_14_02_24.log"
echo $CMD
echo $TORCH_EXTENSIONS_DIR
echo $CUDA_VISIBLE_DEVICES
module list
eval $CMD