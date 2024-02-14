echo $CUDA_VISIBLE_DEVICES
script_path=$(dirname $(realpath $0))
code_path=$(dirname $(dirname $script_path))
echo $code_path
echo $script_path
# echo $(realpath $0)
# echo $(basename $0)
. ${script_path}/data_config.sh
echo $imagenet_path
echo $exps_path
echo $code_path