# check which server we are on
if [[ $(hostname) == 'torrnode8' ]]; then
    echo "We are on torrnode8"
    imagenet_path="/storage2/guangrun/qijia_3d_model/imagenet/imagenet_train_crop224_sz128.zip"
    exps_path="/storage2/guangrun/qijia_3d_model/exps/eg3d/"
    export TRANSFORMERS_CACHE="/storage2/guangrun/qijia_3d_model/huggingface/"
    code_path='/scratch/local/ssd/guangrun/qijia_3d_model/3d_generation_wild_image/'
elif [[ $(hostname) == 'torrnode14' ]]; then
    echo "We are on torrnode14"
    imagenet_path="/storage2/guangrun/qijia_3d_model/imagenet/imagenet_train_crop224_sz128.zip"
    exps_path="/storage2/guangrun/qijia_3d_model/exps/eg3d/"
    export TRANSFORMERS_CACHE="/storage2/guangrun/qijia_3d_model/huggingface/"
    code_path='/scratch/local/ssd/guangrun/qijia_3d_model/3d_generation_wild_image/'
else
    echo "We are on htc"
    data_path=""
    imagenet_path=""
fi