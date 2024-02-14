# check which server we are on
if [[ $(hostname) == 'torrnode8' ]]; then
    echo "We are on torrnode8"
    imagenet_path="/storage2/guangrun/qijia_3d_model/imagenet/imagenet_train_crop224_sz128.zip"
    exps_path="/storage2/guangrun/qijia_3d_model/exps/eg3d/"
    code_path='/scratch/local/ssd/guangrun/qijia_3d_model/3d_generation_wild_image/'
elif [[ $(hostname) == 'torrnode9' ]]; then
    echo "We are on torrnode9"
    data_path="/mnt/data/3d_generation_wild_image"
    imagenet_path="/mnt/data/imagenet"
else
    echo "We are on htc"
    data_path=""
    imagenet_path=""
fi