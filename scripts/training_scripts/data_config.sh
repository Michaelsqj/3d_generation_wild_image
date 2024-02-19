# check which server we are on
if [[ $(hostname) == 'torrnode8' ]]; then
    echo "We are on torrnode8"
    imagenet_path="/storage2/guangrun/qijia_3d_model/imagenet/imagenet_train_crop224_sz128.zip"
    sd_generated_obj_path="/storage2/guangrun/qijia_3d_model/imagenet/sd_generated_2_selected/"
    exps_path="/storage2/guangrun/qijia_3d_model/exps/eg3d/"
    export TRANSFORMERS_CACHE="/storage2/guangrun/qijia_3d_model/huggingface/"
    # code_path='/scratch/local/ssd/guangrun/qijia_3d_model/3d_generation_wild_image/'
elif [[ $(hostname) == 'torrnode14' ]]; then
    echo "We are on torrnode14"
    imagenet_path="/storage2/guangrun/qijia_3d_model/imagenet/imagenet_train_crop224_sz128.zip"
    exps_path="/storage2/guangrun/qijia_3d_model/exps/eg3d/"
    sd_generated_obj_path="/storage2/guangrun/qijia_3d_model/imagenet/sd_generated_2_selected/"
    export TRANSFORMERS_CACHE="/storage2/guangrun/qijia_3d_model/huggingface/"
    # code_path='/scratch/local/ssd/guangrun/qijia_3d_model/3d_generation_wild_image/'
elif [[ $(hostname) == 'torrnode12' ]]; then
    echo "We are on torrnode12"
    imagenet_path="/storage2/guangrun/qijia_3d_model/imagenet/imagenet_train_crop224_sz128.zip"
    sd_generated_obj_path="/storage2/guangrun/qijia_3d_model/imagenet/sd_generated_2_selected/"
    exps_path="/storage2/guangrun/qijia_3d_model/exps/eg3d/"
    export TRANSFORMERS_CACHE="/storage2/guangrun/qijia_3d_model/huggingface/"
    # code_path='/scratch/local/ssd/guangrun/qijia_3d_model/3d_generation_wild_image/'
else
    echo "We are on htc"
    export TRANSFORMERS_CACHE="/data/engs-tvg/engs2305/qijia_3d_model/huggingface/"
    exps_path="/data/engs-tvg/engs2305/qijia_3d_model/storage/exps/"
    imagenet_path="/data/engs-tvg/engs2305/qijia_3d_model/data/imagenet_train_crop224_sz128.zip"
    sd_generated_obj_path="/data/engs-tvg/engs2305/qijia_3d_model/data/sd_generated_2.zip"
fi