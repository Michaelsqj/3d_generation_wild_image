# CUDA_VISIBLE_DEVICES=0 python generate_sd_imagenet_cls.py --idx=0,199 &> log_gen_1 &
CUDA_VISIBLE_DEVICES=4 python generate_sd_imagenet_cls.py --idx=200,399 &> log_gen_2 &
CUDA_VISIBLE_DEVICES=5 python generate_sd_imagenet_cls.py --idx=400,599 &> log_gen_3 &
CUDA_VISIBLE_DEVICES=6 python generate_sd_imagenet_cls.py --idx=600,799 &> log_gen_4 &
CUDA_VISIBLE_DEVICES=7 python generate_sd_imagenet_cls.py --idx=800,999 &> log_gen_5 &