# 获取第一个参数
task=${1}

# 获取第二个参数
gpu_id=${2}

export CUDA_VISIBLE_DEVICES=${gpu_id}

python embedding_getter.py --task ${1}