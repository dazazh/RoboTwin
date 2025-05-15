# 获取第一个参数
task=${1}
should_visualize=${2}
# 获取第二个参数
gpu_id=${3}

export CUDA_VISIBLE_DEVICES=${gpu_id}

python embedding_getter.py --task ${1} --should_visualize ${should_visualize}