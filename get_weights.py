import shutil
from huggingface_hub import hf_hub_download
import os

os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/user/xcs/yuhao/3d-aware/RoboTwin/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 下载模型（存储在默认缓存目录）
downloaded_file = hf_hub_download(
    repo_id="facebook/VGGT-1B",
    filename="model.pt"
)