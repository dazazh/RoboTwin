import cv2
import os
import glob

def images_to_video(path, output_name='output.mp4', freq=30):
    """
    将文件夹中的图片序列转换为视频
    :param path: 图片文件夹路径
    :param output_name: 输出视频文件名
    :param freq: 视频帧率
    """
    # 获取所有图片文件并按数字顺序排序
    image_files = glob.glob(os.path.join(path, '*.[pj][np][g]'))
    image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))  # 按文件名中的数字排序
    
    if not image_files:
        print("未找到图片文件")
        return
    
    # 读取第一张图片获取尺寸信息
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, freq, (width, height))

    # 逐帧写入图片
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)
        
    # 释放资源
    out.release()
    print(f"Video has saved in {output_name}")

# 使用示例
if __name__ == "__main__":
    images_to_video("/data3/yuxuan/robotics/RoboTwin-main/data/dual_bottles_pick_easy_L515/episode0/camera/color/head", "output.mp4", 15)
