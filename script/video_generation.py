import cv2
import os


def video_generation(image_folder, output_video):
    # 获取文件夹中所有图片并按顺序排序
    images = sorted([img[:-4] for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],key=int)
    # 确保图片列表不为空
    if not images:
        raise ValueError("No images found in the specified folder!")

    # 读取第一张图片以获取帧的宽度和高度
    first_image_path = os.path.join(image_folder, images[0]+'.png')
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和帧率
    fps = 30 # 每秒显示的帧数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 视频编码器
    video_dir_path = os.path.dirname(output_video)
    if not os.path.exists(video_dir_path):
        os.makedirs(video_dir_path)
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 遍历所有图片，将它们写入视频
    for image in images:
        img_path = os.path.join(image_folder, image+'.png')
        frame = cv2.imread(img_path)
        video.write(frame) # 将图片写入视频

    # 释放视频对象
    video.release()
    print(f"视频已成功保存为 {output_video}")

if __name__ == '__main__':
    # 设置图片文件夹路径和输出视频路径
    episode_num = 5
    for i in range(episode_num):
        image_folder = f'./data/gpt_beaker_grasp/episode{i}/camera/color/'
        output_video = f'./data/gpt_beaker_grasp/episode{i}/video/' # 替换为输出视频的文件名
        video_generation(os.path.join(image_folder,'front'), os.path.join(output_video,'front.mp4'))
        video_generation(os.path.join(image_folder,'head'), os.path.join(output_video,'head.mp4'))
        video_generation(os.path.join(image_folder,'left'), os.path.join(output_video,'left.mp4'))
        video_generation(os.path.join(image_folder,'right'), os.path.join(output_video,'right.mp4'))  