import cv2
import os

def video_generation(image_folder, output_video):

    # 获取文件夹内的所有图片文件名，假设图片按0.png, 1.png, 2.png等命名
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort(key=lambda f: int(f.split('.')[0]))
    # 获取第一张图片的尺寸，假设所有图片大小一致
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, layers = first_image.shape

    # 定义视频的输出路径，设置视频帧率和分辨率
    fps = 30  # 可以根据需要调整帧率

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 遍历所有图片，将它们写入视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        video_writer.write(img)

    # 释放视频写入对象
    video_writer.release()

    print(f"视频已经保存为 {output_video}")


if __name__ == '__main__':
    # 设置图片所在目录和输出视频的路径
    image_folder = '/home/robot4chem/robot/RoboTwin/data/tube_grasp_WBCD_D435/episode0/camera/color/'  # 替换为你的图片文件夹路径
    output_video = '/home/robot4chem/robot/RoboTwin/video/tube_grasp_WBCD/'  # 输出视频文件名
    positions = ['front','head','left','right','observer']
    for position in positions:
        video_generation(os.path.join(image_folder,position), os.path.join(output_video,position+'.mp4'))