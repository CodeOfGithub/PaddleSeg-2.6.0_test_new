import os
import cv2


def save_img2():  # 提取视频中图片 按照每秒提取   间隔是视频帧率
    video_path = r'D:/NoteBook/yc/vidio/1/'  # 视频所在的路径
    f_save_path = 'data/Save_Img/'  # 保存图片的上级目录
    videos = os.listdir(video_path)  # 返回指定路径下的文件和文件夹列表。
    for video_name in videos:  # 依次读取视频文件
        file_name = video_name.split('.')[0]  # 拆分视频文件名称 ，剔除后缀
        folder_name = f_save_path + file_name  # 保存图片的上级目录+对应每条视频名称 构成新的目录存放每个视频的
        os.makedirs(folder_name, exist_ok=True)  # 创建存放视频的对应目录
        vc = cv2.VideoCapture(video_path + video_name)  # 读入视频文件
        fps = vc.get(cv2.CAP_PROP_FPS)  # 获取帧率
        print(fps)  # 帧率可能不是整数  需要取整
        rval = vc.isOpened()  # 判断视频是否打开  返回True或False
        c = 1
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()  # videoCapture.read() 函数，第一个返回值为是否成功获取视频帧，第二个返回值为返回的视频帧：
            pic_path = folder_name + '/'
            if rval:

                if (c % round(fps) == 0):  # 每隔fps帧进行存储操作   ,可自行指定间隔
                    cv2.imwrite(pic_path + 'video_' + str(round(c / fps)) + '.png',
                                frame)  # 存储为图像的命名 video_数字（第几个文件）.png
                    print('video_' + str(round(c / fps)) + '.png')
                cv2.waitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下键,则接续等待(循环)
                c = c + 1
            else:
                break
        vc.release()
        print('save_success' + folder_name)


save_img2()

#
# import cv2
#
# cap = cv2.VideoCapture("D:/NoteBook/yc/vidio/1/image_data.mp4")
# c = 1
# timeRate = 0.1  # 截取视频帧的时间间隔（这里是每隔10秒截取一帧）
#
# while (True):
#     ret, frame = cap.read()
#     FPS = cap.get(5)
#     if ret:
#         frameRate = int(FPS) * timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
#         if (c % frameRate == 0):
#             print("开始截取视频第：" + str(c) + " 帧")
#             # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
#             cv2.imwrite("./capture_image/" + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
#         c += 1
#         cv2.waitKey(0)
#     else:
#         print("所有帧都已经保存完成")
#         break
# cap.release()


