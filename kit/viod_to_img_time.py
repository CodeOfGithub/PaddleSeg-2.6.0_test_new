import cv2

cap = cv2.VideoCapture('D:\\Code\\python\\ycProjet\\PaddleSeg-2.6.0\\PaddleSeg-2.6.0_test_new\\data\\image_data.mp4')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
# c = 1
# timeRate = 10  # 截取视频帧的时间间隔（这里是每隔10秒截取一帧）
#
# while True:
#     ret, frame = cap.read()
#     FPS = cap.get(5)
#     if ret:
#         frameRate = int(FPS) * timeRate  # 因为cap.get(5)获取的帧数不是整数，所以需要取整一下（向下取整用int，四舍五入用round，向上取整需要用math模块的ceil()方法）
#         if c % frameRate == 0:
#             print("开始截取视频第：" + str(c) + " 帧")
#             # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地
#             cv2.imwrite("D:\\Code\\python\\ycProjet\\PaddleSeg-2.6.0\\PaddleSeg-2.6.0_test_new\\img\\capture_image\\" + str(c) + '.jpg', frame)  # 这里是将截取的图像保存在本地
#         c += 1
#         cv2.waitKey(0)
#     else:
#         print("所有帧都已经保存完成")
#         break
# cap.release()