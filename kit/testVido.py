import cv2


def vido_test():
    test = cv2.VideoCapture("D:/Code/python/ycProjet/PaddleSeg-2.6.0/PaddleSeg-2.6.0_test_new/data/image_data.mp4")
    fps = test.get(cv2.CAP_PROP_FPS)
    print(fps)
    c = 1
    t = 1
    timeRate = 0.1
    val = test.isOpened()
    while val:
        ret, frame = test.read()
        if ret:
            frameRate = fps * timeRate
            if c % frameRate == 0:
                print(frame)
                t = t + 1
            c = c + 1
        else:
            break
    test.release()
    print(t)


vido_test()
