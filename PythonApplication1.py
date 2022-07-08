import math
import numpy as np
import cv2 
import time
def get_tr(img):
    mouse_params = {'x': None, 'width': None, 'height': None,
                    'y': None, 'temp': None}
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse, mouse_params)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    return [mouse_params['x'], mouse_params['y'], mouse_params['width'],
            mouse_params['height']], mouse_params['temp']
def on_mouse(event, x, y, flags, param):
    global img, point1
    img2 = img.copy()
    #左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    #拖拽
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    #结束
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)
        param['x'] = min(point1[0], point2[0])
        param['y'] = min(point1[1], point2[1])
        param['width'] = abs(point1[0] - point2[0])
        param['height'] = abs(point1[1] - point2[1])
        param['temp'] = img[param['y']:param['y']+param['height'],
            param['x']:param['x']+param['width']]
cap1 = cv2.VideoCapture("C:/Users/95856/Desktop/t5.mp4")
videopath=cap1
ret,frame = cap1.read()
img = frame
rect, temp = get_tr(img)
image = frame[rect[0]:rect[0]+rect[2],rect[1]:rect[1]+rect[3]]
# 自己定义的函数 matchSift(image, frame)
def matchSift(findimg, img):
    """转换成灰度图片"""
    gray1 = cv2.cvtColor(findimg, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """创建SIFT对象"""
    sift = cv2.SIFT_create() 
    """创建FLAN匹配器"""
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    """检测关键点并计算键值描述符"""
    kpts1, descs1 = sift.detectAndCompute(gray1, None)
    kpts2, descs2 = sift.detectAndCompute(gray2, None)
    """KnnMatt获得Top2"""
    matches = matcher.knnMatch(descs1, descs2, 2)
    """根据他们的距离排序"""
    matches = sorted(matches, key=lambda x: x[0].distance)
    """比率测试，以获得良好的匹配"""
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
    canvas = img.copy()
    """发现单应矩阵"""
    """当有足够的健壮匹配点对（至少个MIN_MATCH_COUNT）时"""
    MIN_MATCH_COUNT=4
    if len(good) >= MIN_MATCH_COUNT:
        """从匹配中提取出对应点对"""
        """小对象的查询索引，场景的训练索引"""
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        """利用匹配点找到CV2.RANSAC中的单应矩阵"""
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        """计算图1的畸变，也就是在图2中的对应的位置"""
        h, w = findimg.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        """绘制边框"""
        cv2.polylines(canvas, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return img
    return canvas
found = matchSift(image, frame)
def readVideo(videopath, image):
    # 打开视频文件：在实例化的同时进行初始化
    cap = cv2.VideoCapture("C:/Users/95856/Desktop/t5.mp4")
    # 检测是否正常打开：成功打开时，isOpened返回ture
    while(cap.isOpened()):
        # 获取每一帧的图像frame
        ret, frame = cap.read()
        # 这里必须加上判断视频是否读取结束的判断,否则播放到最后一帧的时候出现问题了
        if ret == True:
            start = time.time()
            # 将从视频中的获取的图像与要找的图像进行匹配
            found = matchSift(image, frame)
            end = time.time()
            print(end-start)
            # 获取原图像的大小
            width, height= found.shape[:2]
            # 将原图像缩小到原来的二分之一
            size = (int(height/2), int(width/2))
            found = cv2.resize(found, size, interpolation=cv2.INTER_AREA)
            # 显示图片
            cv2.imshow("found", found)
        else:
            break
        # 因为视频是7.54帧每秒，因此每一帧等待133ms - 62ms
        if cv2.waitKey(133-int((end-start)*1000)) & 0xFF == ord('q'):
            break
    # 停止在最后的一帧图像上
    cv2.waitKey()
    # 关掉所有已打开的GUI窗口
    cv2.destroyAllWindows()
if __name__=="__main__":
    # 要匹配图片的路径
    imagepath = image
    cap1 = cv2.VideoCapture("C:/Users/95856/Desktop/t5.mp4")
    videopath=cap1
    # 读取一张图片
    readVideo(videopath, image)
