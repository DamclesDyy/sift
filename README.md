#   说明文档
##    项目要求
    完成经典算法，基于视频处理，鼠标交互(或其他交互方式)不适用深度学习
##    代码实现功能
    通过sift和鼠标交互实现视频第一帧圈定物体的跟踪
##    代码依赖环境
Python'=='3.7（64-bit)<br>
numpy'=='1.19.4<br>
cv2'=='4.6.0.66<br>
time库<br>
math库<br>
##    算法原理
    一、SIFT算法简介
        SIFT，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于图像处理领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。该方法于1999年由David Lowe 首先发表于计算机视觉国际会议（International Conference on Computer Vision，ICCV），2004年再次经David Lowe整理完善后发表于International journal of computer vision（IJCV）。截止2014年8月，该论文单篇被引次数达25000余次。

1、SIFT算法的特点
（1） SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性；

（2）独特性（Distinctiveness）好，信息量丰富，适用于在海量特征数据库中进行快速、准确的匹配；

（3）多量性，即使少数的几个物体也可以产生大量的SIFT特征向量；

（4）高速性，经优化的SIFT匹配算法甚至可以达到实时的要求；

（5）可扩展性，可以很方便的与其他形式的特征向量进行联合。

2、SIFT算法可以解决的问题
        目标的自身状态、场景所处的环境和成像器材的成像特性等因素影响图像配准/目标识别跟踪的性能。而SIFT算法在一定程度上可解决：

（1）目标的旋转、缩放、平移（RST）        

（2）图像仿射/投影变换（视点viewpoint）

（3）光照影响（illumination）

（4）目标遮挡（occlusion）

（5）杂物场景（clutter）

（6）噪声

        SIFT算法的实质是在不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向。SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如角点、边缘点、暗区的亮点及亮区的暗点等。 

二、SIFT算法分为如下四步
1. 尺度空间极值检测
        搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。

2. 关键点定位
        在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。

3. 方向确定
        基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。

4. 关键点描述
        在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。
##    python代码
###    鼠标交互画框
```python
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
cap1 = cv2.VideoCapture("C:/Users/95856/Desktop/t3.mp4")
videopath=cap1
ret,frame = cap1.read()
img = frame
rect, temp = get_tr(img)
image = frame[rect[0]:rect[0]+rect[2],rect[1]:rect[1]+rect[3]]
```
###    sift特征匹配
```python
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
    """单位矩阵""
    MIN_MATCH_COUNT=4
    if len(good) >= MIN_MATCH_COUNT:
        """生成目标"""
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
```
###    处理视频取每一帧并且输出找到的画框图案
```Python
def readVideo(videopath, image):
    # 打开视频文件：在实例化的同时进行初始化
    cap = cv2.VideoCapture("C:/Users/95856/Desktop/t3.mp4")
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
```
###    主函数
```python
if __name__=="__main__":
    # 要匹配图片的路径
    imagepath = image
    cap1 = cv2.VideoCapture("C:/Users/95856/Desktop/t3.mp4")
    videopath=cap1
    # 读取一张图片
    readVideo(videopath, image)
```
