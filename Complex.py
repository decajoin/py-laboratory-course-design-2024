import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# img0 = np.zeros((len(dilate_img), len(dilate_img[0])), np.uint8)
# img0.fill(200)
# cv2.drawContours(img0, cnts, -1, (0, 0, 255), 2)  # 绘图
# cv2.imwrite('cnts.jpg', img0)

# 图片中的答题卡框数量. 比如一张图片可以划分为左右2个答题卡框
ANSWER_CARD_SIZE = 3

# 答题卡框的最小宽度
ANSWER_CARD_MIN_WIDTH = 20

# 大标题序号
TITLE_NUM = ['一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、', '十一、', '十二、', '十三、', '十四、', '十五、', '十六、',
             '十七、']
# 识别左上角大标题序号, 识别范围的宽
TITLE_TOP_LEFT_CORNER_WIDTH = 50

# 识别左上角大标题序号, 识别范围的高
TITLE_TOP_LEFT_CORNER_HEIGTH = 65


def order_points(pts):
    """4边形4点排序函数

    Args:
        pts ([type]): 4边形任意顺序的4个顶点

    Returns:
        [type]: 按照一定顺序的4个顶点
    """

    rect = np.zeros((4, 2), dtype="float32")  # 按照左上、右上、右下、左下顺序初始化坐标

    s = pts.sum(axis=1)  # 计算点xy的和
    rect[0] = pts[np.argmin(s)]  # 左上角的点的和最小
    rect[2] = pts[np.argmax(s)]  # 右下角的点的和最大

    diff = np.diff(pts, axis=1)  # 计算点xy之间的差
    rect[1] = pts[np.argmin(diff)]  # 右上角的差最小
    rect[3] = pts[np.argmax(diff)]  # 左下角的差最小
    return rect  # 返回4个顶点的顺序


def four_point_transform(image, pts):
    """4点变换

    Args:
        image ([type]): 原始图像
        pts ([type]): 4个顶点

    Returns:
        [type]: 变换后的图像
    """

    rect = order_points(pts)  # 获得一致的顺序的点并分别解包他们
    (tl, tr, br, bl) = rect

    # 计算新图像的宽度(x)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # 右下和左下之间距离
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # 右上和左上之间距离
    maxWidth = max(int(widthA), int(widthB))  # 取大者

    # 计算新图像的高度(y)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # 右上和右下之间距离
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # 左上和左下之间距离
    maxHeight = max(int(heightA), int(heightB))

    # 有了新图像的尺寸, 构造透视变换后的顶点集合
    dst = np.array(
        [
            [0, 0],  # -------------------------左上
            [maxWidth - 1, 0],  # --------------右上
            [maxWidth - 1, maxHeight - 1],  # --右下
            [0, maxHeight - 1]
        ],  # ------------左下
        dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)  # 计算透视变换矩阵
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 执行透视变换

    return warped  # 返回透视变换后的图像


def sort_contours(cnts, method="left-to-right"):
    """轮廓排序

    Args:
        cnts ([type]): 轮廓
        method (str, optional): 排序方式. Defaults to "left-to-right".

    Returns:
        [type]: 排序好的轮廓
    """

    if cnts is None or len(cnts) == 0:
        return [], []

    # 初始化逆序标志和排序索引
    reverse = False
    i = 0

    # 是否需逆序处理
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # 是否需要按照y坐标函数
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 构造包围框列表，并从上到下对它们进行排序
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(
        zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # 返回已排序的轮廓线和边框列表
    return cnts, boundingBoxes


def get_init_process_img(img_path):
    """
    对图片进行初始化处理，包括灰度，高斯模糊，腐蚀，膨胀和边缘检测等
    :param roi_img: ndarray
    :return: ndarray
    """
    image = cv2.imread(img_path)
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edged = cv2.Canny(blurred, 75, 200)
    # edged = auto_canny(blurred)
    return edged


def capture_img(image, target_image_path, contour):
    """根据轮廓截取图片

    Args:
        origin_image_path ([type]): 原始图片路径
        target_image_path ([type]): 目标图片路径
        contour ([type]): 截取轮廓

    Returns:
        [type]: [description]
    """
    # 根据轮廓或者坐标
    x, y, w, h = cv2.boundingRect(contour)
    # cv2.imwrite(target_image_path, image[y:y + h, x:x + w])
    cv2.imwrite(target_image_path, image[0:y, :])
    cut_image = image[y + h:, :]
    return cut_image


def save_img_by_cnts(save_image_path, image_size, cnts):
    """通过提取的轮廓绘制图片并保存

    Args:
        save_image_path ([type]): 图片存储路径
        image ([type]): 绘制的图片尺寸, 长与宽
        cnts ([type]): 轮廓列表
    """
    black_background = np.ones(image_size, np.uint8) * 0
    cv2.drawContours(black_background, cnts, -1, (255, 255, 255), 2)
    plt.figure(figsize=(10, 5))
    plt.imshow(black_background)
    plt.axis('off')
    plt.savefig(save_image_path)


def ocr_single_line_img(image_path, ocr):
    """ocr识别图片

    Args:
        origin_image_path ([type]): 原始图片路径
        ocr ([type]): ocr

    Returns:
        [type]: [description]
    """

    image = cv2.imread(image_path)
    res = ocr.ocr_for_single_line(image[0:TITLE_TOP_LEFT_CORNER_WIDTH, 0:TITLE_TOP_LEFT_CORNER_HEIGTH])
    if len(res) > 0:
        res[0] = ''
    return res


def get_exam_num_area(image_path):
    """ 获取图片中待检测的考号填充区域

    Args:
        image_path (String): 图片地址

    Returns:
        [type]: [description]
    """
    image = Image.open(image_path)
    image_width = image.width

    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 先腐蚀与膨胀, 高亮化学生填充的考号
    kernel = np.ones((7, 7), np.uint8)
    erode_img = cv2.erode(threshold_img, kernel, iterations=1)
    kernel = np.ones((7, 7), np.uint8)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=1)

    # 学生填充的考号，最左边边缘的x轴坐标
    exam_number_left_x = float("inf")
    # 学生填充的考号，最右边边缘的x轴坐标
    exam_number_right_x = 0
    cnts, _ = cv2.findContours(dilate_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x_array = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if x > image_width / 2:
            x_array.append(x)
            if x + w > exam_number_right_x:
                exam_number_right_x = x + w
    x_array = sorted(x_array, reverse=False)
    # print(x_array)
    exam_number_left_x = x_array[1]

    # 通过x轴坐标，缩小待检测区域的范围
    threshold_img = threshold_img[:, (exam_number_left_x - 20):(exam_number_right_x + 10)]

    # 再通过检测图片中面积最大的轮廓（考号手写区域, 而不是填充区域）, 进一步缩小范围
    cnts, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    num_card_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        (x, y, w, h) = cv2.boundingRect(approx)
        # print((x, y, w, h), approx)
        if len(approx) == 4:
            threshold_img = threshold_img[y:y + h, x:x + w]
            num_card_cnt = c
            break

    cv2.imwrite('answerCard/output/num_card.jpg', threshold_img)
    return threshold_img, num_card_cnt


def get_exam_num_height(img):
    """ 获取考号填充区域, 行中心与行中心的y轴坐标间隔

    Args:
        img ([type]): 图片

    Returns:
        [float]: 行中心与行中心的y轴坐标间隔
    """

    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    dilate_img = cv2.dilate(img, kernel, iterations=1)

    cnts, _ = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    x_array = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > len(img) / 2:
            x_array.append(c)
    cnts, _ = sort_contours(x_array, 'top-to-bottom')
    (x, y, w, h) = cv2.boundingRect(cnts[-1])

    return y, h / 10


def detection_exam_num(image_path):
    """ 识别图片中学生填充的考号

    Args:
        image_path (String): 图片地址

    Returns:
        [list]: 识别的考号结果
    """
    # 获取图片中考号填充区域范围
    thresh_img, _ = get_exam_num_area(image_path)

    # 获取考号填充区域, 每2行的中心y轴坐标间隔
    y_locate, line_y_height = get_exam_num_height(thresh_img)

    # 腐蚀与膨胀
    kernel = np.ones((7, 7), np.uint8)
    erode_img = cv2.erode(thresh_img, kernel, iterations=1)
    kernel = np.ones((7, 7), np.uint8)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=1)
    # 学生填充考号的识别结果

    num_card = []
    cnts, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts, boundingBoxes = sort_contours(cnts, 'left-to-right')

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        num_card.append(int((y - y_locate) / line_y_height))
    return num_card


#
# def detection_choice_question(images_path):
#     """ 选择题自动识别与批改
#     Args:
#         images_path (list): 图片地址列表
#     Returns:
#         [list]: 每张图片的识别结果
#     """
#
#     sub_answer_cnt_szie = 0
#     question_answers = []
#     for img_path in images_path:
#         image = cv2.imread(img_path)
#         # if not is_choice_question(image):
#         #     continue
#         if img_path == 'out/sub_answer_card_0.jpg':
#             continue
#
#         # 获取图片中填充的全部答案轮廓
#         answer_option_cnts = get_answer_option_cnts(image)
#         if len(answer_option_cnts) > 0:
#             save_img_by_cnts('out/answer_cnt_' + str(sub_answer_cnt_szie) + '.png', image.shape[:2], answer_option_cnts)
#
#         # 所有被填充的选择项的中心的x坐标
#         answer_options_center = get_cnt_center(answer_option_cnts)
#         print(answer_options_center)
#
#         # 获取所有选择项的轮廓及其题序轮廓
#         all_choice_option_cnts, question_number_cnts = get_choice_option_cnts(image)
#         if len(all_choice_option_cnts) > 0:
#             save_img_by_cnts('out/choice_cnt_' + str(sub_answer_cnt_szie) + '.png', image.shape[:2],
#                              all_choice_option_cnts)
#             save_img_by_cnts('out/ques_num_' + str(sub_answer_cnt_szie) + '.png', image.shape[:2], question_number_cnts)
#
#         sub_answer_cnt_szie = sub_answer_cnt_szie + 1
#
#         # 选择题自动批改
#         if len(all_choice_option_cnts) > 0:
#             question_answer_dict = get_choice_question_answer_index(image, all_choice_option_cnts, answer_option_cnts,
#                                                                     question_number_cnts)
#             question_answers.append(question_answer_dict)
#     return question_answers
#
#
# def get_answer_option_cnts(img):
#     """ 识别图片中的填充的全部答案轮廓
#
#     Args:
#         img_path (String): 图片
#
#     Returns:
#         [list]: 候选项轮廓
#     """
#     # 转灰度
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # OTSU二值化（黑底白字）
#     thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#
#     # 腐蚀
#     kernel = np.ones((3, 3), np.uint8)
#     erode_img = cv2.erode(thresh_img, kernel, iterations=2)
#     # 膨胀
#     kernel = np.ones((9, 9), np.uint8)
#     dilate_img = cv2.dilate(erode_img, kernel, iterations=1)
#
#     # 提取答案的轮廓
#     answer_cnts, _ = cv2.findContours(dilate_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     # img0 = np.zeros((len(dilate_img), len(dilate_img[0])), np.uint8)
#     # img0.fill(200)
#     # cv2.drawContours(img0, answer_cnts, -1, (0, 0, 255), 2)  # 绘图
#     # cv2.imwrite('cnts.jpg', img0)
#
#     # 减少答案轮廓的边数
#     answer_option_cnts = []
#     for cnt in answer_cnts:
#         peri = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
#         (x, y, w, h) = cv2.boundingRect(approx)
#         if not x > 900:
#             answer_option_cnts.append(approx)
#
#     # self.assertTrue(choiceAnswerCnts % 4 == 0, "候选框提取异常, 提取的数量不是4的整数")
#     return answer_option_cnts
#
#
# def get_choice_option_cnts(img):
#     """识别图片中的所有的选择项轮廓与题序轮廓
#
#     Args:
#         img ([type]): [description]
#         all_option_center_x ([type]): [description]
#
#     Returns:
#         [type]: [description]
#     """
#     # 灰度
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 二值化（黑底白字）
#     thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#
#     # 对高亮部分膨胀
#     # 因为候选区域由三部分组成（左括号、右括号、大写的英文字母），通过膨胀将三个区域连成一片
#     kernel = np.ones((7, 7), np.uint8)
#     dilate_img = cv2.dilate(thresh_img, kernel, iterations=1)
#     # cv2.imwrite('cnts.jpg', dilate_img)
#
#     # 提取膨胀后的轮廓
#     option_cnts, _ = cv2.findContours(dilate_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     img0 = np.zeros((len(dilate_img), len(dilate_img[0])), np.uint8)
#     img0.fill(200)
#     cv2.drawContours(img0, option_cnts, -1, (0, 0, 255), 2)  # 绘图
#     cv2.imwrite('cnts.jpg', img0)
#
#     # 所有候选框的轮廓
#     choice_option_cnts = []
#     # 每道选择题的题序
#     question_number_cnts = []
#     for c in option_cnts:
#         peri = cv2.arcLength(c, True)
#         area = cv2.contourArea(c)
#         approx = cv2.approxPolyDP(c, 0.1 * peri, True)
#         (x, y, w, h) = cv2.boundingRect(approx)
#         ar = h / float(w)
#         # (x, y, w, h) = cv2.boundingRect(c)
#
#         # 筛选轮廓为四边形的目前轮廓
#         #    if y >= 60 and w >= 20 and w <= 60 and ar >= 1 and ar <= 2 and area > 700:
#         if y < 1000 and ar > 0.5 and ar < 2:
#             if w > 20 and h > 10:
#                 choice_option_cnts.append(c)
#             elif w > 15 and area < 300:
#                 question_number_cnts.append(c)
#         # if w > 5 and h < 20:
#         #     choice_option_cnts.append(c)
#         # elif w > 10 and h < 5:
#         #     question_number_cnts.append(c)
#         """ 分块 """
#     # 7行
#     print(img.shape[1]/7)
#     # 4列
#     print((img.shape[0]-60)/4)
#     return choice_option_cnts, question_number_cnts
#
#
# def get_cnt_center(cnts):
#     """返回轮廓中心的x轴坐标
#
#     Args:
#         cnts (list): 轮廓列表
#
#     Returns:
#         [list]: 中心x轴坐标
#     """
#     center = []
#     for cnt in cnts:
#         (x, y, w, h) = cv2.boundingRect(cnt)
#         if (2 * x + 2) / 2 < 1000:
#             center.append((((2 * x + w) / 2), ((2 * y + h) / 2)))
#     return center
#
#
# def get_choice_option_center_x(img):
#     """ 识别所有未被填充的选择项的中心的x坐标
#
#     Args:
#         img ([type]): [description]
#
#     Returns:
#         [type]: [description]
#     """
#     img = cv2.imread(img)
#     ocr_reslut = pytesseract.image_to_data(img, output_type=Output.DICT, lang='chi_sim')
#
#     choice_option_center_x = []
#     for i in range(len(ocr_reslut['text'])):
#         text_i = ocr_reslut['text'][i]
#         (x, y, w, _) = (ocr_reslut['left'][i], ocr_reslut['top'][i], ocr_reslut['width'][i], ocr_reslut['height'][i])
#         if y > 60 and ('A' in text_i or 'B' in text_i or 'C' in text_i or 'D' in text_i):
#             choice_option_center_x.append((2 * x + w) / 2)
#     return choice_option_center_x
#

def get_sub_answer_card_cnts(img_path):
    """ 获得答题卡的子区域

    Args:
        img ([type]): 图片
    Returns:
        [type]: 答题卡的左右答题区域轮廓
    """
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # warped_answer_image_1 = four_point_transform(gray, answer_contour_1.reshape(4, 2))

    # 二值化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # 在二值图像中查找轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立
    thresh_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    cnt_size = 0
    sub_answer_cnts = []
    if len(thresh_cnts) > 0:
        # 将轮廓按大小, 降序排序
        thresh_cnts = sorted(thresh_cnts, key=cv2.contourArea, reverse=True)
        for c in thresh_cnts:
            cnt_size = cnt_size + 1

            # arcLength 计算周长
            peri = cv2.arcLength(c, True)

            # 计算轮廓的边界框
            (x, y, w, h) = cv2.boundingRect(c)

            # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if w > 700 and h < 10:
                sub_answer_cnts.append(approx)

    # 从上到下，将轮廓排序
    sub_answer_cnts = sort_contours(sub_answer_cnts, method="top-to-bottom")[0]
    return sub_answer_cnts


#
# def get_question_num_dict(image, question_number_cnts, ocr):
#     """获取图片中所有的选择题的题序
#
#     Args:
#         image ([type]): 图片
#         question_number_cnts ([type]): 图片中的所有的选择题的题序轮廓
#         ocr ([type]): ocr识别工具
#
#     Returns:
#         [dict]: key: 题序, value: 题序轮廓的坐标
#     """
#     question_num_dict = {}
#     for question_number_cnt in question_number_cnts:
#         peri = cv2.arcLength(question_number_cnt, True)
#         approx = cv2.approxPolyDP(question_number_cnt, 0.1 * peri, True)
#         (x, y, w, h) = cv2.boundingRect(approx)
#
#         # ocr识别题型轮廓区域的文本
#         res = ocr.ocr_for_single_line(image[y:y + h, x:x + w])
#         question_num = res['text']
#         question_num = question_num.replace('.', '')
#
#         # 文本是否为数字
#         if question_num.isdigit():
#             (x, y, w, h) = cv2.boundingRect(question_number_cnt)
#             question_num_dict[int(question_num)] = (x, y, w, h)
#
#     # 按照题序从小到大排序
#     question_num_list = sorted(question_num_dict.items(), key=lambda item: item[0])
#     return dict(question_num_list)
#
#
# def get_choice_question_answer_index(image, choice_option_cnts, answer_option_cnts, question_number_cnts):
#     """自动批改, 返回每道试题对应的答案索引. \
#        注意：(1)用户可能没有填充答案 (2)选择题的答案数量可能大于1
#
#     Args:
#         choice_option_cnts (list): 试题的选择项轮廓
#         answer_option_cnts (list): 用户填充的答案轮廓
#         question_number_cnts (list): 试题的题序轮廓
#     Returns:
#         [dict]: key  题序, value 答案索引列表
#     """
#
#     # 获取所有选择题的题序
#     question_num_dict = get_question_num_dict(image, question_number_cnts)
#
#     question_answer_dict = {}
#     for key in question_num_dict.keys():
#         (num_x, num_y, num_w, num_h) = question_num_dict[key]
#         num_center_x = (2 * num_x + num_w) / 2
#         num_center_y = (2 * num_y + num_h) / 2
#
#         # 获取同一行中，本题序右侧第一个题序的中心x坐标
#         min_num_center_x = float("inf")  # 无穷大
#         for question_number_cnt in question_number_cnts:
#             (x, y, w, h) = cv2.boundingRect(question_number_cnt)
#             right_num_center_x = (2 * x + w) / 2
#             if num_center_y > y and num_center_y < y + h and right_num_center_x > num_center_x and right_num_center_x < min_num_center_x:
#                 min_num_center_x = right_num_center_x
#         # print(min_num_center_x)
#
#         # 获取本题的全部答案轮廓的中心x坐标列表
#         # 一道选择题题可能有多个答案， 所以answers_center_x为列表
#         answers_center_x = []
#         for answer_option_cnt in answer_option_cnts:
#             (x, y, w, h) = cv2.boundingRect(answer_option_cnt)
#             answer_cnt_center_x = (2 * x + w) / 2
#             if num_center_y > y and num_center_y < y + h and answer_cnt_center_x > num_center_x and answer_cnt_center_x < min_num_center_x:
#                 answers_center_x.append(answer_cnt_center_x)
#         # print('answers_center_x', answers_center_x)
#
#         # 获取本题的全部选择项轮廓
#         question_choice_option_cnts = []
#         for choice_option_cnt in choice_option_cnts:
#             # print(len(question_choice_option_cnts))
#             (x, y, w, h) = cv2.boundingRect(choice_option_cnt)
#             choice_option_center_x = (2 * x + w) / 2
#             if num_center_y > y and num_center_y < y + h and choice_option_center_x > num_center_x and choice_option_center_x < min_num_center_x:
#                 question_choice_option_cnts.append(choice_option_cnt)
#
#         question_choice_option_cnts, _ = sort_contours(question_choice_option_cnts, 'left-to-right')
#         # print('question_choice_option_cnts', len(question_choice_option_cnts))
#
#         # 答案列表
#         answer_indexes = []
#         # 答案索引
#         answer_index = 0
#         for choice_option_cnt in question_choice_option_cnts:
#             answer_index = answer_index + 1
#             (x, y, w, h) = cv2.boundingRect(choice_option_cnt)
#             # print((x, y, w, h), answers_center_x)
#             for answer_center_x in answers_center_x:
#                 if answer_center_x > x and answer_center_x < x + w:
#                     answer_indexes.append(answer_index)
#                     break
#         question_answer_dict[key] = answer_indexes
#
#     # 返回每道试题对应的答案索引
#     question_answer_dict = sorted(question_answer_dict.items(), key=lambda item: item[0])
#     return dict(question_answer_dict)
#
#
# def is_choice_question(img):
#     """判断当前图片是否属于选择题
#
#     Args:
#         image_path ([type]): 图片
#
#     Returns:
#         [boolean]: false 不是  true 是
#     """
#     ocr_result = pytesseract.image_to_data(img, output_type=Output.DICT, lang='chi_sim')
#     ocr_text = ocr_result['text']
#     return '[A]' in ocr_text or '[B]' in ocr_text or '[C]' in ocr_text or '[D]' in ocr_text


if __name__ == '__main__':

    # 将答题卡区域切分
    answer_card_images_path = []
    answer_card_images_path.append('answerCard\\answer01.jpg')
    sub_answer_card_images_path = []
    sub_answer_cnt_szie = 0
    for answer_card_image in answer_card_images_path:
        sub_answer_cnts = get_sub_answer_card_cnts(answer_card_image)
        image = cv2.imread(answer_card_image)
        if len(sub_answer_cnts) > 1:
            sub_answer_cnts = sub_answer_cnts[1:len(sub_answer_cnts)]
        if len(sub_answer_cnts) > 0:
            for c in sub_answer_cnts:
                sub_answer_card_image_path = 'answerCard/output/sub_answer_card_' + str(sub_answer_cnt_szie) + '.jpg'
                sub_answer_card_images_path.append(sub_answer_card_image_path)
                image = capture_img(image, sub_answer_card_image_path, c)
                sub_answer_cnt_szie = sub_answer_cnt_szie + 1
        # 切图结束后将最后一张图加入
        sub_answer_card_image_path = 'answerCard/output/sub_answer_card_' + str(sub_answer_cnt_szie) + '.jpg'
        sub_answer_card_images_path.append(sub_answer_card_image_path)
        cv2.imwrite(sub_answer_card_image_path, image)
        sub_answer_cnt_szie = sub_answer_cnt_szie + 1
    print('试题切分结果：', sub_answer_card_images_path)

    # 学生考号自动识别
    num_card = detection_exam_num('answerCard/output/sub_answer_card_0.jpg')
    print('学生考号: ', num_card)

    # # 选择题自动识别与批改
    # question_answer_dict = detection_choice_question(sub_answer_card_images_path)
    # print('每道选择题答案(key 题序, value: 对应题序的答案列表):')
    # print(question_answer_dict)
