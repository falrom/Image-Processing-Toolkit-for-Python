import numpy as np
import cv2
import sys
import os


def progress_bar(done_num, total_num, width=40):
    rate = done_num / total_num
    rate_num = int(rate * width)
    r = '\r[%s%s] (%d%%) %d done of %d          ' % \
        ("=" * rate_num, " " * (width - rate_num), int(rate * 100), done_num, total_num)
    sys.stdout.write(r)
    sys.stdout.flush()


def format_time(t_bgn, t_end):
    secd = t_end - t_bgn
    mint = secd // 60
    hour = mint // 60
    secd_show = secd % 60
    mint_show = mint % 60
    hour_show = hour
    return '%dh %dm %ds (Total: %.2fs)' % (hour_show, mint_show, secd_show, secd)


def cv_imread(file_path, channel='BGR'):
    """
    mode: 'BGR' or 'RGB'
    """
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if channel == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cv_imwrite(output_path, img, channel='BGR', ext='.png'):
    """
    image: 8-bit single-channel or 3-channel (with 'BGR' channel order) images
    """
    if channel == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imencode(ext, img)[1].tofile(output_path)


def YUVread(path, size, frame_num=None, start_frame=0, mode='420', bits=8, endian='<'):
    """
    Only for 4:2:0 and 4:4:4 for now.
    
    :param path: yuv file path
    :param size: [height, width]
    :param frame_num: The number of frames you want to read, and it shouldn't smaller than the frame number of original
        yuv file. Defult is None, means read from start_frame to the end of file.
    :param start_frame: which frame begin from. Default is 0.
    :param mode: yuv file mode, '420' or '444 planar'
    :param bits: yuv file bit depth, 8 or 12 or 16
    :param endian: '<' or '>'
    :return: y, u, v with a shape of [frame_num, height, width] of each
    """
    [height, width] = size
    if mode == '420':
        frame_size = int(height * width / 2 * 3)
    else:
        frame_size = int(height * width * 3)

    pixel_size = 1
    pixel_type = np.uint8
    pixel_type_str = 'B'
    if bits > 8:
        pixel_size = 2
        pixel_type = np.uint16
        pixel_type_str = endian + 'H'

    all_y = pixel_type([])
    all_u = pixel_type([])
    all_v = pixel_type([])
    with open(path, 'rb') as file:
        file.seek(frame_size * pixel_size * start_frame)
        if frame_num is None:
            frame_num = 0
            while True:
                if mode == '420':
                    y = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                    u = np.frombuffer(file.read(pixel_size * height * width >> 2), dtype=pixel_type_str)
                    v = np.frombuffer(file.read(pixel_size * height * width >> 2), dtype=pixel_type_str)
                else:
                    y = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                    u = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                    v = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                if y.shape == (0,):
                    break
                all_y = np.concatenate([all_y, y])
                all_u = np.concatenate([all_u, u])
                all_v = np.concatenate([all_v, v])
                frame_num += 1
        else:
            for fn in range(frame_num):
                if mode == '420':
                    y = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                    u = np.frombuffer(file.read(pixel_size * height * width >> 2), dtype=pixel_type_str)
                    v = np.frombuffer(file.read(pixel_size * height * width >> 2), dtype=pixel_type_str)
                else:
                    y = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                    u = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                    v = np.frombuffer(file.read(pixel_size * height * width), dtype=pixel_type_str)
                if y.shape == (0,):
                    break
                all_y = np.concatenate([all_y, y])
                all_u = np.concatenate([all_u, u])
                all_v = np.concatenate([all_v, v])

    all_y = np.reshape(all_y, [frame_num, height, width])
    if mode == '420':
        all_u = np.reshape(all_u, [frame_num, height >> 1, width >> 1])
        all_v = np.reshape(all_v, [frame_num, height >> 1, width >> 1])
    else:
        all_u = np.reshape(all_u, [frame_num, height, width])
        all_v = np.reshape(all_v, [frame_num, height, width])

    return all_y, all_u, all_v


def Yread(path, size, frame_num=None, start_frame=0):
    """
    Assuming that the file contains only the Y component.

    :param path: yuv file path
    :param size: [height, width]
    :param frame_num: The number of frames you want to read, and it shouldn't smaller than the frame number of original
        yuv file. Defult is None, means read from start_frame to the end of file.
    :param start_frame: Which frame begin from. Default is 0.
    :return: byte_type y with a shape of [frame_num, height, width]
    """
    [height, width] = size
    frame_size = int(height * width)
    all_y = np.uint8([])
    with open(path, 'rb') as file:
        file.seek(frame_size * start_frame)
        if frame_num is None:
            frame_num = 0
            while True:
                y = np.uint8(list(file.read(height * width)))
                if y.shape == (0,):
                    break
                all_y = np.concatenate([all_y, y])
                frame_num += 1
        else:
            for fn in range(frame_num):
                y = np.uint8(list(file.read(height * width)))
                if y.shape == (0,):
                    break
                all_y = np.concatenate([all_y, y])

    all_y = np.reshape(all_y, [frame_num, height, width])

    return all_y


def YUVwrite(y, u, v, path):
    """
    Ndarray to file. If '444', write by YUV444 planar mode.

    :param y: y with a shape of [frame_num, height, width] or [height, width]
    :param u: u with a shape of [frame_num, height, width] or [height, width]
    :param v: v with a shape of [frame_num, height, width] or [height, width]
    :param path: save path
    """
    if len(np.shape(y)) == 3:
        frame_num = np.shape(y)[0]
        with open(path, 'wb') as file:
            for fn in range(frame_num):
                file.write(y[fn].tobytes())
                file.write(u[fn].tobytes())
                file.write(v[fn].tobytes())
    else:
        with open(path, 'wb') as file:
            file.write(y.tobytes())
            file.write(u.tobytes())
            file.write(v.tobytes())


def Ywrite(y, path):
    """
    Only Y channel.

    :param y: y with a shape of [frame_num, height, width] or [height, width]
    :param path: save path
    """
    if len(np.shape(y)) == 3:
        frame_num = np.shape(y)[0]
        with open(path, 'wb') as file:
            for fn in range(frame_num):
                file.write(y[fn].tobytes())
    else:
        with open(path, 'wb') as file:
            file.write(y.tobytes())


def YUVcut(y, u, v, new_size, new_frame_num=None, start_frame=0, start_point=(0, 0)):
    """
    Cut frames/patches from yuv. Only for 4:2:0 or 4:4:4.

    :param y: y with a shape of [frame_num, height, width]
    :param u: u with a shape of [frame_num, height, width]
    :param v: v with a shape of [frame_num, height, width]
    :param new_size: [height, width]
    :param new_frame_num: How many frames you want to get.
    :param start_frame: Begin from which frame. Default is 0.
    :param start_point: The left_up point of new patch. Default is (0, 0)
    :return: cut yuv
    """
    [new_height, new_width] = new_size
    [sh, sw] = start_point
    if new_frame_num is None:
        new_frame_num = np.shape(y)[0] - start_frame

    if np.shape(y)[1] == np.shape(u)[1]:  # 444
        new_y = y[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
        new_u = u[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
        new_v = v[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
    else:  # 420
        if new_height % 2 != 0:
            new_height += 1
        if new_width % 2 != 0:
            new_width += 1
        if sh % 2 != 0:
            sh += 1
        if sw % 2 != 0:
            sw += 1
        new_y = y[start_frame:start_frame + new_frame_num, sh:sh + new_height, sw:sw + new_width]
        new_u = u[start_frame:start_frame + new_frame_num, (sh >> 1):((sh >> 1) + (new_height >> 1)),
                  (sw >> 1):((sw >> 1) + (new_width >> 1))]
        new_v = v[start_frame:start_frame + new_frame_num, (sh >> 1):((sh >> 1) + (new_height >> 1)),
                  (sw >> 1):((sw >> 1) + (new_width >> 1))]

    return new_y, new_u, new_v


def YUV_change_mode(y, u, v, direction='420to444'):
    """
    y, u, v with a shape of [frame_num, height, width]
    
    derection: '420to444' or '444to420'
    """
    if direction == '420to444':
        u = np.array([cv2.resize(ch, (u.shape[2] * 2, u.shape[1] * 2), interpolation=cv2.INTER_CUBIC) for ch in u])
        v = np.array([cv2.resize(ch, (v.shape[2] * 2, v.shape[1] * 2), interpolation=cv2.INTER_CUBIC) for ch in v])
    if direction == '444to420':
        u = np.array([cv2.resize(ch, (u.shape[2] // 2, u.shape[1] // 2), interpolation=cv2.INTER_CUBIC) for ch in u])
        v = np.array([cv2.resize(ch, (v.shape[2] // 2, v.shape[1] // 2), interpolation=cv2.INTER_CUBIC) for ch in v])
    return y, u, v


def yuv2rgb(y, u, v, clip=True):
    """
    Transform YUV image to RGB image. Multi-frames. Only for 4:2:0 or 4:4:4.
    The value range inputs should be limited in [0, 1].

    :param y: y with a shape of [frame_num, height, width]
    :param u: u with a shape of [frame_num, height, width]
    :param v: v with a shape of [frame_num, height, width]
    :param clip: Whether to limit the range of values to [0, 1]
    :return: (r, g, b) and with a shape of [frame_num, height, width] for each channel. Limited in [0, 1].
    """
    if np.shape(y)[1] != np.shape(u)[1]:  # '420' inputs
        y, u, v = YUV_change_mode(y, u, v, '420to444')
    y = 255 * y
    u = 255 * u
    v = 255 * v
    r = (1.164 * (y - 16) + 1.596 * (v - 128)) / 255
    g = (1.164 * (y - 16) - 0.392 * (u - 128) - 0.813 * (v - 128)) / 255
    b = (1.164 * (y - 16) + 2.017 * (u - 128)) / 255
    if clip:
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)
    return r, g, b


def rgb2yuv(r, g, b, mode='444'):
    """
    Transform RGB image to YUV image. Multi-frames.
    The value range inputs should be limited in [0, 1].

    :param r: r with a shape of [frame_num, height, width]
    :param g: g with a shape of [frame_num, height, width]
    :param b: b with a shape of [frame_num, height, width]
    :param mode: '444' or '420', the YUV mode of outputs 
    :return: (y, u, v) and with a shape of [frame_num, height, width] for each channel. Limited in [0, 1].
    """
    r = 255 * r
    g = 255 * g
    b = 255 * b
    y = 00.257 * r + 0.504 * g + 0.098 * b + 16
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    v = 00.439 * r - 0.368 * g - 0.071 * b + 128
    if mode == '420':
        y, u, v = YUV_change_mode(y, u, v, '444to420')
    return (y / 255), (u / 255), (v / 255)


def save_YUV_img(y, u, v, output_path, ext='.png', algorithm='custom'):
    """
    Save YUV to picture file. Inputs: [0, 255]

    :param y: y with a shape of [frame_num, height, width]
    :param u: u with a shape of [frame_num, height, width]
    :param v: v with a shape of [frame_num, height, width]
    :param output_path: output_path
    :param ext: file extension
    :param algorithm: 'cv' or 'custom', choose which yuv2rgb algorithm to use
    :return:
    """
    if np.shape(y)[1] != np.shape(u)[1]:  # '420' inputs
        y, u, v = YUV_change_mode(y, u, v, '420to444')
    if y.shape[0] == 1:
        if algorithm == 'cv':
            img = cv2.cvtColor(
                np.concatenate([y[0, :, :, np.newaxis], u[0, :, :, np.newaxis], v[0, :, :, np.newaxis]], 2),
                cv2.COLOR_YUV2RGB)
        elif algorithm == 'custom':
            img = np.array(yuv2rgb(y / 255, u / 255, v / 255))[:, 0, :, :].transpose([1, 2, 0])
            img = (255 * img).astype(np.uint8)
        else:
            raise Exception('Error: Undefined yuv2rgb algorithm.')
        cv_imwrite(output_path, img, 'RGB', ext)
    else:
        path = os.path.splitext(output_path)[0]
        for fn in range(y.shape[0]):
            if algorithm == 'cv':
                img = cv2.cvtColor(
                    np.concatenate([y[fn, :, :, np.newaxis], u[fn, :, :, np.newaxis], v[fn, :, :, np.newaxis]], 2),
                    cv2.COLOR_YUV2RGB)
            elif algorithm == 'custom':
                img = np.array(yuv2rgb(y[fn:fn + 1, :] / 255, u[fn:fn + 1, :] / 255, v[fn:fn + 1, :] / 255))[:, 0, :, :]
                img = img.transpose([1, 2, 0])
                img = (255 * img).astype(np.uint8)
            else:
                raise Exception('Error: Undefined yuv2rgb algorithm.')
            cv_imwrite(path + '_' + str(fn) + ext, img, 'RGB', ext)


def nearest_interpolation(img, x, y):
    """
    最邻近插值算子。超出图像范围0.5 pixel的部分返回0。

    :param img: 输入图像，尺寸为[height, width]或[height, width, channels]
    :param x: 需要求解的亚像素位置的横向坐标（从左到右）
    :param y: 需要求解的亚像素位置的纵向坐标（从上到下）
    :return: 返回该亚像素位置的像素值。具体通道数同输入img。
    """
    h, w = img.shape[:2]

    if (x <= -0.5) or (x >= (w - 0.5)) or (y <= -0.5) or (y >= (h - 0.5)):
        return np.zeros(img[0, 0].shape, np.float32)

    return img[int(round(y)), int(round(x))]


def bicubic_interpolation(img, x, y, a=-0.5):
    """
    双三次插值算子。超出图像范围0.5 pixel的部分返回0。

    :param img: 输入图像，尺寸为[height, width]或[height, width, channels]
    :param x: 需要求解的亚像素位置的横向坐标（从左到右）
    :param y: 需要求解的亚像素位置的纵向坐标（从上到下）
    :return: 返回该亚像素位置的像素值。具体通道数同输入img。
    """
    h, w = img.shape[:2]
    assert h > 3
    assert w > 3

    if (x <= -0.5) or (x >= (w - 0.5)) or (y <= -0.5) or (y >= (h - 0.5)):
        return np.zeros(img[0, 0].shape, np.float32)

    def _bicubic_1_w(x, a):
        x = np.abs(x)
        x2 = x * x
        x3 = x * x2
        return (a + 2) * x3 - (a + 3) * x2 + 1
    
    def _bicubic_2_w(x, a):
        x = np.abs(x)
        x2 = x * x
        x3 = x * x2
        return a * x3 - 5 * a * x2 + 8 * a * x - 4 * a

    def _bicubic_w(x, a=-0.5):
        if x >= -1 and x <= 1:
            return _bicubic_1_w(x, a)
        if x > -2 and x < 2:
            return _bicubic_2_w(x, a)
        return 0.0

    def _bicubic(delta_x, delta_y, neighbor, a=-0.5):
        row = []
        for i in range(4):
            row.append(
                neighbor[i][0] * _bicubic_w(-delta_x - 1, a) +
                neighbor[i][1] * _bicubic_w(-delta_x - 0, a) +
                neighbor[i][2] * _bicubic_w(+1 - delta_x, a) +
                neighbor[i][3] * _bicubic_w(+2 - delta_x, a)
            )
        return (
                row[0] * _bicubic_w(-delta_y - 1, a) +
                row[1] * _bicubic_w(-delta_y - 0, a) +
                row[2] * _bicubic_w(+1 - delta_y, a) +
                row[3] * _bicubic_w(+2 - delta_y, a)
        )

    neighbor = np.empty([4, 4], img.dtype)
    xl = int(np.floor(x))
    yu = int(np.floor(y))

    if x >= 1 and y >= 1 and x < (w - 2) and y < (h - 2):
        neighbor[:, :] = img[yu - 1:yu + 3, xl - 1:xl + 3]
        return _bicubic(x - xl, y - yu, neighbor, a)

    neighbor = np.array([
        [img[min(h - 1, max(0, yu - 1)), min(w - 1, max(0, xl - 1))],
         img[min(h - 1, max(0, yu - 1)), min(w - 1, max(0, xl - 0))],
         img[min(h - 1, max(0, yu - 1)), min(w - 1, max(0, xl + 1))],
         img[min(h - 1, max(0, yu - 1)), min(w - 1, max(0, xl + 2))]],
        [img[min(h - 1, max(0, yu - 0)), min(w - 1, max(0, xl - 1))],
         img[min(h - 1, max(0, yu - 0)), min(w - 1, max(0, xl - 0))],
         img[min(h - 1, max(0, yu - 0)), min(w - 1, max(0, xl + 1))],
         img[min(h - 1, max(0, yu - 0)), min(w - 1, max(0, xl + 2))]],
        [img[min(h - 1, max(0, yu + 1)), min(w - 1, max(0, xl - 1))],
         img[min(h - 1, max(0, yu + 1)), min(w - 1, max(0, xl - 0))],
         img[min(h - 1, max(0, yu + 1)), min(w - 1, max(0, xl + 1))],
         img[min(h - 1, max(0, yu + 1)), min(w - 1, max(0, xl + 2))]],
        [img[min(h - 1, max(0, yu + 2)), min(w - 1, max(0, xl - 1))],
         img[min(h - 1, max(0, yu + 2)), min(w - 1, max(0, xl - 0))],
         img[min(h - 1, max(0, yu + 2)), min(w - 1, max(0, xl + 1))],
         img[min(h - 1, max(0, yu + 2)), min(w - 1, max(0, xl + 2))]]
    ])
    return _bicubic(x - xl, y - yu, neighbor, a)    


def bilinear_interpolation(img, x, y):
    """
    双线性插值算子。超出图像范围0.5 pixel的部分返回0。

    :param img: 输入图像，尺寸为[height, width]或[height, width, channels]
    :param x: 需要求解的亚像素位置的横向坐标（从左到右）
    :param y: 需要求解的亚像素位置的纵向坐标（从上到下）
    :return: 返回该亚像素位置的像素值。具体通道数同输入img。
    """
    h, w = img.shape[:2]

    # old: (x < 0) or (x > (w - 1)) or (y < 0) or (y > (h - 1))
    if (x <= -0.5) or (x >= (w - 0.5)) or (y <= -0.5) or (y >= (h - 0.5)):
        return np.zeros(img[0, 0].shape, np.float32)

    if x < 0: x = 0
    if y < 0: y = 0
    if x > (w - 1): x = (w - 1)
    if y > (h - 1): y = (h - 1)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    if x0 >= (w - 1): x0 -= 1
    if y0 >= (h - 1): y0 -= 1
    x1 = x0 + 1
    y1 = y0 + 1

    res = (x1 - x) * (y1 - y) * img[y0, x0] + \
          (x - x0) * (y1 - y) * img[y0, x1] + \
          (x1 - x) * (y - y0) * img[y1, x0] + \
          (x - x0) * (y - y0) * img[y1, x1]

    return res


interpolations = {
    'nearest': nearest_interpolation, 
    'bicubic': bicubic_interpolation,
    'bilinear': bilinear_interpolation,
}


def test_interpolations():
    """
    Test interpolations.
    """
    import matplotlib.pyplot as plt
    # plt.ion(), plt.close('all')

    img = np.array([
        [214, 205, 168, 174],
        [205, 135,  58,  84],
        [ 82, 110, 132,  75],
        [  8,  87, 215, 102]
    ], dtype=np.uint8)
    plt.figure(), plt.title('orignal image'), plt.imshow(img)

    scl = 10
    img_o = np.ones([len(img) * scl, len(img) * scl]) * 255
    for name, interpolation in interpolations.items():
        # print('\n\n\n*****%s*****\n\n' % name)
        for y in range(len(img_o)):
            for x in range(len(img_o[0])):
                x_sub = (x + 0.5) / scl - 0.5
                y_sub = (y + 0.5) / scl - 0.5
                img_o[y, x] = interpolation(img, x_sub, y_sub)
                # print('%.2f %.2f' % (x_sub, y_sub))
        plt.figure(), plt.title(name), plt.imshow(img_o)
    
    plt.show()
    # os.system('pause')


def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


def psnr(img1, img2, max=1.0):
    _mse = mse(img1, img2)
    return 10 * np.log10(max * max / _mse)


if __name__ == "__main__":
    # test_interpolations()
    pass


