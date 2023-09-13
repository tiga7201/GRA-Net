import matplotlib.pyplot as plt
import matplotlib.image as mp


def show_img(path):
    """ 读取并展示图片

    :param path: 图片路径
    :return:
    """
    img = mp.imread(path)
    print('图片的shape:', img.shape)
    plt.imshow(img)
    plt.show()


# show_img("D:/PyCharm/Segmentation/GRA-Net/VOCdevkit/VOC2007/SegmentationClass/004328.png")
# show_img("D:/PyCharm/Segmentation/pGRA-Net/miou_out/detection-results/019.png")
show_img("D:/PyCharm/Segmentation/GRA-Net/img_out/11.jpg")

