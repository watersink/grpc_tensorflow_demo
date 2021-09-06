from tensorrtserver.api import *
import cv2
import numpy as np

class Transformation(object):
    """
    对图片的预处理，主要是比改变长宽比的方式进行resize和pad
    后处理，主要是根据info中的pad信息和原始图片的宽和高来进行恢复
    """

    def __init__(self, dst_size,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        dst_size: id 0 : height
                  id 1 : width
        """
        self.dst_size = dst_size
        self.mean = mean
        self.std = std
        self.info = []
        self.restore_ratio = 0

    def resize_and_pad(self, im):
        """
        im: array
        根据dst_size，在不改变原图长宽比的情况下，resize到dst_size相应的尺寸，
        constant value 127 to pad
        """
        height, width, _ = im.shape
        std_ratio = self.dst_size[0] / self.dst_size[1]
        img_ratio = height / width

        if img_ratio >= std_ratio:
            n_h = self.dst_size[0]
            n_w = int(n_h / img_ratio)
            self.restore_ratio = height / n_h
            im = cv2.resize(im, (n_w, n_h))
            pad_w = int((self.dst_size[1] - n_w) / 2)
            im = np.pad(im, ((0, 0), (pad_w, self.dst_size[1] - n_w - pad_w), (0, 0)), 'constant',
                        constant_values=127)
            self.info = [0, 0, pad_w, self.dst_size[1] - n_w - pad_w]
        else:
            n_w = self.dst_size[1]
            n_h = int(n_w * img_ratio)
            self.restore_ratio = height / n_h
            im = cv2.resize(im, (n_w, n_h))
            pad_h = int((self.dst_size[0] - n_h) / 2)
            im = np.pad(im, ((pad_h, self.dst_size[0] - n_h - pad_h), (0, 0), (0, 0)), 'constant',
                        constant_values=127)
            self.info = [pad_h, self.dst_size[0] - n_h - pad_h, 0, 0]

        self.info.extend([height, width])
        return im

    def normalize(self, image):
        """
        对图片进行翻转，并根据均值和方差进行归一化
        """
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def restore_to_origin_size(self, pred_result):
        """
        根据self.info中的信息，将预测的图片恢复到原始尺寸
        """
        assert len(self.info) == 6
        pred_result = pred_result[self.info[0]:self.dst_size[0] - self.info[1],
                      self.info[2]:self.dst_size[1] - self.info[3]]
        pred_result = cv2.resize(pred_result, (self.info[5], self.info[4]), cv2.INTER_NEAREST)
        return pred_result


triton = {
    'url': '192.168.1.94:8020',
    # 'url': '172.16.21.100:8001',
    'protocol': 'http',
    'model_name': 'mnist_lenet',
    'batch_size': 1,
    'verbose': False,
    'streaming': False,
}


def parse_model(url, protocol, model_name, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    status = server_status.model_status[model_name]
    status_config = status.config

    input = status_config.input[0]
    output = status_config.output[0]
    return input.name, output.name


def magic_triton_client_run(image, cfg):
    protocol = ProtocolType.from_str(cfg['protocol'])
    # print('run begin')
    input_name, output_name = parse_model(cfg['url'], protocol, cfg['model_name'], cfg['verbose'])
    #print(input_name, output_name)

    #input_name, output_name = "input__0", "output__0"

    ctx = InferContext(cfg['url'], protocol, cfg['model_name'],
                       verbose=cfg['verbose'], correlation_id=0, streaming=cfg['streaming'])

    # res = ctx.run({input_name: image},
    #               {output_name: InferContext.ResultFormat.RAW},
    #               1)
    # # [array([-6.9815645, 4.8164535], dtype=float32)]
    res = ctx.run({input_name: image},
                  {output_name: (InferContext.ResultFormat.CLASS, 10)},
                  1)
    # print(res)
    return res[output_name]


def predict(img_path):
    image = cv2.imread(img_path)
    #trans_func = Transformation([28, 28])
    #image = trans_func.resize_and_pad(image)
    #image = trans_func.normalize(image)

    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray = np.expand_dims(image_gray,2)
    image_normed = image_gray/255.0
    #tempimg = np.expand_dims(np.transpose(image_normed,(2,0,1)),0).astype(np.float32)
    tempimg = image_normed.transpose((2, 0, 1)).astype(np.float32)
    batch_images = [tempimg]
    batch_preds = magic_triton_client_run(batch_images, triton)
    return batch_preds

def sigmoid(x):
    return 1/(1+np.exp(-x))



img_path = "4.jpg"
outputs = predict(img_path)
print(outputs)
for item in outputs[0]:
    res = sigmoid(item[1])
    print(res)

