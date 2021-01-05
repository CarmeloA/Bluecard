'''
根据不同的模型文件初始化模型
'''
import MNN
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import logging
F = MNN.expr

logger = logging.getLogger(__name__)


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output

class Model():
    def __init__(self,model,mnn_quan=False):
        self.model = model
        self.mnn_quan = mnn_quan
        self.d_model = self.init_model()

    def init_model(self):
        if self.model.endswith('.mnn'):
            model = self.mnn_init()
        elif self.model.endswith('.pt'):
            model = self.yolov5_init()
        return model

    def mnn_init(self):
        if self.mnn_quan:
            model = MNN.nn.load_module_from_file(self.model, for_training=True)
        else:
            model = MNN.Interpreter(self.model)
        return model

    def yolov5_init(self):
        device = self.select_device('1')
        model = self.attempt_load(self.model, map_location=device)
        model = model.half()
        return model

    def detect(self,image):
        if self.model.endswith('.mnn'):
            if self.mnn_quan:
                pred = self.mnn_quan_detect(image)
            else:
                pred = self.mnn_detect(image)
        elif self.model.endswith('.pt'):
            pred = self.yolov5_detect(image)
        return pred

    # MNN
    def mnn_detect(self,image):
        image = image.astype(np.float32)
        # interpreter = MNN.Interpreter(self.model)
        # session = interpreter.createSession()
        interpreter = self.d_model
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)

        tmp_input = MNN.Tensor((1, 3, 192, 320), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)

        output_tensor0 = interpreter.getSessionOutput(session, 'output')
        output_tensor1 = interpreter.getSessionOutput(session, '743')
        output_tensor2 = interpreter.getSessionOutput(session, '744')

        tmp_output0 = MNN.Tensor((1, 327, 24, 40), MNN.Halide_Type_Float, np.ones([1, 327, 24, 40]).astype(np.float32),
                                 MNN.Tensor_DimensionType_Caffe)
        tmp_output1 = MNN.Tensor((1, 327, 12, 20), MNN.Halide_Type_Float, np.ones([1, 327, 12, 20]).astype(np.float32),
                                 MNN.Tensor_DimensionType_Caffe)
        tmp_output2 = MNN.Tensor((1, 327, 6, 10), MNN.Halide_Type_Float, np.ones([1, 327, 6, 10]).astype(np.float32),
                                 MNN.Tensor_DimensionType_Caffe)

        output_tensor0.copyToHostTensor(tmp_output0)
        output_tensor1.copyToHostTensor(tmp_output1)
        output_tensor2.copyToHostTensor(tmp_output2)

        x = [torch.tensor(tmp_output0.getData()),
             torch.tensor(tmp_output1.getData()),
             torch.tensor(tmp_output2.getData())]

        pred = self.mnn_inference(x)
        return pred

    def mnn_quan_detect(self, image):
        c, h, w = image.shape
        data = F.const(image.flatten().tolist(), [1, c, h, w], F.data_format.NCHW)
        MNN.nn.compress.train_quant(self.d_model, quant_bits=8)
        self.d_model.train(False)
        predict1 = self.d_model(data)
        predict1.read()
        p1 = F.Var.read(predict1)
        p1 = torch.tensor(p1)
        x1, x2, x3 = torch.split(p1, [2880, 720, 180], 1)
        x1 = x1.view(-1, 327, 24, 40)
        x2 = x2.view(-1, 327, 12, 20)
        x3 = x3.view(-1, 327, 6, 10)
        x = [x1, x2, x3]
        pred = self.mnn_inference(x)
        return pred

    def mnn_inference(self,x):
        nc = 104
        no = 109
        # anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 226, 242]]
        # anchors = [[26, 11, 35, 17, 48, 25], [70, 33, 96, 57, 127, 74], [160, 96, 195, 125, 233, 163]]
        nl = len(anchors)
        na = len(anchors[0]) // 2
        grid = [torch.zeros(1)] * nl
        a = torch.tensor(anchors).float().view(nl, -1, 2)
        anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2)
        stride = torch.tensor([8., 16., 32.])
        # ch = [327,327,327]
        # m = nn.ModuleList(nn.Conv2d(x, no * na, 1) for x in ch)
        z = []  # inference output

        for i in range(nl):
            # x[i] = m[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, 3, 109, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                grid[i] = self.make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(x[i].device)) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(bs, -1, no))

        return torch.cat(z, 1)

    def make_grid(self,nx,ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    # YOLOv5
    def yolov5_detect(self,image):
        image = torch.from_numpy(image)
        image = image.half()
        image = image.unsqueeze(0)
        image = image.cuda()
        # device = self.select_device('0')
        # model = self.attempt_load(self.model, map_location=device)
        # model = model.half()
        pred = self.d_model(image.cuda(), augment=False)[0]
        return pred

    def attempt_load(self,weights, map_location=None):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

        if len(model) == 1:
            return model[-1]  # return model
        else:
            print('Ensemble created with %s\n' % weights)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model  # return ensemble

    def select_device(self,device='', batch_size=None):
        # device = 'cpu' or '0' or '0,1,2,3'
        cpu_request = device.lower() == 'cpu'
        if device and not cpu_request:  # if device requested other than 'cpu'
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
            assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

        cuda = False if cpu_request else torch.cuda.is_available()
        if cuda:
            c = 1024 ** 2  # bytes to MB
            ng = torch.cuda.device_count()
            if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
                assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
            x = [torch.cuda.get_device_properties(i) for i in range(ng)]
            s = 'Using CUDA '
            for i in range(0, ng):
                if i == 1:
                    s = ' ' * len(s)
                logger.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                            (s, i, x[i].name, x[i].total_memory / c))
        else:
            logger.info('Using CPU')

        logger.info('')  # skip a line
        return torch.device('cuda:0' if cuda else 'cpu')


