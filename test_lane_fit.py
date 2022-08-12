import logging
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from lib.config import Config
from lib.experiment import Experiment
import numpy as np
import cv2
from PIL import Image
from hnet import HNet
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser(description="Train lane detector")
    parser.add_argument(
        "mode", choices=["train", "test"], help="Train or test?")
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
    # 输入当前待测试图像文件路径的参数
    parser.add_argument("--imageFile", help="Config file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
    parser.add_argument("--cpu", action="store_true",
                        help="(Unsupported) Use CPU instead of GPU")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions to pickle file")
    parser.add_argument(
        "--view", choices=["all", "mistakes"], help="Show predictions")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")
    args = parser.parse_args()
    if args.cfg is None and args.mode == "train":
        raise Exception(
            "If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
    if args.resume and args.mode == "test":
        raise Exception(
            "args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == 'train':
        raise Exception(
            "The `epoch` parameter should not be set when training")
    if args.view is not None and args.mode != "test":
        raise Exception('Visualization is only available during evaluation')
    if args.cpu:
        raise Exception(
            "CPU training/testing is not supported: the NMS procedure is only implemented for CUDA")
    return args


def main():
    # 传递参数
    args = parse_args()
    # 当前模式和当前的net
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available(
    ) or args.cpu else torch.device('cuda')
    model = cfg.get_model()
    model_path = exp.get_checkpoint_path(
        args.epoch or exp.get_last_checkpoint_epoch())
    # 加载模型
    logging.getLogger(__name__).info('Loading model %s', model_path)
    model.load_state_dict(exp.get_epoch_model(
        args.epoch or exp.get_last_checkpoint_epoch()))
    model = model.to(device)
    # 测试模式
    model.eval()
    test_parameters = cfg.get_test_parameters()
    # 预测的点
    predictions = []
    exp.eval_start_callback(cfg)
    imageFile = args.imageFile
    # 读取测试文件
    image = cv2.imread(imageFile)
    # resize到模型需要的输入大小
    img_fit = cv2.resize(image, (64, 128))
    image = cv2.resize(image, (640, 360))
    # 转换成0-1之间的tensor
    image = image/255.
    # 转换到float格式在cuda上处理
    image = torch.from_numpy(image).cuda().float()
    # 将channel维度提到前面  在最前面增添一个batch维度
    image = torch.unsqueeze(image.permute(2, 0, 1), 0)
    # 得到预测输出
    output = model(image, **test_parameters)# 4,3,77
    # 将输出decode为lane
    prediction = model.decode(output, as_lanes=True)
    # 添加到预测list中
    predictions.extend(prediction)
    # 将channel通道放在后面，转换成numpy格式待会儿存储
    img = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.resize(img, (640, 360))
    img_h, img_w, _ = img.shape
    pad = 0
    # 有一部分anchor超出特征图的边界，所以进行0填充
    if pad > 0:
        img_pad = np.zeros((360 + 2 * pad, 640 + 2 * pad, 3), dtype=np.uint8)
        img_pad[pad:-pad, pad:-pad, :] = img
        img = img_pad

    # lanefit
    net=HNet()
    net=net.to(device)
    
    img_fit = torch.from_numpy(img_fit).cuda().float()
    # 将channel维度提到前面  在最前面增添一个batch维度
    img_fit = torch.unsqueeze(img_fit.permute(2, 0, 1), 0)
    # 网络预测输出的H矩阵参数
    # fit_params = net(img_fit)# 4,3,77
    loss_function = nn.MSELoss()

    # construct an optimizer
    net_params = [p for p in net.parameters() if p.requires_grad]
    #net_params = [p.requires_grad_() for p in net.parameters()]
    #net_params=Variable(net.parameters(),requires_grad=True)
    #net_params.requires_grad_()
    optimizer = optim.Adam(net_params, lr=0.0001)

    epochs = 15
    best_acc = 0.0
    save_path = './HNet.pth'
    train_steps=epochs
    #train_steps = len(train_loader)
    # 对于预测的lane的每一个点
    for i, l in enumerate(prediction[0]):
        points = l.points  # <class 'lib.lane.Lane'>
        # 缩放到每个点在原图中的位置
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]

        pointx_final=points[:, 0]
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            #train_bar = tqdm(train_steps, file=sys.stdout)

        #for step in range(epochs):
        # images, labels = data
            optimizer.zero_grad()
            params = net(img_fit.to(device))# 生成的h矩阵的六个参数，得到H矩阵
            params=params[0].cpu().detach().numpy()

            # 利用得到的矩阵对预测的点进行映射，得到在新的空间的点的坐标
            H=np.array([[params[0],params[1],params[2]],[0,params[3],params[4]],[0,params[5],1]])
            H_inv=np.linalg.inv(H.astype(np.float))
            H_inv=torch.tensor(H_inv,dtype=float)
            pointx_hat= points[:, 0]*params[0]+points[:, 1]*params[1]+params[2]
            pointy_hat=points[:, 1]*params[3]+params[4]

            # 利用新的空间的点的坐标进行二次多项式拟合，得到拟合方程式，对新的空间点的y计算出拟合的x
            fit_param = np.polyfit(pointy_hat, pointx_hat, 2)# 二次多项式拟合

            plot_y = np.linspace(10, img_w, img_w - 10)# 从10到640生成630个等距的点
            fit_x = fit_param[0] * pointy_hat ** 2 + fit_param[1] * pointy_hat + fit_param[2]# 利用拟合的方程式计算固定间距的y对应的x
            # 利用拟合之后的点和H矩阵的逆将拟合之后的点映射到二维空间，此时y不变
            pointx_reinv=H_inv[0][0]*pointx_hat+H_inv[0][1]*pointy_hat+H_inv[0][2]

            # 计算经过两次映射之后的x与原始的x的距离loss
            loss = loss_function(torch.from_numpy(points[:,0]).to(device), pointx_reinv.to(device))# 平方损失函数
            loss=loss.requires_grad_()
            # loss反向传播
            loss.backward()
            optimizer.step()

            # 保存映射后的pointx
            pointx_final=pointx_reinv
            # pointsx=pointx_reinv
            # print statistics 
            running_loss += loss.item()
            print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                    epochs,
                                                                    loss))
        # 最后使用的是经过两次映射之后的点
        points[:, 0] = pointx_final
        # 对缩放之后的点进行H映射
        # 四舍五入
        points = points.round().astype(int)
        # 因为有填充所以有偏移，加上偏移量
        points += pad
        # 连接相邻的两个点
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                           tuple(curr_p),
                           tuple(next_p),
                           color=(255, 0, 255),
                           thickness=3)
    # 保存测试文件
    cv2.imwrite("./"+imageFile[:-4] +
                "_predict_lane_result_tusimple_pretrained.jpg", img)


if __name__ == '__main__':
    main()
