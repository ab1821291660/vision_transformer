import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
####
from my_dataset import MyDataSet, read_split_data
from vit_model import vit_base_patch16_224_in21k as create_model
# from utils import read_split_data, train_one_epoch, evaluate
####
from tqdm import tqdm
import sys
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)####1  # 累计损失
    accu_num = torch.zeros(1).to(device)####1  # 累计预测正确的样本数
    ####
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data####b8-c3-224-224  ####b8
        sample_num += images.shape[0]
        ##===================================
        pred = model(images.to(device))####b8-2----####b8-c3-224-224
        ##===================================
        loss = loss_function(pred, labels.to(device))####tensor(0.7541, grad_fn=<NllLossBackward>)
        loss.backward()
        accu_loss += loss.detach()
        ####
        pred_classes = torch.max(pred, dim=1)[1]########b8
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item()/(step + 1),    accu_num.item()/sample_num
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    ####
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        ##===================================
        pred = model(images.to(device))
        ##===================================
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        ####
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
####===================================================================================================================================
####===================================================================================================================================
####===================================================================================================================================
####===================================================================================================================================
####===================================================================================================================================
def main(args):
    tb_writer = SummaryWriter()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)####cpu
    ####
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    ####[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]####[0, 0, 0, 0, 1, 1, 1, 1]
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    data_transform = {
                        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                        "val": transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(),    batch_size if batch_size > 1 else 0,    8])  # number of workers
    print(nw)####8
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    ####===================================================================================================================================
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    ####===================================================================================================================================
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    ##===================================
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ####===================================================================================================================================
    ####===================================================================================================================================
    ####===================================================================================================================================
    ####===================================================================================================================================
    for epoch in range(args.epochs):####2
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        ####===================================================================================================================================
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,default="data/catdog_mini20")
    ##===================================
    parser.add_argument('--num_classes', type=int,    default='2')####原5
    parser.add_argument('--epochs', type=int, default=2)####原10
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--model-name', default='vit', help='create model name')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')####./vit_base_patch16_224_in21k.pth
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)####原True----
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
# cpu
# 80 images were found in the dataset.
# 64 images for training.
# 16 images for validation.
# 8
# Using 8 dataloader workers every process
# [train epoch 0] loss: 1.275, acc: 0.438: 100%|██████████| 8/8 [00:41<00:00,  5.18s/it]
# [valid epoch 0] loss: 1.028, acc: 0.438: 100%|██████████| 2/2 [00:05<00:00,  2.74s/it]
# [train epoch 1] loss: 0.853, acc: 0.484: 100%|██████████| 8/8 [00:42<00:00,  5.36s/it]
# [valid epoch 1] loss: 0.790, acc: 0.500: 100%|██████████| 2/2 [00:05<00:00,  2.87s/it]
# Process finished with exit code 0


# #cpu
# 40 images were found in the dataset.
# 32 images for training.
# 8 images for validation.
# 8
# Using 8 dataloader workers every process
# [train epoch 0] loss: 0.932, acc: 0.500: 100%|██████████| 4/4 [00:21<00:00,  5.37s/it]
# [valid epoch 0] loss: 0.722, acc: 0.500: 100%|██████████| 1/1 [00:03<00:00,  3.78s/it]
# [train epoch 1] loss: 0.753, acc: 0.438: 100%|██████████| 4/4 [00:22<00:00,  5.66s/it]
# [valid epoch 1] loss: 0.728, acc: 0.500: 100%|██████████| 1/1 [00:04<00:00,  4.04s/it]
# Process finished with exit code 0
    ##===================================
    ##===================================
    ##===================================
    ##===================================
    # parser = argparse.ArgumentParser()
    # # 数据集所在根目录
    # # http://download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,default="/data/flower_photos")
    # ##===================================
    # parser.add_argument('--num_classes', type=int, default=5)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--batch-size', type=int, default=8)
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--lrf', type=float, default=0.01)
    # parser.add_argument('--model-name', default='', help='create model name')
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth', help='initial weights path')
    # # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    # parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # opt = parser.parse_args()
    # main(opt)







