import os
import argparse
import time
import torch
from thop import profile, clever_format
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from datetime import datetime
from utils import read_split_data, train_one_epoch, evaluate
from models.Swin import swin_tiny_patch4_window7_224 as create_model
from models.Mobilevit import mobile_vit_xx_small as create_model

def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    # data = torch.randn((1,3, 224, 224)).cuda()
    # measure_inference_speed(model, (data,))
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter:", total)
    # print("Number of parameter: %.1fM" % (total / 1e6))
    myinput = torch.zeros((1, 3, 224, 224)).to(device)
    flops, params = profile(model.to(device), inputs=(myinput,))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # optimizer = Adan(pg, lr=args.lr, weight_decay=5E-2)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)


    val_accs = []
    val_losss = []
    # train_accs = []
    best_acc = 0.
    best_epoch = 1
    start_time = time.time()
    lr_list = []
    step = 1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        # train_accs.append(train_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "./weights/mobilevit_cabbage.pth")
        if epoch == 49:
            print("THe max accuracy: %.3f, epoch is %d" % (best_acc, best_epoch))

        if epoch % 1 == 0:
            for p in optimizer.param_groups:
                print("epoch from {} to {}, lr={}".format(epoch, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))
                p['lr'] *= 0.1
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        # current_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率
        # lr_list.append(current_lr)
        # print("current_lr:", current_lr)

        # scheduler.step()  # 调整学习率

        # adjusted_lr = scheduler.get_last_lr()
        # print("adjusted_lr:", adjusted_lr)
        # print("-----")


    print("THe max accuracy: %.3f, epoch is %d" % (best_acc, best_epoch))
    #
    with open("./curves/vit_coffee_acc.txt", 'w') as val_acc:
        val_acc.write(str(val_accs))
    with open("./curves/vit_coffee_loss.txt", 'w') as val_loss:
        val_loss.write(str(val_losss))
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time: {}'.format(total_time_str))
    #
    # with open("./figure/mobilevit_flower17_val_acc.txt", 'w') as val_acc:
    #     val_acc.write(str(val_accs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                        default="D:\datasets\classification\\coffee")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--pretrained', type=str, default='False',
                        help='i')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

