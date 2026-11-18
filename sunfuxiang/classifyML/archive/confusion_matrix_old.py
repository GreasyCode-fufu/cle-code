import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm.rich import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
# from sklearn.metrics import f1_score

# from models.Swin_ET import swin_tiny_patch4_window7_224 as create_model
# from models.Resnet import resnet34 as create_model
# from models.FasterNet import fast_t0 as create_model
# from models.Pvt import pvt_tiny as create_model
from models.DenseNet import densenet121 as create_model
# from models.Mobilevit import mobile_vit_xx_small as create_model
# from models.VanilaNet import vanillanet_5 as create_model
# from models.Vit import vit_base_patch16_224 as create_model
# from models.Swin import swin_tiny_patch4_window7_224 as create_model
# from models.Vit import vit_base_patch16_224 as create_model
# from models.GhostNet import ghostnet as create_model
# from models.DenseNet import densenet121 as create_model
# from models.Pvt import pvt_tiny as create_model
# from models.Swin import swin_tiny_patch4_window7_224 as create_model
# from models.RepViT import repvit_m1 as create_model
# from models.EfficientFormerV2 import efficientformerv2_s0 as create_model
# from models.ShuffleNet import shufflenet_v2_x0_5 as create_model


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("===========================================================")
        print()
        print()
        print("{} images for testing.".format(len(validate_dataset)))
        print("the model Top1 accuracy is: ", acc)
        print()
        print()
        print("===========================================================")

        # precision, recall, specificity, and F1 score
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            # F1 = round(f1_score(labels, preds, average='macro'), 3)
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig("./figure/denseNet_ricepest.png")
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = "/home/u2321283047/clsData/ricepest"  # flower data set path
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform)

    batch_size = 32
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=8)
    net = create_model(num_classes=10)
    # load pretrain weights
    model_weight_path = "/home/u2321283047/big/weights/densenet_ricepest.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), False)
    net.to(device)

    # read class_indict
    json_label_path = '/home/u2321283047/big/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=10, labels=labels)
    net.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            preds.extend(outputs.to("cpu").numpy())
            true_labels.extend(val_labels.to("cpu").numpy())
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
