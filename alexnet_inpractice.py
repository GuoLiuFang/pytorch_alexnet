import torch
import torch.nn as nn
# 有新的版本了。
from torch.utils.tensorboard import SummaryWriter
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        # 看起来比较酷。
        super(AlexNet, self).__init__()
        # self.features;这个名字很酷.在__init__中初始化网络的结构。
        # 96, 227;      64, 224;
        # input size should be: (b * 3 * 227 * 227)
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            # b * 96 * 55 * 55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # b * 96 * 27 * 27
            # 
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            # b * 256 * 27 * 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # b * 256 * 13 * 13
            # 
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # b * 384 * 13 * 13
            # 
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # b * 384 * 13 * 13
            # 
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # b * 256 * 13 * 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # b * 256 * 6 * 6
        )
        # # b * 256 * 6 * 6
        # self.avgpool = nn.AdaptiveAvgPool2d(
        #     (6, 6),
        # )
        # # 其实没有必要。
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            # b * 4096
            nn.ReLU(),
            # 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            # b * 4096
            nn.ReLU(),
            # 
            nn.Linear(4096, num_classes),
            # b * 1000
        )
        # 对参数进行初始化。。
        self.init_weight_bias()

    def forward(self, x):
        x = self.features(x)
        # b * 256 * 6 * 6
        # start_dim     the first dim to flatten
        x = torch.flatten(x, 1)
        # x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

        

    def init_weight_bias(self):
        # use caffe initialization
        for layer in self.features:
            # We initialized the weights in each layer from a zero-mean Gau
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # We initialized the neuron biases in the second, fourth, and fifth convon
        # as well as the fully-connected hidden layers, with the constant 1.
        nn.init.constant_(self.features[3].bias, 0.1)   # from paper 1 to caffe 0.1
        nn.init.constant_(self.features[8].bias, 0.1)   # from paper 1 to caffe 0.1
        nn.init.constant_(self.features[10].bias, 0.1)   # from paper 1 to caffe 0.1
        # 使用caffe版的代码。
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.005)   # caffe set to 0.005
                nn.init.constant_(layer.bias, 0.1)   # from paper 1 to caffe 0.1
        # according to caffe last fc 
        nn.init.normal_(self.classifier[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.classifier[-1].bias, 0)
LOG_DIR='./runs/'

seed = torch.initial_seed()
print('Used seed: {}'.format(seed))

tbwriter = SummaryWriter(log_dir=LOG_DIR)
print('TensorboardX summary writer is created')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_IDS = [0, 1]
NUM_CLASSES = 2
alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
print(alexnet)
print('AlexNet created')
import torchvision.datasets as datasets
from torchvision import transforms
IMAGE_DIM = 227
# 用的ImageNet的mean和std。。
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
TRAIN_IMG_DIR = './kagglecatsanddogs_3367a/PetImages/'
dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transform=preprocess)
print('Dataset created')
BATCH_SIZE = 128
from torch.utils import data
dataloader = data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True
)
print("Dataloader created")
import torch.optim as optim
# The learning rate was initialized at 0.01
# with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005. 
# 5e-4很经典这个数字。
optimizer = optim.Adam(params=alexnet.parameters(), lr=0.01, weight_decay=0.0005)
print('Optimizer created')
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
print('LR Scheduler created')
import os
print('Starting training...')
import torch.nn.functional as F
# update gradient的次数。。也就是batch_size的个数
total_steps = 1
NUM_EPOCHS = 101
CHECKPOINT_DIR = './save_model/'
for epoch in range(NUM_EPOCHS):
    # train
    # validation
    # step()
    lr_scheduler.step()
    # 
    # every batch
    for imgs, classes in dataloader:
        # 全是tensor
        imgs, classes = imgs.to(device), classes.to(device)
        # calculate the loss
        output = alexnet(imgs)
        # TODO: needs debug
        loss = F.cross_entropy(output, classes)

        # 
        # update the parameters
        # every batch 先把梯度清零。
        optimizer.zero_grad()
        loss.backward()
        # step()就是进行一次update。。
        optimizer.step()
        # 
        # 

        # log the information and add to tensorboard
        if total_steps % 10 == 0:
            # torch.no_grad()   一般只进行forward运算。
            with torch.no_grad():
                # dim = 1
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)
                print('Epoch: {}\t Step: {}\t Loss: {:.4f} \t Acc: {}'
                .format(
                    epoch + 1, total_steps, loss.item(), accuracy.item()*1.0/BATCH_SIZE
                ))
                # 每10个batch。记录一次在tensorboard上。
                tbwriter.add_scalar('loss', loss.item(), total_steps)
                tbwriter.add_scalar('accuracy', accuracy.item()*1.0/BATCH_SIZE, total_steps)

        # 每个batch，计算一次。
        total_steps += 1
    # 以epoch为单位进行模型存储。。

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
    state = {
        'epoch': epoch + 1,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'model': alexnet.state_dict(),
        'seed': seed,
    }
    torch.save(state, checkpoint_path)