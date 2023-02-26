import cv2,os
import torch,json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load labels
def load_labels(label_path):
    with open(label_path, 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']

#customed class for data loading # load the data from files
class ReadData(Dataset):
    def __init__(self,root_path):
        super(ReadData, self).__init__()
        self.root_path=root_path
        self.img_path=self.root_path+"\\images"
        self.label_path=self.root_path+"\\labels"
        self.img_names=os.listdir(self.img_path)
        self.label_names=os.listdir(self.label_path)

    def __getitem__(self, item):
        img_path=self.img_names[item]
        img=cv2.imread(self.img_path+"\\"+img_path)
        img = img[:, :, ::-1]  # BGR 2 RGB
        img=img/255
        img=cv2.resize(img,(224,224))
        labels=load_labels(self.label_path+"\\"+self.label_names[item])

        return img,labels
    def __len__(self):
        return len(self.img_names)

#iter for datasets
train_data=ReadData("E:\\opencv\\face_detect\\aug_data\\train")
train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
test_data=ReadData("E:\\opencv\\face_detect\\aug_data\\test")
test_loader=DataLoader(test_data,batch_size=16,shuffle=True)
val_data=ReadData("E:\\opencv\\face_detect\\aug_data\\val")
val_loader=DataLoader(val_data,batch_size=16,shuffle=True)

# display samples
# for img_batch, label_batch in train_loader:
#     img_batch = img_batch.numpy()
#     label_class=label_batch[0]
#     label_coor=label_batch[1]
#     fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
#     for i in range(4):
#         if label_class[0][i]==1:
#             x1=label_coor[0][i].numpy()
#             y1=label_coor[1][i].numpy()
#             x2 = label_coor[2][i].numpy()
#             y2 = label_coor[3][i].numpy()
#             cv2.rectangle(img_batch[i],
#                           tuple(np.multiply((x1,y1), [120, 120]).astype(int)),
#                           tuple(np.multiply((x2,y2), [120, 120]).astype(int)),
#                           (255, 0, 0), 2)
#             ax[i].imshow(img_batch[i])
#     break




class DetectNet(nn.Module):
    def __init__(self):
        super(DetectNet,self).__init__()
        self.vgg16=models.vgg16(pretrained=True)
        self.feature_layer=self.vgg16.features

        self.f1=nn.AdaptiveAvgPool2d((1,1))
        self.class1=nn.Linear(512,400)
        self.class2=nn.Linear(400,1)

        self.f2=nn.AdaptiveAvgPool2d((1,1))
        self.reg1=nn.Linear(512,400)
        self.reg2=nn.Linear(400,4)

    def forward(self,x):
        x=torch.transpose(x,1,3).to(torch.float32)
        x=self.feature_layer(x)

        # Classification Model
        f1=self.f1(x)
        f1=torch.squeeze(f1)
        class1=F.relu(self.class1(f1))
        class2=F.sigmoid(self.class2(class1))

        # Bounding box model
        f2=self.f2(x)
        f2=torch.squeeze(f2)
        regress1=F.relu(self.reg1(f2))
        regress2=F.sigmoid(self.reg2(regress1))

        return class2,regress2





detect_model=DetectNet()
detect_model.to(device)

class DetectLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, yhat):
        delta_coord =torch.sum(torch.square(y_true[:, :2] - yhat[:, :2]))

        h_true = y_true[:, 3] - y_true[:, 1]
        w_true = y_true[:, 2] - y_true[:, 0]

        h_pred = yhat[:, 3] - yhat[:, 1]
        w_pred = yhat[:, 2] - yhat[:, 0]

        delta_size =torch.sum(torch.square(w_true - w_pred)+torch.square(h_true - h_pred))

        return delta_coord+delta_size


def fit(net,batch_train_data,batch_val_data,lr,gamma,epochs,save_period=3):
    #loss function
    class_loss=nn.BCELoss()
    detect_loss=DetectLoss()

    #optim
    opt=torch.optim.SGD(net.parameters(),lr=lr,momentum=gamma)

    #monitor the process
    samples=0
    loss_history=[]

    for epoch in range(0,epochs):
        for batch_idx,(x,y) in enumerate(batch_train_data):
            net.train()
            yhat_class,yhat_regress=net.forward(x.to(device))
            yhat_class, yhat_regress =yhat_class.to(device),yhat_regress.to(device)
            label_class = y[0][0].reshape((-1,1)).to(torch.float32).to(device)
            label_coor=   y[1]
            x1 = label_coor[0].reshape((-1,1))
            y1 = label_coor[1].reshape((-1,1))
            x2 = label_coor[2].reshape((-1,1))
            y2 = label_coor[3].reshape((-1,1))
            coor_true=torch.cat([x1,y1,x2,y2],dim=1).to(torch.float32).to(device)

            closs=class_loss(yhat_class,label_class)
            rloss=detect_loss(coor_true,yhat_regress)
            loss=closs+rloss
            loss.backward()
            #update gradients
            opt.step()
            opt.zero_grad()
            samples += x.shape[0]

            if (batch_idx + 1) % 10== 0 or batch_idx == (len(batch_train_data) - 1):
                # 监督模型进度
                print("train loss:Epoch{}:[{}/{} {: .0f}%], cLoss:{:.2f},rLoss:{:.2f} ".format(
                    epoch + 1
                    , samples
                    , epochs * len(batch_train_data.dataset)
                    , 100 * samples / (epochs * len(batch_train_data.dataset))
                    , closs.data.item(),rloss.data.item()))

        val_loss_epoch = []
        for batch_idx,(x,y) in enumerate(batch_val_data):
            net.eval()
            with torch.no_grad():
                yhat_class, yhat_regress = net.forward(x.to(device))
                yhat_class, yhat_regress = yhat_class.to(device), yhat_regress.to(device)
                label_class = y[0][0].reshape((-1, 1)).to(torch.float32).to(device)
                label_coor = y[1]
                x1 = label_coor[0].reshape((-1, 1))
                y1 = label_coor[1].reshape((-1, 1))
                x2 = label_coor[2].reshape((-1, 1))
                y2 = label_coor[3].reshape((-1, 1))
                coor_true = torch.cat([x1, y1, x2, y2], dim=1).to(torch.float32).to(device)

                closs = class_loss(yhat_class, label_class)
                rloss = detect_loss(coor_true, yhat_regress)
                val_loss = closs + rloss
                val_loss_epoch.append(val_loss.cpu().numpy())

                if (batch_idx + 1) % 10== 0 or batch_idx == (len(batch_val_data) - 1):
                    # 监督模型进度
                    print("val loss:Epoch{}:[{}/{} {: .0f}%], cLoss:{:.2f},rLoss:{:.2f} ".format(
                        epoch + 1
                        , samples
                        , epochs * len(batch_val_data.dataset)
                        , 100 * samples / (epochs * len(batch_val_data.dataset))
                        , closs.data.item(), rloss.data.item()))
        ave_val_loss=sum(val_loss_epoch)/16
        loss_history.append(ave_val_loss)

        save_dir="E:\\opencv\\face_detect\\model_save"
        if (epoch)%save_period==0:
            print('model saved ')
            torch.save(net.state_dict(), os.path.join(save_dir, 'ep %03d -val_loss%.3f.pth' % (epochs,ave_val_loss)))

        if ave_val_loss < min(loss_history):
            print('Save best model to best_epoch_weights.pth')
            torch.save(net.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))




lr = 0.15
gamma = 0.8
epochs = 50
fit(detect_model, train_loader,val_loader,lr, gamma, epochs)







