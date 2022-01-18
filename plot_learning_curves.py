import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("path/to/dir")

with open("example_training_result.txt", "r") as f:
    loss_list = []
    acc_list = []
    IOU_list = []
    val_loss_list = []
    val_acc_list = []
    val_IOU_list = []
    for line in f.readlines():
        line = line.strip()
        spli = line.split("-")
        loss = spli[0]
        acc = spli[1]
        IOU = spli[2]
        val_loss = spli[3]
        val_acc = spli[4]
        val_IOU = spli[5]
        loss_list.append(loss)
        acc_list.append(acc)
        IOU_list.append(IOU)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_IOU_list.append(val_IOU)
    
    acc_list = np.array(acc_list).astype(float)
    val_acc_list = np.array(val_acc_list).astype(float)
    loss_list = np.array(loss_list).astype(float)
    val_loss_list = np.array(val_loss_list).astype(float)
    IOU_list = np.array(IOU_list).astype(float)
    val_IOU_list = np.array(val_IOU_list).astype(float)
    
    plt.plot(acc_list, label="Train")
    plt.plot(val_acc_list, label="Val")
    plt.title("Training accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    #plt.ylim(0.65, 1.00)
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    #plt.ylim(0.25, 0.60)
    plt.legend(["Train", "Val"], loc="upper right")
    plt.show()
    
    plt.plot(IOU_list, label="Train")
    plt.plot(val_IOU_list, label="Val")
    plt.title("Training IOU core")
    plt.ylabel("IOU score")
    plt.xlabel("Epochs")
    #plt.ylim(0.260, 0.330)
    plt.legend(loc="upper left")
    plt.show()
    
