import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("D:/user/Desktop/(Karl) Lab_rotation/Malaria_segmentation_model/Unet/results/weight_map")

with open("(weight map)for_plotting_training_results.TXT", "r") as f:
    loss_list = []
    acc_list = []
    IOU_list = []
    val_loss_list = []
    val_acc_list = []
    val_IOU_list = []
    for line in f.readlines():
        loss = line.split("-")[0]
        acc = line.split("-")[1]
        IOU = line.split("-")[2]
        val_loss = line.split("-")[3]
        val_acc = line.split("-")[4]
        val_IOU = line.split("-")[5]
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
    plt.title("Training accuracy (with weight map)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    #plt.ylim(0.65, 1.00)
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.title("Training loss (with weight map)")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    #plt.ylim(0.25, 0.60)
    plt.legend(["Train", "Val"], loc="upper right")
    plt.show()
    
    plt.plot(IOU_list, label="Train")
    plt.plot(val_IOU_list, label="Val")
    plt.title("Training IOU core (with weight map)")
    plt.ylabel("IOU score")
    plt.xlabel("Epochs")
    #plt.ylim(0.260, 0.330)
    plt.legend(loc="upper left")
    plt.show()