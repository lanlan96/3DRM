import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_curve(title, train_data, test_data, save, log_dir):
    plt.figure(title)
    plt.gcf().clear()
    plt.plot(train_data, label='train')
    plt.plot(test_data, label='valid')
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, title+".png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

def draw_curve_train(title, train_data, save, log_dir):
    plt.figure(title)
    plt.gcf().clear()
    plt.plot(train_data, label='train')
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, title+".png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

def plot_loss(loss, test_loss, save=True, log_dir=None):
    # train_loss_history.append([mloss, macc_sin, macc_cls, mcls_loss, mreg_loss, msin_loss])  6
    # test_loss_history.append([mloss_test, macc_sin_test, macc_cls_test, mcls_test_loss, mreg_test_loss, msin_test_loss])  6
    train_loss = np.array(loss)
    test_loss = np.array(test_loss)

    draw_curve("loss", train_loss[:, 0], test_loss[:, 0], save, log_dir)
    draw_curve("acc", train_loss[:, 1], test_loss[:, 1], save, log_dir)
    # draw_curve("acc_cls", train_loss[:, 2], test_loss[:, 2], save, log_dir)

    if train_loss.shape[-1] > 2:
        draw_curve("cls_loss", train_loss[:, 2], test_loss[:, 2], save, log_dir)
        draw_curve("reg_loss", train_loss[:, 3], test_loss[:, 3], save, log_dir)


    if train_loss.shape[-1] > 4:
        draw_curve("rn_loss", train_loss[:, 4], test_loss[:, 4], save, log_dir)
        draw_curve("rn_acc", train_loss[:, 5], test_loss[:, 5], save, log_dir)
    if train_loss.shape[-1] > 6:
        draw_curve("rn0_loss", train_loss[:, 6], test_loss[:, 6], save, log_dir)
        draw_curve("rn1_loss", train_loss[:, 7], test_loss[:, 7], save, log_dir)
        draw_curve("rn2_loss", train_loss[:, 8], test_loss[:, 8], save, log_dir)
        draw_curve("rn3_loss", train_loss[:, 9], test_loss[:, 9], save, log_dir)
    if train_loss.shape[-1] > 10:
        draw_curve("rn0_acc", train_loss[:, 10], test_loss[:, 10], save, log_dir)
        draw_curve("rn1_acc", train_loss[:, 11], test_loss[:, 11], save, log_dir)
        draw_curve("rn2_acc", train_loss[:, 12], test_loss[:, 12], save, log_dir)
        draw_curve("rn3_acc", train_loss[:, 13], test_loss[:, 13], save, log_dir)

def plot_loss_train_single_rn(loss, save=True, log_dir=None):
    # train_loss_history.append([mloss, macc_sin, macc_cls, mcls_loss, mreg_loss, msin_loss])  6
    # test_loss_history.append([mloss_test, macc_sin_test, macc_cls_test, mcls_test_loss, mreg_test_loss, msin_test_loss])  6
    train_loss = np.array(loss)

    draw_curve_train("loss", train_loss[:, 0], save, log_dir)
    draw_curve_train("acc", train_loss[:, 1], save, log_dir)
    # draw_curve("acc_cls", train_loss[:, 2], test_loss[:, 2], save, log_dir)

    if train_loss.shape[-1] > 2:
        draw_curve_train("cls_loss", train_loss[:, 2], save, log_dir)
        draw_curve_train("reg_loss", train_loss[:, 3], save, log_dir)


    if train_loss.shape[-1] > 4:
        draw_curve_train("rn_loss", train_loss[:, 4], save, log_dir)
        draw_curve_train("rn_acc", train_loss[:, 5], save, log_dir)



def plot_loss_train(loss, save=True, log_dir=None):
    # train_loss_history.append([mloss, macc_sin, macc_cls, mcls_loss, mreg_loss, msin_loss])  6
    # test_loss_history.append([mloss_test, macc_sin_test, macc_cls_test, mcls_test_loss, mreg_test_loss, msin_test_loss])  6
    train_loss = np.array(loss)

    draw_curve_train("loss", train_loss[:, 0], save, log_dir)
    draw_curve_train("acc", train_loss[:, 1], save, log_dir)
    # draw_curve("acc_cls", train_loss[:, 2], test_loss[:, 2], save, log_dir)

    if train_loss.shape[-1] > 2:
        draw_curve_train("cls_loss", train_loss[:, 2], save, log_dir)
        draw_curve_train("reg_loss", train_loss[:, 3], save, log_dir)


    if train_loss.shape[-1] > 4:
        draw_curve_train("rn_loss", train_loss[:, 4], save, log_dir)
        draw_curve_train("rn_acc", train_loss[:, 5], save, log_dir)
    if train_loss.shape[-1] > 6:
        draw_curve_train("rn0_loss", train_loss[:, 6], save, log_dir)
        draw_curve_train("rn1_loss", train_loss[:, 7],  save, log_dir)
        draw_curve_train("rn2_loss", train_loss[:, 8],  save, log_dir)
        draw_curve_train("rn3_loss", train_loss[:, 9],  save, log_dir)
    if train_loss.shape[-1] > 10:
        draw_curve_train("rn0_acc", train_loss[:, 10],  save, log_dir)
        draw_curve_train("rn1_acc", train_loss[:, 11],  save, log_dir)
        draw_curve_train("rn2_acc", train_loss[:, 12],  save, log_dir)
        draw_curve_train("rn3_acc", train_loss[:, 13],  save, log_dir)

def plot_loss_new(loss, test_loss, save=True, log_dir=None):
    # train_loss_history.append([mloss, macc_sin, macc_cls, mcls_loss, mreg_loss, msin_loss])  6
    # test_loss_history.append([mloss_test, macc_sin_test, macc_cls_test, mcls_test_loss, mreg_test_loss, msin_test_loss])  6
    train_loss = np.array(loss)
    test_loss = np.array(test_loss)

    draw_curve("loss", train_loss[:, 0], test_loss[:, 0], save, log_dir)
    draw_curve("acc_sin", train_loss[:, 1], test_loss[:, 1], save, log_dir)
    draw_curve("acc_cls", train_loss[:, 2], test_loss[:, 2], save, log_dir)

    if train_loss.shape[-1] > 3:
        draw_curve("cls_loss", train_loss[:, 3], test_loss[:, 3], save, log_dir)
        draw_curve("reg_loss", train_loss[:, 4], test_loss[:, 4], save, log_dir)
        draw_curve("sin_loss", train_loss[:, 5], test_loss[:, 5], save, log_dir)

    # if train_loss.shape[-1] > 4:
    #     draw_curve("rn_loss", train_loss[:, 4], test_loss[:, 4], save, log_dir)
    #     draw_curve("rn_acc", train_loss[:, 5], test_loss[:, 5], save, log_dir)
def plot_loss_selected(loss, test_loss, save=True, log_dir=None):
    # train_loss_history.append([mloss, macc_sin, macc_cls, mcls_loss, mreg_loss, msin_loss])  6
    # test_loss_history.append([mloss_test, macc_sin_test, macc_cls_test, mcls_test_loss, mreg_test_loss, msin_test_loss])  6
    train_loss = np.array(loss)
    test_loss = np.array(test_loss)

    draw_curve("loss", train_loss[:, 0], test_loss[:, 0], save, log_dir)
    draw_curve("acc_cls", train_loss[:, 1], test_loss[:, 1], save, log_dir)
    draw_curve("cls_loss", train_loss[:, 2], test_loss[:, 2], save, log_dir)
    draw_curve("reg_loss", train_loss[:, 3], test_loss[:, 3], save, log_dir)

def plot_loss_scannet_stage1(loss, test_loss, save=True, log_dir=None):
    train_loss = np.array(loss)
    test_loss = np.array(test_loss)

    draw_curve("loss", train_loss[:, 0], test_loss[:, 0], save, log_dir)
    draw_curve("acc_single", train_loss[:, 1], test_loss[:, 1], save, log_dir)

