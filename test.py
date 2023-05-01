import argparse
import os
import torch
import torch.nn as nn

def test_model(model, PATH, test_dataloader):
    model = torch.load(PATH)
    #---------------------- 모델 시험 ---------------------#
    model.eval()
    total =0.0
    accuracy =0.0
    for batch_idx, data in enumerate(test_dataloader):
        x_test, y_test = data
            
        t_hat = model(x_test)
        _, predicted = torch.max(t_hat.data, 1)
        total += y_test.size(0)
        accuracy += (predicted == y_test).sum().item()
    accuracy = (accuracy / total)
    #------------------------------------------------------#
    print(accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type = str,
        default = 'dnn'
    )
    parser.add_argument(
        '-p',
        '--path',
        default = './data/'
    )
    parser.add_argument(
        '-c',
        '--cuda',
        type = int,
        default = 0
    )

    args = parser.parse_args()

    print("---------------------------------------------")
    print("------------------GPU Check------------------")
    print("쿠다 가능 :{}".format(torch.cuda.is_available()))
    print("현재 디바이스 :{}".format(torch.cuda.current_device()))
    print("디바이스 갯수 :{}".format(torch.cuda.device_count()))
    
    for idx in range(0, torch.cuda.device_count()):
        print("디바이스 : {}".format(torch.cuda.device(idx)))
        print("디바이스 이름 : {}".format(torch.cuda.get_device_name(idx)))
    print("---------------------------------------------")

    device = torch.device('cuda:1' if args.cuda == 1 else 'cuda:0')

    from trainmodel.dnn import KAMP_DNN
    model = KAMP_DNN().to(device)
    PATH = './model/dnn_model.pt'
    
    if args.model == 'cnn': 
        PATH = './model/cnn_model.pt'
        from trainmodel.cnn import KAMP_CNN
        model = KAMP_CNN(3).to(device)

    elif args.model == 'rnn': 
        PATH = './model/rnn_model.pt'
        from trainmodel.rnn import KAMP_RNN
        model = KAMP_RNN().to(device)
    

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    import numpy as np

    test = np.load('./data/x_valid.npy')
    test_label = np.load('./data/y_valid.npy')

    x_test = torch.from_numpy(test).float().to(device)
    y_test = torch.from_numpy(test_label).float().T[0].to(device)

    test = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test, batch_size = len(x_test), shuffle=False)
    
    test_model(model, PATH, test_dataloader)

    