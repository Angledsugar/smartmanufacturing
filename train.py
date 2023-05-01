import argparse
import os
import torch
import torch.nn as nn

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "0,2"

def train_model(model, criterion, optimizer, num_epochs, train_dataloader, valid_dataloader, PATH):
    loss_values = []
    loss_values_v = []
    check = 0
    accuracy_past = 0

    for epoch in range(1, num_epochs +1):
    #---------------------- 모델 학습 ---------------------#
        model.train()
        batch_number =0
        running_loss =0.0
    
        for batch_idx, samples in enumerate(train_dataloader):
            x_train, y_train = samples
            # 변수 초기화
            optimizer.zero_grad()
            y_hat = model.forward(x_train)
            loss = criterion(y_hat,y_train.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_number +=1
        
        loss_values.append(running_loss / batch_number)
        #---------------------- 모델 검증 ---------------------#
        model.eval()
        accuracy =0.0
        total =0.0

        for batch_idx, data in enumerate(valid_dataloader):
            x_valid, y_valid = data
            
            v_hat = model.forward(x_valid)
            v_loss = criterion(v_hat,y_valid.long())
            _, predicted = torch.max(v_hat.data, 1)
            total += y_valid.size(0)
            accuracy += (predicted == y_valid).sum().item()
        
        loss_values_v.append(loss.item())
        accuracy = (accuracy / total)
        
        #----------------Check for early stopping---------------#
        if epoch % 1 == 0:
            print('[Epoch {}/{}] [Train_Loss: {:.6f} /Valid_Loss: {:.6f}]'.format(epoch, num_epochs, loss.item(),v_loss.item()))
            print('[Epoch {}/{}] [Accuracy : {:.6f}]'.format(epoch, num_epochs, accuracy))
        
        if accuracy_past > accuracy:
            check += 1
        else:
            check = 0
            accuracy_past = accuracy
        
        if check > 50:
            print('This is time to do early stopping')
    
    torch.save(model, PATH + args.trainmodel + '_model.pt')
    
    return loss_values, loss_values_v

if __name__ == '__main__':
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device:', device)  # 출력결과: cuda 
    # print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (2, 3 두개 사용하므로)
    # print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (2, 3 중 앞의 GPU #2 의미)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--trainmodel',
        type = str,
        default='dnn'
    )
    parser.add_argument(
        '-c',
        '--criterion',
        default = nn.CrossEntropyLoss()
    )
    # parser.add_argument(
    #     '-o',
    #     '--optimizer',
    #     default = torch.optim.Adam(DNN_model.parameters())
    # )
    parser.add_argument(
        '-n',
        '--num_epochs',
        type = int,
        default = 1000
    )
    parser.add_argument(
        '-i',
        '--input_dir',
        type = str,
        default='./data/'
    )
    parser.add_argument(
        '-o',
        '--model_output',
        type = str,
        default='./model/'
    )
    parser.add_argument(
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

    ### model select
    from trainmodel.dnn import KAMP_DNN
    model = KAMP_DNN().to(device)
    
    if args.trainmodel == 'cnn':
        from trainmodel.cnn import KAMP_CNN
        model = KAMP_CNN(3).to(device)

    if args.trainmodel == 'rnn':
        from trainmodel.rnn import KAMP_RNN
        model = KAMP_RNN().to(device)

    print(model)

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    import numpy as np
    
    train = np.load('./data/x_train.npy')
    train_label = np.load('./data/y_train.npy')

    valid = np.load('./data/x_test.npy')
    valid_label = np.load('./data/y_test.npy')
    
    test = np.load('./data/x_valid.npy')
    test_label = np.load('./data/y_valid.npy')

    x_train = torch.from_numpy(train).float().to(device)
    y_train = torch.from_numpy(train_label).float().ravel().to(device)
    # print(x_train)
    # print(y_train)

    x_valid = torch.from_numpy(valid).float().to(device)
    y_valid = torch.from_numpy(valid_label).float().T[0].to(device)
    # print(x_valid)
    # print(y_valid)

    x_test = torch.from_numpy(test).float().to(device)
    y_test = torch.from_numpy(test_label).float().T[0].to(device)
    # print(y_test)

    train = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train, batch_size =5000, shuffle=True)
    valid = TensorDataset(x_valid, y_valid)
    valid_dataloader = DataLoader(valid, batch_size =len(x_valid), shuffle=False)
    test = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test, batch_size =len(x_valid), shuffle=False)

    a, b = train_model(model, args.criterion, torch.optim.Adam(model.parameters()), args.num_epochs, 
                       train_dataloader, 
                       valid_dataloader,
                       args.model_output)
  