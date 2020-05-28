from config import opt
from data_handler import *
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import ImgModule, TxtModule, HashModule, LabModule
from utils import calc_map_k


def train(**kwargs):
    opt.parse(kwargs)
    gamma = [0.2,0.5,0.8,1.0,1.3,1.5,1.8,2.0,2.5]
    images, tags, labels = load_data(opt.data_path)
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    y_dim = tags.shape[1]
    label_num = labels.shape[1]
    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')
    for nnnn in range(len(gamma)):
        img_model = ImgModule(opt.bit, pretrain_model)
        txt_model = TxtModule(y_dim, opt.bit)
        hash_model = HashModule(opt.bit)
        label_model = LabModule(label_num)
        if opt.use_gpu:
            img_model = img_model.cuda()
            txt_model = txt_model.cuda()
            hash_model = hash_model.cuda()
            label_model = label_model.cuda()
        train_L = torch.from_numpy(L['train'])
        train_x = torch.from_numpy(X['train'])
        train_y = torch.from_numpy(Y['train'])

        query_L = torch.from_numpy(L['query'])
        query_x = torch.from_numpy(X['query'])
        query_y = torch.from_numpy(Y['query'])

        retrieval_L = torch.from_numpy(L['retrieval'])
        retrieval_x = torch.from_numpy(X['retrieval'])
        retrieval_y = torch.from_numpy(Y['retrieval'])

        num_train = train_x.shape[0]

        F_buffer = torch.randn(num_train, opt.bit)
        G_buffer = torch.randn(num_train, opt.bit)
        X_fea_buffer = torch.randn(num_train, opt.X_fea_nums)
        Y_fea_buffer = torch.randn(num_train,opt.Y_fea_nums)
        X_label_buffer = torch.randn(num_train, label_num)
        Y_label_buffer = torch.randn(num_train, label_num)
        
        Label_buffer = torch.randn(num_train, label_num)
        Label_hash_buffer = torch.randn(num_train, opt.bit)
        Label_label_buffer = torch.randn(num_train, label_num)
        
        if opt.use_gpu:
            train_L = train_L.cuda()
            F_buffer = F_buffer.cuda()
            G_buffer = G_buffer.cuda()
            X_fea_buffer = X_fea_buffer.cuda()
            Y_fea_buffer = Y_fea_buffer.cuda()
            Label_buffer = Label_buffer.cuda()
            X_label_buffer = X_label_buffer.cuda()
            Y_label_buffer =  Y_label_buffer.cuda()
            Label_hash_buffer = Label_hash_buffer.cuda()
            Label_label_buffer = Label_label_buffer.cuda()
        Sim = calc_neighbor(train_L, train_L)
        ###############ddddddd
        B = torch.sign(F_buffer + G_buffer)
        B_buffer = torch.sign(F_buffer + G_buffer)
        batch_size = opt.batch_size

        lr = opt.lr
        optimizer_img = SGD(img_model.parameters(), lr=lr)
        optimizer_txt = SGD(txt_model.parameters(), lr=lr)
        optimizer_hash = SGD(hash_model.parameters(), lr=lr)
        optimizer_label = SGD(label_model.parameters(), lr=lr)

        learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
        result = {
            'loss': [],
            'hash_loss' : [],
            'total_loss' : []
        }

        ones = torch.ones(batch_size, 1)
        ones_ = torch.ones(num_train - batch_size, 1)
        unupdated_size = num_train - batch_size

        max_mapi2t = max_mapt2i = 0.

        for epoch in range(opt.max_epoch):
            # train label net
            for i in tqdm(range(num_train // batch_size)):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)
                sample_L = Variable(train_L[ind, :])
                label = Variable(train_L[ind,:].unsqueeze(1).unsqueeze(-1).type(torch.float))
                if opt.use_gpu:
                    label = label.cuda()
                    sample_L = sample_L.cuda()
                # similar matrix size: (batch_size, num_train)
                S = calc_neighbor(sample_L, train_L)
                label_hash, label_label = label_model(label)  #
                Label_hash_buffer[ind, :] = label_hash.data
                Label_label_buffer[ind, :] = label_label.data
                Label = Variable(train_L)
                Label_B = torch.sign(label_hash)
                Label_H = Variable(Label_hash_buffer) 
                
                theta_l = 1.0 / 2 * torch.matmul(label_hash, Label_H.t())
                logloss_l = -torch.sum(S * theta_l - torch.log(1.0 + torch.exp(theta_l)))
                quantization_l = torch.sum(torch.pow(Label_hash_buffer[ind, :] - Label_B, 2))
                labelloss_l = torch.sum(torch.pow(Label[ind, :].float() - label_label, 2))
                loss_label = logloss_l + opt.beta * quantization_l + opt.alpha * labelloss_l  # + logloss_x_fea
                loss_label /= (batch_size * num_train)

                optimizer_label.zero_grad()
                loss_label.backward()
                optimizer_label.step()
            # train image net
            for i in tqdm(range(num_train // batch_size)):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)
                sample_L = Variable(train_L[ind, :])
                image = Variable(train_x[ind].type(torch.float))
                if opt.use_gpu:
                    image = image.cuda()
                    sample_L = sample_L.cuda()
                # similar matrix size: (batch_size, num_train)
                S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
                image_fea, cur_f, image_label = img_model(image)  # cur_f: (batch_size, bit)
                X_fea_buffer[ind, :] = image_fea.data
                F_buffer[ind, :] = cur_f.data
                X_label_buffer[ind, :] = image_label.data
                G = Variable(G_buffer)
                H_l = Variable(Label_hash_buffer)
                B_x = torch.sign(F_buffer)

                theta_x = 1.0 / 2 * torch.matmul(cur_f, H_l.t())
                logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
                quantization_xh = torch.sum(torch.pow(B_buffer[ind, :] - cur_f, 2))
                quantization_xb = torch.sum(torch.pow(B_x[ind, :]- cur_f, 2))
                labelloss_x = torch.sum(torch.pow(train_L[ind, :].float() - image_label,2))
                loss_x = logloss_x + opt.beta * quantization_xh + opt.alpha * labelloss_x + gamma[nnnn] * quantization_xb# + logloss_x_fea
                loss_x /= (batch_size * num_train)

                optimizer_img.zero_grad()
                loss_x.backward()
                optimizer_img.step()
            # train txt net
            for i in tqdm(range(num_train // batch_size)):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)
                sample_L = Variable(train_L[ind, :])
                text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
                text = Variable(text)
                if opt.use_gpu:
                    text = text.cuda()
                    sample_L = sample_L.cuda()
                # similar matrix size: (batch_size, num_train)
                S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
                txt_fea, cur_g, txt_label = txt_model(text)  # cur_f: (batch_size, bit)
                Y_fea_buffer[ind, :] = txt_fea.data
                G_buffer[ind, :] = cur_g.data
                Y_label_buffer[ind, :] = txt_label.data
                F = Variable(F_buffer)
                H_l = Variable(Label_hash_buffer)
                B_y = torch.sign(F)
                # calculate loss
                # theta_y: (batch_size, num_train)
                theta_y = 1.0 / 2 * torch.matmul(cur_g, H_l.t())
                logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
                quantization_yh = torch.sum(torch.pow(B_buffer[ind, :] - cur_g, 2))
                quantization_yb = torch.sum(torch.pow(B_y[ind, :] - cur_g, 2))
                labelloss_y = torch.sum(torch.pow(train_L[ind, :].float() - txt_label, 2))
                loss_y = logloss_y + opt.beta * quantization_yh + opt.alpha * labelloss_y + gamma[nnnn] * quantization_yb# + logloss_y_fea
                loss_y /= (num_train * batch_size)
            
                optimizer_txt.zero_grad()
                loss_y.backward()
                optimizer_txt.step()

            #train hash net
            for i in tqdm(range(num_train // batch_size)):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)
                
                sample_L = Variable(train_L[ind, :])
                #W = norm(X_fea_buffer[ind, :], Y_fea_buffer[ind, :])
                #fea = 1.0 / 2 * (torch.matmul(W, X_fea_buffer[ind, :]) + torch.matmul(W, Y_fea_buffer[ind, :]))
                fea = torch.cat([X_fea_buffer[ind, :], Y_fea_buffer[ind, :]], dim=1)
                fea = Variable(fea)
                if opt.use_gpu:
                    fea = fea.cuda()
                    sample_L = sample_L.cuda()
                S = calc_neighbor(sample_L, train_L)
                A = caculateAdj(sample_L, sample_L)
                cur_B, label_hash = hash_model(fea, A)
                B_buffer[ind, :] = cur_B.data
                #caculate loss
                B = Variable(torch.sign(B_buffer))
                theta_hash = 1.0 / 2 * torch.matmul(cur_B, B_buffer.t())
                logloss_hash = -torch.sum(S * theta_hash - torch.log(1.0 + torch.exp(theta_hash)))
                label_loss = torch.sum(torch.pow(train_L[ind, :].float() - label_hash, 2))
                hashloss = torch.sum(torch.pow(B[ind, :] - cur_B, 2))
                loss_hash = logloss_hash + opt.alpha * label_loss + opt.beta * hashloss

                optimizer_hash.zero_grad()
                loss_hash.backward()
                optimizer_hash.step()
            # train image net
            for i in tqdm(range(num_train // batch_size)):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)
                sample_L = Variable(train_L[ind, :])
                image = Variable(train_x[ind].type(torch.float))
                if opt.use_gpu:
                    image = image.cuda()
                    sample_L = sample_L.cuda()
                # similar matrix size: (batch_size, num_train)
                S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
                image_fea, cur_f, image_label = img_model(image)  # cur_f: (batch_size, bit)
                X_fea_buffer[ind, :] = image_fea.data
                F_buffer[ind, :] = cur_f.data
                X_label_buffer[ind, :] = image_label.data
                G = Variable(G_buffer)
                H_l = Variable(Label_hash_buffer)
                B_x = torch.sign(F_buffer)

                theta_x = 1.0 / 2 * torch.matmul(cur_f, H_l.t())
                logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
                quantization_xh = torch.sum(torch.pow(B_buffer[ind, :] - cur_f, 2))
                quantization_xb = torch.sum(torch.pow(B_x[ind, :] - cur_f, 2))
                labelloss_x = torch.sum(torch.pow(train_L[ind, :].float() - image_label, 2))
                loss_x = logloss_x + gamma[nnnn] * quantization_xh + opt.alpha * labelloss_x + opt.beta * quantization_xb  # + logloss_x_fea
                loss_x /= (batch_size * num_train)

                optimizer_img.zero_grad()
                loss_x.backward()
                optimizer_img.step()
            # train txt net
            for i in tqdm(range(num_train // batch_size)):
                index = np.random.permutation(num_train)
                ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(num_train), ind)
                sample_L = Variable(train_L[ind, :])
                text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
                text = Variable(text)
                if opt.use_gpu:
                    text = text.cuda()
                    sample_L = sample_L.cuda()
                # similar matrix size: (batch_size, num_train)
                S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
                txt_fea, cur_g, txt_label = txt_model(text)  # cur_f: (batch_size, bit)
                Y_fea_buffer[ind, :] = txt_fea.data
                G_buffer[ind, :] = cur_g.data
                Y_label_buffer[ind, :] = txt_label.data
                F = Variable(F_buffer)
                H_l = Variable(Label_hash_buffer)
                B_y = torch.sign(F)
                # calculate loss
                # theta_y: (batch_size, num_train)
                theta_y = 1.0 / 2 * torch.matmul(cur_g, H_l.t())
                logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
                quantization_yh = torch.sum(torch.pow(B_buffer[ind, :] - cur_g, 2))
                quantization_yb = torch.sum(torch.pow(B_y[ind, :] - cur_g, 2))
                labelloss_y = torch.sum(torch.pow(train_L[ind, :].float() - txt_label, 2))
                loss_y = logloss_y + gamma[nnnn] * quantization_yh + opt.alpha * labelloss_y + opt.beta * quantization_yb  # + logloss_y_fea
                loss_y /= (num_train * batch_size)

                optimizer_txt.zero_grad()
                loss_y.backward()
                optimizer_txt.step()

            # calculate total loss
            loss, hash_loss, total_loss = calc_loss(B, F, G, Variable(Sim), opt.alpha, opt.beta,Label_buffer, train_L, X_label_buffer,Y_label_buffer)

            print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
            print('...epoch: %3d, hash_loss: %3.3f, lr: %f' % (epoch + 1, hash_loss.data, lr))
            print('...epoch: %3d, total_loss: %3.3f, lr: %f' % (epoch + 1, total_loss.data, lr))
            result['loss'].append(float(loss.data))
            result['hash_loss'].append(float(hash_loss.data))
            result['total_loss'].append(float(total_loss.data))

            if opt.valid:
                mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                       query_L, retrieval_L)
                print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
                if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                    max_mapi2t = mapi2t
                    max_mapt2i = mapt2i
                    img_model.save(img_model.module_name + '.pth')
                    txt_model.save(txt_model.module_name + '.pth')
                    hash_model.save(hash_model.module_name+'.pth')

            lr = learning_rate[epoch + 1]

            # set learning rate
            for param in optimizer_img.param_groups:
                param['lr'] = lr
            for param in optimizer_txt.param_groups:
                param['lr'] = lr

        print('...training procedure finish')
        if opt.valid:
            print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
            result['mapi2t'] = max_mapi2t
            result['mapt2i'] = max_mapt2i
        else:
            mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                                   query_L, retrieval_L)
            print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
            result['mapi2t'] = mapi2t
            result['mapt2i'] = mapt2i

        write_result(result,nnnn)


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def test(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.load_img_path:
        img_model.load(opt.load_img_path)

    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
    print('-----------------------')
    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    if opt.use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim

def caculateAdj(label1, label2):
    if opt.use_gpu:
        A = (label1.matmul(label2.transpose(0, 1))).type(torch.cuda.FloatTensor)
    else:
        A = (label1.matmul(label2.transpose(0, 1))).type(torch.FloatTensor)
    return A

def calc_loss(B, F, G, Sim, gamma, eta, Label, L, label_x, label_y):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term1 = term1 * 2
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(L.float() - label_x, 2) + torch.pow(L.float() - label_y, 2))
    loss = term1 + gamma * term2 + eta * term3

    theta1 = torch.matmul(B, B.transpose(0,1))/2
    term4 = torch.sum(torch.log(1+torch.exp(theta1)) - Sim * theta1)
    label_loss = torch.sum(torch.pow(L.float() - Label, 2))
    hash_loss = term4 + label_loss
    
    total_loss = loss + hash_loss
    return loss, hash_loss, total_loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        _, cur_f, _ = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        _, cur_g, _ = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def write_result(result,num):
    import os
    with open(os.path.join(opt.result_dir, 'resultgamma'+str(num)+'.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')

def norm(m1, m2):
    result = torch.matmul(m1, m2.t())
    s = torch.mul(result, result)
    s = torch.sum(s, 1)
    s = torch.sqrt(s)
    s = s.unsqueeze(1)
    result = result / s
    return result 
    
def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    import fire
    fire.Fire(train())
