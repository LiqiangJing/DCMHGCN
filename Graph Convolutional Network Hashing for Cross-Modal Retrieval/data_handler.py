import h5py
import scipy.io as scio


def load_data(path):
    file = h5py.File(path)
    images = file['images'][:].astype('float')#(20015, 3, 224, 224)
    labels = file['LAll'][:]
    tags = file['YAll'][:]

    file.close()
    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)


if __name__ == '__main__':
    '''
    a = {'s': [12, 33, 44],
         's': 0.111}
    from os import open
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)
    '''
    datapath = '/home/jingliqiang/PycharmProjects/DCMH_pytorch/data/FLICKR-25K.mat'
    file = h5py.File(datapath)
    labels = file['LAll'][:]
    tags = file['YAll'][:]
    print(tags.shape)
    print(labels.shape[1])
