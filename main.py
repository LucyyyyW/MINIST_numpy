# from msilib.schema import Class
from dataclasses import dataclass
from webbrowser import get
import numpy as np
from data import ImageLoader
from tools import draw_loss, load_model, save_model
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Numpy Classifier')
    ### input path
    parser.add_argument('--train_img_path', default='/Users/wsh/Downloads/train-images.idx3-ubyte', type=str)
    parser.add_argument('--train_lab_path', default='/Users/wsh/Downloads/train-labels.idx1-ubyte', type=str)
    parser.add_argument('--test_img_path', default='/Users/wsh/Downloads/t10k-images.idx3-ubyte', type=str)
    parser.add_argument('--test_lab_path', default='/Users/wsh/Downloads/t10k-labels.idx1-ubyte', type=str)
    ### params for network
    parser.add_argument('--iterations_num', default=15, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay_rate', default=0.01, type=float) #using L2 Norm
    parser.add_argument('--in_cn', default=28*28, type=int)
    parser.add_argument('--out_cn', default=10, type=int)
    parser.add_argument('--hid_cn', default=1024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--schedule_rate', default=0.5, type=float)
    parser.add_argument('--schedule_step', default=100, type=float)
    ### checkponits and output
    parser.add_argument('--record_per', default=2, type=int)
    parser.add_argument('--ck_per', default=5, type=int)
    parser.add_argument('--train_output', default='/Users/wsh/Documents/grade_1/DeepLearning/hw1/train_loss', type=str)
    parser.add_argument('--test_output', default='/Users/wsh/Documents/grade_1/DeepLearning/hw1/test_loss', type=str)
    parser.add_argument('--acc_output', default='/Users/wsh/Documents/grade_1/DeepLearning/hw1/test_acc', type=str)
    parser.add_argument('--model_output', default='/Users/wsh/Documents/grade_1/DeepLearning/hw1/model', type=str)
    parser.add_argument('--resume', default='/Users/wsh/Documents/grade_1/DeepLearning/hw1/model.npy', type=str)

    parser.add_argument('--state', default='test', type=str)
    args = parser.parse_args()
    return args
    
    
class build_net():
    def __init__(self, args) -> None:
        self.in_cn = args.in_cn
        self.out_cn = args.out_cn
        self.hid_cn = args.hid_cn
        self.batch = args.batch_size
        self.lr = args.lr
        self.decay = args.decay_rate
        self.rate = args.schedule_rate
        self.schedule_step = args.schedule_step
        self.params = self.build_params()
    
    def resume(self, checkpoint):
        self.lr = checkpoint['lr']
        self.decay = checkpoint['decay']
        assert self.hid_cn == checkpoint['hid_cn'], 'please check model structure'
        # self.hid_cn = checkpoint['hid_cn']
        self.params = checkpoint['params']
    
    def build_params(self):
        w_1 = np.random.randn(self.in_cn, self.hid_cn) ## (in_cn, hid_cn)
        w_2 = np.random.randn(self.hid_cn, self.out_cn)
        b_1 = np.random.randn(self.hid_cn)
        b_2 = np.random.randn(self.out_cn)
        params = {
            'layer1_w': w_1,
            'layer2_w': w_2,
            'layer1_b': b_1,
            'layer2_b': b_2,
        }
        return params
    
    def sigmoid(self,x):
        return 1./ (1+ np.exp(-x))

    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x),1,keepdims=True) 
    
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        '''
        x: (batch_size, in_cn)
        '''
        # print(self.params['layer1_b'].shape)
        layer1_z = np.dot(x, self.params['layer1_w']) + self.params['layer1_b']
        layer1 = self.relu(layer1_z) # (4,100) (100,1440) -> (4,1440)
        layer2_z = np.dot(layer1, self.params['layer2_w']) + self.params['layer2_b'] # (4,1440)(1440,10) -> (4,10)
        layer2 = self.relu(layer2_z)
        L2 = np.sum(np.square(layer1)) + np.sum(np.square(layer2))
        value_dict = {
            'name':['layer1', 'layer2'],
            'x':x,
            'layer1_z': layer1_z,
            'layer1': layer1,
            'layer2': layer2_z,
            'layer2_z': layer2
        }
        return value_dict, L2
           
    #adopt cross entropy as loss function
    def loss(self, y, y_hat, L2):
        # print(y,y_hat)
        y_shift = y_hat - np.max(y_hat, axis=-1, keepdims=True)
        y_exp = np.exp(y_shift)
        y_probability = y_exp / np.sum(y_exp, axis=-1,keepdims=True)
        CE = np.mean(np.sum(-y * np.log(y_probability+1e-8),-1))
        delta = y_probability - y
        return CE+self.decay*L2, delta #(4,10)

    #calculate delta for one layer
    def update_delta(self, value, delta_old, key):
        # print(value.shape)
        grad_z = np.sum(value * (1-value),0, keepdims=True) / self.batch
        delta_new = np.dot(delta_old* grad_z,self.params[key+'_w'].T)
        return delta_new
    
    
    def get_one_grad(self, next_delta, weight, z):
        N = z.shape[0]
        delta = np.dot(next_delta, weight.T) #(4,10) (10,2048) -> (4,2048)
        grad_w = np.dot(z.T, next_delta)
        grad_b = np.sum(next_delta, 0)
        return grad_w / N, grad_b / N, delta
    
    def activegrade(self, delta, z):
        return np.where(np.greater(z,0), delta, 0)
        

    #calculate grad for params
    def get_grad(self, value_dict, delta):
        grad_dict = {}
        ls = value_dict['name'][::-1]
        grad_w2, grad_b2, delta_2 = self.get_one_grad(delta,self.params['layer2_w'], value_dict['layer1'])
        delta_2 = self.activegrade(delta_2, value_dict['layer1_z'])
        grad_w1, grad_b1, delta_1 = self.get_one_grad(delta_2,self.params['layer1_w'], value_dict['x'])
        grad_dict['layer1_w'] = grad_w1
        grad_dict['layer1_b'] = grad_b1
        grad_dict['layer2_w'] = grad_w2
        grad_dict['layer2_b'] = grad_b2
        return grad_dict

    def SGD_update(self, params, grads):
        for key in params.keys():
            if 'w' in key:
                params[key] = (1- self.lr*self.decay) * params[key] - self.lr *grads[key]
            else:
                params[key] -= self.lr * grads[key]
        # for key in params.keys():
        #     # print(grads[key].shape)
        #     params[key] -= self.lr * grads[key]
        return params
    
    def lr_schedule(self, iteration):
        if iteration % self.schedule_step ==0:
            self.lr = self.lr*self.rate
        return
    
    
    def accuray(self, y, y_hat):
        y = np.argmax(y,1)
        y_hat = np.argmax(y_hat,1)
        interact = np.where((y - y_hat)==0, 1, 0)
        # print(y, y_hat, interact)
        acc = np.sum(interact) / self.batch
        return acc
    
    #training for one iteration
    def update(self, x,y,iteration):
        value_dict, L2 = self.forward(x)
        # print(value_dict['layer2'].shape)
        ce, delta = self.loss(y, value_dict['layer2'], L2)
        acc = self.accuray(y, value_dict['layer2'])
        grad_dict = self.get_grad(value_dict, delta)
        self.params = self.SGD_update(self.params, grad_dict)
        self.lr_schedule(iteration)
        return ce, acc
    
    def test_one(self, x,y):
        value_dict, L2 = self.forward(x)
        ce, _ = self.loss(y, value_dict['layer2'], L2)
        acc = self.accuray(y, value_dict['layer2'])
        return ce, acc
    
    def test(self,testset):
        acc_ls, loss_ls = [], []
        testiter = iter(testset)
        for j in range(10000):
            try:
                img_test, lab_test = next(testiter)
            except:
                testiter = iter(testset)
                img_test, lab_test = next(testiter)
            loss_test, acc_test = Classifer.test_one(img_test, lab_test)
            acc_ls.append(acc_test)
            loss_ls.append(loss_test)
        return np.mean(loss_ls), np.mean(acc_ls)

    def get_params(self):
        checkpoint = {
            'lr': self.lr,
            'decay': self.decay,
            'hid_cn': self.hid_cn,
            'params': self.params
        }
        return checkpoint


if __name__ == '__main__':
    args = get_args()
    Classifer = build_net(args)
    checkpoint = {}

    if args.resume:
        last_checkpoint = load_model(args.resume)
        Classifer.resume(last_checkpoint)
        
    if args.state == 'train':
        trainset = ImageLoader(args.train_img_path, args.train_lab_path, args.batch_size)
        testset = ImageLoader(args.test_img_path, args.test_lab_path, args.batch_size)
        trainiter = iter(trainset)
        testiter = iter(testset)
        loss_train_list = []
        loss_test_list = []
        acc_test_list = []
        acc_train_list = []
        index_list = []
        for i in range(args.iterations_num):
            try:
                img_train, lab_train = next(trainiter)
            except:
                trainiter = iter(trainset)
                img_train, lab_train = next(trainiter)
            # img_test, lab_test = next(testiter)
            # print(img_train.shape, lab_train.shape)
            img_train = img_train
            loss_train, acc_train = Classifer.update(img_train, lab_train, i)
            if i % args.ck_per ==0:
                checkpoint['%s'%i] = Classifer.get_params()
            if i % args.record_per==0:
                # print('iteration:%s / %s'%(i,1000))
                test_loss, test_acc = Classifer.test(testset)
                loss_train_list.append(loss_train)
                loss_test_list.append(test_loss)
                acc_train_list.append(acc_train)
                acc_test_list.append(test_acc)
                index_list.append(i)
                print('iteration:%s / %s, acc_train: %s, acc_test: %s'%(i,args.iterations_num, acc_train, test_acc))
            # i += 1
        
        checkpoint['last'] = Classifer.get_params()
        draw_loss(index_list, loss_train_list, args.train_output, 'train_loss')
        draw_loss(index_list,loss_test_list, args.test_output, 'test_loss')
        draw_loss(index_list,acc_test_list, args.acc_output, 'test_acc')
        save_model(checkpoint,args.model_output)
    else:
        assert args.resume, 'resume should not be null'
        testset = ImageLoader(args.test_img_path, args.test_lab_path, args.batch_size)
        test_loss, test_acc = Classifer.test(testset)
        print('test loss: %s, acc:%s'%(test_loss, test_acc))
        


    
    
    
    
        
        



        
        
        
    
        


