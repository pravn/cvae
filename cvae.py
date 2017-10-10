from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class CVAE(nn.Module):
    def __init__(self, input_size, label_size=10, batch_size=128):
        super(CVAE, self).__init__()

        self.input_size = input_size
        self.label_size = label_size
        self.enc_size = 400
        self.dec_size = 400
        self.z_size = 20

        enc_size = self.enc_size
        dec_size = self.dec_size
        z_size   = self.z_size

        self.fc1 = nn.Linear(input_size+label_size, enc_size)
        self.fc21 = nn.Linear(enc_size, z_size)
        self.fc22 = nn.Linear(enc_size, z_size)
        self.fc3 = nn.Linear(z_size+label_size , dec_size)
        self.fc4 = nn.Linear(dec_size, input_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, catz):
        h3 = self.relu(self.fc3(catz))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, label):
        input_size = self.input_size
        label_size = self.label_size
        mu, logvar = self.encode(torch.cat((x, label), 1))
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z,label),1)), mu, logvar


data_size = 784
label_size = 10

model = CVAE(data_size,label_size)
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, data_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * data_size

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    #label_vector = torch.zeros(128, 10)
    batch_size = 128
    label_vector_tmp = torch.FloatTensor(batch_size, 10)
    
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data)

        if(data.size(0)!=128):
            break
        
        #label = Variable(label)
        v = label.view(-1,1)
        label_vector_tmp.zero_()
        label_vector_tmp.scatter_(1,v,1)
        label_vector = Variable(label_vector_tmp)

        data = data.view(-1,data_size)

        if args.cuda:
            data = data.cuda()
            label_vector = label_vector.cuda()
            
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label_vector)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    batch_size = 128
    model.eval()
    test_loss = 0

    label_vector_tmp = torch.FloatTensor(batch_size, 10)
    
    for i, (data, label) in enumerate(test_loader):
        if(data.size(0)!=128):
            break

        v = label.view(-1,1)
        label_vector_tmp.zero_()
        label_vector_tmp.scatter_(1,v,1)
        label_vector = Variable(label_vector_tmp)

        
        data = data.view(-1, data_size)
        
        if args.cuda:
            data = data.cuda()
            label_vector = label_vector.cuda()
            
        data = Variable(data, volatile=True)
        
        recon_batch, mu, logvar = model(data, label_vector)

        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data.view(batch_size, 1, 28, 28)[:n],
                                  recon_batch.view(batch_size, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


label_ = torch.LongTensor(64,1)

for i in range(64):
    label_[i][0] = 5

onehot_tmp = torch.FloatTensor(64,10)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    
    sample = Variable(torch.randn(64, 20))
    onehot_tmp.zero_()
    onehot_tmp.scatter_(1,label_,1)
    onehot = Variable(onehot_tmp)
    
    
    if args.cuda:
       sample = sample.cuda()
       onehot = onehot.cuda()
       sample = model.decode(torch.cat((sample,onehot),1)).cpu()
       save_image(sample.data.view(64, 1, 28, 28),
                  'results/sample_' + str(epoch) + '.png')
