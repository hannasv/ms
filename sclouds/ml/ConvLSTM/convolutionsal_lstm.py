import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """ Inspired by https://github.com/automan000/Convolution_LSTM_PyTorch. """
    def __init__(self, input_channels, hidden_channels, kernel_size, dialation = 1, stride = 1,
                 GPU = False, padding = 'same'):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        # The number of hidden channels need to be larger than ..

        self.input_channels = input_channels # nr of meteorological parameters
        self.hidden_channels = hidden_channels # nr of filters makes out the hidden channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = None

        """
        From the Convolution layer, the most important ones are:

        filters: The number of output filters in the convolution.
        kernel_size: Specifying the height and width of the convolution window.
        padding: One of "valid" or "same".
        data_format: Images format, if channel comes first ("channels_first") or last ("channels_last").
        activation: Activation function. Default is the linear function a(x) = x.

        i - input gate
        f - forget gate
        c - (?) gate
        o - output gate

        les her https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4

        """

        # TWO D convolution since we have a grid.
        # Why the bias in a certain way.
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cpu() #cuda() here
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cpu()#.cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cpu()#.cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cpu(), #cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cpu() )#cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1,
                 effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step # what is this
        self.effective_step = effective_step
        self._all_layers = []

        # set up number of cells.
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i],
                                self.hidden_channels[i],
                                self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        samlples, days, lat, lon, channels = input.shape
        # adds number of
        if padding == 'same':
            """ Padds so that the output dimention and input are the same """
            self.padding = padding_same( self.kernel_size, height, width,
                                         dialation=self.dilation,
                                         stride=self.stride  )

        elif padding == "valid":
            # In keras it means no padding.
            self.padding = int((kernel_size - 1) / 2)

        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize,
                                                             hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

def padding_same(kernel_size, height, width, dialation = 1, stride = 1):
    """ Extended formula from ML lectures includes dialution """
    # TODO: implement for not sqauare filter?
    def p(dim):
        return 0.5*( dim - 1 - stride + dim*stride + 2*kernel_size +
              kernel_size*dialation + dialation )
    return P(height), P(width)

if __name__ == '__main__':
    """
    Add functionalty for available gpus and cpu, training, testing, saving a model.
    Study oblig 2 from in5400.

    .cuda() sends data to gpu. DOT cpu sends it to cup. ''
    add something called .kernal_type() where kernel_typ can be either cpu or cuda.


    input shape : (samples == time_steps, channels (met variables), rows, cols)
    output : (rows, cols) because you predict one timestep

    Fasit should contain the values the timestep behind the train.

    """
    # gradient check
    # TODO different kernelsizes in different hidden layers
    convlstm = ConvLSTM(input_channels = 512, hidden_channels = [128, 64, 64, 32, 32],
                        kernel_size = 3, step = 5,
                        effective_step = [4]).cpu()

    loss_fn = torch.nn.MSELoss()
    example_batch = torch.randn(1, 512, 64, 32)
    example_target = torch.randn(1, 32, 64, 32)
    # Shape of input [samples (nbr days), nr hours, lat, lon, metvars]
    input = Variable(example_batch).cpu()
    target = Variable(example_target).double().cpu()

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
