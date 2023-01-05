import tensorlayerx as tlx
import tensorflow as tf
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax


class GaANConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Parameters
    ----------
    in_channels: int or tuple
        Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels: int
        Size of each output sample.
    heads: int, optional
        Number of multi-head-attentions.
        (default: :obj:`1`)
    concat: bool, optional
        If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope: float, optional
        LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout_rate: float, optional
        Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    add_self_loops: bool, optional
        If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """

    def __init__(self, in_channels, out_channels, heads=2, m=8,
                 negative_slope=0.2, dropout_rate=0., concat=True, add_bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.m = m
        # print("in:"+str(in_channels))
        # print("out:"+str(out_channels))
        # print("head:"+str(heads))
        self.negative_slope = negative_slope
        self.dropout = dropout_rate
        self.concat = concat
        self.add_bias = add_bias

        self.lin_l = tlx.layers.Linear(
            in_features=self.in_channels, out_features=self.out_channels * self.heads)
        self.lin_r = self.lin_l

        initor = tlx.initializers.TruncatedNormal()
        self.att_r = self._get_weights("att_r", shape=(
            1, self.heads, self.out_channels), init=initor, order=True)
        self.att_l = self._get_weights("att_l", shape=(
            1, self.heads, self.out_channels), init=initor, order=True)
        # GaAN layers
        self.g_lin = tlx.layers.Linear(
            in_features=2*self.in_channels+m, out_features=self.heads)
        self.m_lin = tlx.layers.Linear(
            in_features=self.in_channels, out_features=self.m)
        self.final_lin = tlx.layers.Linear(
            in_features=self.in_channels+self.heads*self.out_channels, out_features=self.heads*self.out_channels)

        if self.add_bias and concat:
            self.bias = self._get_weights("bias", shape=(
                self.heads * self.out_channels,), init=initor)
        elif self.add_bias and not concat:
            self.bias = self._get_weights(
                "bias", shape=(self.out_channels,), init=initor)

    def message(self, Wx, z, alpha, m):

        H, C = self.heads, self.out_channels

        alpha = alpha[0] + alpha[1]
        # print("alpha:")
        # print(alpha.shape)
        alpha = tf.nn.leaky_relu(alpha, self.negative_slope)
        alpha = tf.nn.softmax(alpha)
        alpha = tf.nn.dropout(alpha, self.dropout)

        alpha = tf.reshape(alpha,shape=(-1,H,1))  # alpha = [E,H,1]
        # alpha [E, H, C]
        # print("alpha:")
        # print(alpha.shape)
        # print("x_j")
        # print(x_j.shape)
        Wx_j = Wx[1]
        out = Wx_j * alpha  # [N,H,C] * [E,H,C](alpha进行广播)
        # print("in msg:")
        # print(out.shape)

        # GaAN修改部分的系数：由节点i和其邻居N_i计算得到自己的g_i

        return (out, m[1], z[1])

    def forward(self, x, edge_index, num_nodes=None):

        H, C = self.heads, self.out_channels

        w_l = tlx.reshape(self.lin_l(x), shape=(-1, H, C))
        w_r = tlx.reshape(self.lin_r(x), shape=(-1, H, C))
        # print(w_r.shape)
        alpha_l = tf.reduce_sum(w_l * self.att_l, axis=-1)
        alpha_r = tf.reduce_sum(w_r * self.att_r, axis=-1)

        m = self.m_lin(x)
        out, m, z = self.propagate(x, edge_index, Wx=(w_l, w_r), z=(x, x), alpha=(
            alpha_l, alpha_r), m=(m, m), dim_size=num_nodes)
        # 计算gate_i向量
        g_in = tf.concat((x.T, m.T, z.T)).T
        g_out = self.g_lin(g_in)
        g_i = tlx.nn.Sigmoid(g_out)
        # print(g_i.shape)
        # print(out.shape)
        # 为各个head的结果一一乘上对应的g_i

        out = tlx.reshape(out * g_i, shape=(-1, H, 1))
        out = tlx.reshape(out, shape=reshape(-1, H * C))
        # 拼接上了本层的数据x，假定是要这么做吧
        out_cat = tf.concat((x.T, out.T)).T
        out_cat = self.final_lin(out_cat)
        # print("in forward:")
        # print(out1.shape)

        # if concat 没做
        if self.add_bias:
            x += self.bias
        return out_cat
    
