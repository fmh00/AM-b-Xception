from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""

	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    #获取当前的维度顺序
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1#判断图像的格式然后返回图像的维度顺序
    channel = input_feature._keras_shape[channel_axis]
    #shareMLP-W1先定义
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',#kernel_initializer: 卷积核的初始化器.
                             use_bias=True,#Boolean型，是否使用偏置项
                             bias_initializer='zeros')#偏置项的初始化器，默认初始化为0
    # shareMLP-W2先定义
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    #平均池化层
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)#[1:]分片操作，shape（1,1,1,channel）——>（1,1,channel）
                                                       # assert断言操作，如果为false抛出异常

    #经过w1
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    #经过w2
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    #最大池化层
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    #经过w1
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    #w2
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    # 逐元素相加，sigmoid激活
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    # 确定维度顺序
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    # 逐元素相乘，得到下一步空间操作的输入feature
    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7   #卷积核7*7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)#permute实现维度的任意顺序排列（或称置换）
    else:
        channel = input_feature._keras_shape[-1]#取最后一个元素（channel）
        cbam_feature = input_feature
    #经过平均池化
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # axis等于几，就理解成对那一维值进行压缩；二维矩阵（0为列1为行），x为四维矩阵（1,h,w,channel）所以axis=3，对矩阵每个元素求平均；
    # keepdims保持其矩阵二维特性
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1

    # 从channel维度进行拼接
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2

    # 进行卷积操作
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    # 逐元素相乘，得到feature
    return multiply([input_feature, cbam_feature])