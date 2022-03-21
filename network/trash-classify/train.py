# Training
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.optimizers import SGD
from keras import backend as K
from models.model5 import M_b_Xception_728
import tensorflow as tf


trainset_dir = '../../data811/train'
valset_dir = '../../data811/val'

num_classes = 6
learning_rate = 1e-3
batch_size = 64
input_shape = (229, 229, 3)
momentum = 0.9
# base_model = 'M_b_Xception_896'
attention_module = 'cbam_block'
# model_type = base_model if attention_module==None else base_model+'_'+attention_module

# tf.device('/gpu:1')


train_datagen = ImageDataGenerator(rescale=1. / 255,#重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
                                   #rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，
                                   # 在一些模型当中，直接输入原图的像素值可能会落入激活函数的“死亡区”，因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。
                                   shear_range=0.2,#float, 透视变换的范围，所谓shear_range就是错切变换
                                   zoom_range=0.2,#缩放范围，参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作。
                                   horizontal_flip=True,#水平反转
                                   rotation_range=40,#旋转范围
                                   width_shift_range=0.2,#水平平移范围
                                   height_shift_range=0.2,
                                   fill_mode='nearest')#（1）图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；
                                                       #（2）对每一个批次的训练图片，适时地进行数据增强处理（data augmentation）；



val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    trainset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size)
print("train_generator",train_generator)
val_generator = val_datagen.flow_from_directory(
    valset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=True)
print("val_generator",val_generator)
K.clear_session()

optim = SGD(lr=learning_rate, momentum=momentum)
model = M_b_Xception_728(input_shape, num_classes, attention_module)

model.compile(optimizer=optim, loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

csv_path = 'result_show/M_b_Xception_8969.csv'
log_dir = 'result_show/M_b_Xception_896/'
save_weights_path = 'weights/M_b_Xception_896/trash-model-weight-ep-{epoch:02d}-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}.h5'

checkpoint = ModelCheckpoint(save_weights_path, monitor='val_acc', verbose=1,
                             save_weights_only=True, save_best_only=True)
# filepath: 保存模型的路径。
# 　monitor: 被监测的数据。val_acc或val_loss。
# 　verbose: 详细信息模式，0 或者1。0为不打印输出信息，1为打印。
# 　save_best_only: 如果save_best_only=True，将只保存在验证集上性能最好的模型mode: {auto, min, max} 的其中之一。 如果save_best_only=True，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。 对于val_acc，模式就会是max；而对于val_loss，模式就需要是min。在auto模式中，方式会自动从被监测的数据的名字中判断出来。
# 　save_weights_only: 如果 True，那么只有模型的权重会被保存 (model.save_weights(filepath))， 否则的话，整个模型会被保存 (model.save(filepath))。
# 　period: 每个检查点之间的间隔（训练轮数）。

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=20, verbose=1, min_lr=1e-4)
# 回调函数ReduceLROnPlateau，更新学习率
# monitor：监测的值，可以是accuracy，val_loss,val_accuracy
# factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
# patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
# mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率最小值，能缩小到的下限
# ————————————————

#earlystop = EarlyStopping(monitor='val_acc', patience=25, verbose=1)
logging = TensorBoard(log_dir=log_dir, batch_size=batch_size)#可视化工具
csvlogger = CSVLogger(csv_path, append=True)#将epoch的训练结果保存在csv文件中

callbacks = [checkpoint, reduce_lr, logging, csvlogger]



num_epochs = 200

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),#当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
                    epochs=num_epochs,#整数，数据迭代的轮数
                    verbose=1, #日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                    callbacks=callbacks,
                    validation_data=val_generator, #生成验证集的生成器
                    validation_steps=len(val_generator),#当validation_data为生成器时，本参数指定验证集的生成器返回次数
                    workers=8,
                    pickle_safe=True
                    ) #keras中的fit_generator是keras用来为训练模型生成批次数据的工具
# fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1,
#               callbacks=None, validation_data=None, validation_steps=None,
#               class_weight=None, max_q_size=10
#               , workers=1, pickle_safe=False, initial_epoch=0)

tf.device('/gpu:1')