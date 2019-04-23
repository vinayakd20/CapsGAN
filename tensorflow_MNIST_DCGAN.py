import os, time, itertools ,imageio
import  pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from config import cfg
#from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer
from tensorflow.examples.tutorials.mnist import input_data
#import sys
#sys.path.append('capslayer')
#sys.path.append('models')






def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# G(z)
def generator(m, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(m, 1024, [3,3], strides=(1, 1), padding='valid')
#        print conv1.shape
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
#        print lrelu1.shape
        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [3, 3], strides=(1, 1), padding='valid')
#        print conv2.shape
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
#        print lrelu2.shape
        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [3, 3], strides=(1, 1), padding='valid')
#        print conv3.shape
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
#        print lrelu3.shape
        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
#        print conv4.shape
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
#        print conv5.shape
        o = tf.nn.tanh(conv5)
#        print o.shape
        
        # 4th hidden layer
        '''conv5 = tf.layers.conv2d_transpose(lrelu4, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)
        print(lrelu5)
        # output layer
        conv6 = tf.layers.conv2d_transpose(lrelu5, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv6)
        print(o)'''

        return o

# D(x)
def discriminator(input, isTrain=True, reuse=False):
    epsilon = 1e-9
#    if isTrain:
    with tf.variable_scope('discriminator') as scope:
        if reuse:
                labels = tf.constant(0, shape=[cfg.batch_size, ])
        else:
                labels = tf.constant(1, shape=[cfg.batch_size, ])
        Y = tf.one_hot(labels, depth=2, axis=1, dtype=tf.float32)
            
        if reuse:
            scope.reuse_variables()
        with tf.variable_scope('Conv1_layer'):
                # Conv1, [batch_size, 20, 20, 256]
                #print input.get_shape()
                
                conv1 = tf.contrib.layers.conv2d(input, num_outputs=256,
                                                 kernel_size=9, stride=1,
                                                 padding='VALID')
#                print conv1.shape[0].value
                #assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]
    
            # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
                primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
#                print conv1.shape
                caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
                
                #assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]
    
            # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
                """changing the num_outputs to 2 from 10"""
                digitCaps = CapsLayer(num_outputs=2, vec_len=16, with_routing=True, layer_type='FC')
                
                caps2 = digitCaps(caps1)
                v_length = tf.sqrt(reduce_sum(tf.square(caps2),
                                                   axis=2, keepdims=True) + epsilon)
        
        """Loss """
        max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
            # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
        """changing assert value to be [batch, 2, 1, 1] from [batch, 10, 1, 1]"""
        #assert max_l.get_shape() == [cfg.batch_size, 2, 1, 1]
    
            # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
    
            # calc T_c: [batch_size, 10]
            # T_c = Y, is my understanding correct? Try it.
        T_c = Y
            # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
    
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        return margin_loss

fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
'''def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
    plt.ioff()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (32, 32)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
'''

def show_result(num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    ### Loading into pickle###
    path2 = path[:-4] + '.dat'
    #f = open(path2, 'w+')
    test_images.dump(path2)
    #pickle.dump(test_images, f)
    #f.close()

    plt.ioff()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        #   current_image = np.reshape(test_images[k], (32, 32))


        #	fig2 = plt.figure(k)
        #	ax = fig.add_subplot(1,1,1)
        #   ax.imshow(current_image, cmap='gray')
        #	fig2.savefig('./MNIST_DCGAN_results/images/{}.png'.format(k))
        ax[i, j].imshow(np.reshape(test_images[k], (32, 32)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = cfg.batch_size
lr = 0.0001
train_epoch = 20

# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# variables : input
x = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
z = tf.placeholder(tf.float32, shape=(cfg.batch_size, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator

'''D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
'''

#realProb, realReconsLoss = discriminator(x, isTrain)
realLoss = discriminator(x, isTrain)
#fakeProb, fakeReconsLoss = discriminator(G_z, isTrain, reuse=True)
fakeLoss = discriminator(G_z, isTrain, reuse=True)
#D_loss = tf.reduce_mean(fakeProb) - tf.reduce_mean(realProb) + 0.0005*realReconsLoss + 0.0005*fakeReconsLoss  # This optimizes the discriminator.
D_loss = fakeLoss - realLoss
#G_loss = - tf.reduce_mean(fakeProb) # This optimizes the generator.
G_loss = - fakeLoss

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr*0.5, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)
writer = tf.summary.FileWriter('./my_graph/mnist', sess.graph)
tf.global_variables_initializer().run()

# MNIST resize and normalization
#train_set = tf.image.resize_images(mnist.train.images, [32, 32]).eval()
train_set = mnist.train.images
#train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(mnist.train.num_examples // cfg.batch_size):
        # update discriminator
        x_ = train_set[iter*cfg.batch_size:(iter+1)*cfg.batch_size]
        z_ = np.random.normal(0, 1, (cfg.batch_size, 1, 1, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (cfg.batch_size, 1, 1, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
        G_losses.append(loss_g_)
        #if iter % 10 == 0:
        print('Batch Done: ', iter + 1)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
writer.close()
sess.close()
