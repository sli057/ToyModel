import os
import numpy as np
import tensorflow as tf
import time
import random
import cnn_model
from random import shuffle
import PIL.Image as Image #cv2
import scipy.io as sio

batch_size = 64                    # YOU MAY MODIFY THIS
max_epoch = 30                      # YOU MAY MODIFY THIS
init_lr = 1e-3                      # YOU MAY MODIFY THIS
summary_ckpt = 50                   # YOU MAY MODIFY THIS
model_ckpt = 500                    # YOU MAY MODIFY THIS
model_save_path = './model'         
tensorboard_path = './Tensorboard'
n_class = 10
image_height = 32
image_width = 32
num_channels = 3
use_pretrained_model = False
gpu_number = 1

if not os.path.exists(model_save_path):
   os.mkdir(model_save_path)

def get_loss(logits, labels):
   # FILL IN; cross entropy loss between logits and labels
   labels = tf.one_hot(labels,10);#not sure if needed
   ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
               labels = labels,
               logits = logits));
   #tf.add_to_collection('losses', ce_loss)
   losses = tf.get_collection('weight_losses')
   total_loss = ce_loss + tf.reduce_mean(losses, name='total_loss')
   return total_loss

def read_data(L, readpos=None):
   image = []
   label = []
   if readpos is None:
      readpos = random.sample(range(len(L)), batch_size)
   for i in range(len(readpos)):
      # FILL IN. Read images and label. image should be of dimension (batch_size,32,32,3) and label of dimension (batch_size,)
      line = L[readpos[i]].strip('\n').split();
      img = Image.open(line[0]);
      image.append(np.array(img));
      label.append(int(line[1]));
   return np.array(image).astype('float32')/128 - 1, np.array(label).astype('int64')

def main():

   # Placeholders
   #starter_learning_rate = tf.placeholder(tf.float32)
   keep_prob = tf.placeholder(tf.float32)
   images = tf.placeholder(tf.float32, [None, image_height, image_width, num_channels])
   labels = tf.placeholder(tf.int64, [None])
   phase = tf.placeholder(tf.bool, [])

   with tf.device('/gpu:%d' %gpu_number):
      logits = cnn_model.inference(images, phase=phase, dropout_rate=keep_prob)
      var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      loss = get_loss(logits, labels)
      # FILL IN. Obtain accuracy of given batch of data.
      correct_pred = tf.equal(tf.argmax(logits, 1), labels)
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
   if use_pretrained_model:
       global_step_ini = np.load('globol_step.npy');
   else:
       global_step_ini = 0;
   # modify learning rate
   global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(global_step_ini),
                    trainable=False
                    )
   learning_rate = tf.train.exponential_decay(init_lr,global_step,
         decay_steps=1000, decay_rate=0.95)
   tf.summary.scalar('Learning Rate',learning_rate);
   # end of modificaiton

   apply_gradient_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step = global_step) # YOU MAY MODIFY THE OPTIMIZER

   # Summary list
   tf.summary.scalar('Total Loss',loss)
   tf.summary.image('Input', images, max_outputs=batch_size)
   for var in var_list:
      tf.summary.histogram(var.op.name, var)

   # Initialize
   init = tf.global_variables_initializer()
   config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
   config.gpu_options.allow_growth = True
   sess = tf.Session(config=config)
   sess.run(init)
   merged = tf.summary.merge_all()
   train_writer = tf.summary.FileWriter(tensorboard_path+"/train", sess.graph)
   test_writer = tf.summary.FileWriter(tensorboard_path+"/test", sess.graph)
   saver = tf.train.Saver(tf.trainable_variables())

   # Start from scratch or load model
   if use_pretrained_model==False:
      lr = init_lr
      epoch_num = 0
   else: 
      lr = np.load('learning_rate.npy')
      epoch_num = np.load('epoch_num.npy')
      saver.restore(sess,model_save_path + '/model')

   trlist = list(open('train.list','r'))
   testlist = list(open('test.list','r'))

   test_accuracy = []
   train_accuracy = []
   train_loss = []
   # Start training
   for i in range(epoch_num, max_epoch):

      # Update learning rate if required

      shuffle(trlist)

      for pos in range(0,len(trlist),batch_size):

         # Load batch data
         t = time.time()
         batch_images, batch_labels = read_data(trlist, range(pos,min(pos+batch_size,len(trlist))))
         dt = time.time()-t

         # Train with batch
         t = time.time()
         _, cost, acc,now_lr,summary = sess.run([apply_gradient_op, loss, accuracy,learning_rate,merged], feed_dict={images:batch_images, labels: batch_labels, phase: True, keep_prob: 0.8})
         print('Epoch: %d, Item : %d, lr: %.5f, Loss: %.5f, Train Accuracy: %.2f, Data Time: %.2f, Network Time: %.2f' %(i, pos, now_lr, cost, acc, dt, time.time()-t))
         train_loss.append(cost)
         train_accuracy.append(acc)

	 train_writer.add_summary(summary,pos)
      # Test, Save model
      # FILL IN. Obtain test_accuracy on the entire test set and append it to variable test_accuracy. 
      batch_images, batch_labels = read_data(testlist, range(len(testlist)));
      summary, acc = sess.run([merged,accuracy],feed_dict={images:batch_images, labels:batch_labels,phase: False, keep_prob: 1.0});
      test_writer.add_summary(summary,pos)
      print('*****************************************');
      print('test accuracy %.2f' % acc);
      test_accuracy.append(acc);
      global_step_check = sess.run(global_step);
      np.save('test_accuracy.npy',test_accuracy); sio.savemat('test_accuracy.mat', mdict={'test_accuracy': test_accuracy})
      np.save('train_accuracy.npy',train_accuracy); sio.savemat('train_accuracy.mat', mdict={'train_accuracy': train_accuracy})
      np.save('train_loss.npy',train_loss); sio.savemat('train_loss.mat', mdict={'train_loss': train_loss})
      np.save('learning_rate.npy', lr)
      np.save('global_step.npy',global_step_check);
      np.save('epoch_num.npy', i)
      saver.save(sess,model_save_path + '/model')
        
   print('Training done.')

if __name__ == "__main__":
   main()
