import tensorflow as tf

def _variable_with_weight_decay(name, shape, wd):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('weight_losses', weight_decay)
  return var  
def conv_layer(scope, input, filter_size, weight_decay,dropout_rate, phase, use_pool):
	# filter_size [filter_height, filter_width, in_channels, out_channels]
	with tf.variable_scope(scope) as var_scope:
		w = _variable_with_weight_decay('w',filter_size,weight_decay);
		b = _variable_with_weight_decay('b',filter_size[-1],weight_decay);
		conv = tf.nn.bias_add(
			tf.nn.conv2d(input,w,[1,1,1,1],"SAME", name='conv'),
			b);
		relu = tf.nn.relu(conv,name='relu');
		drop_out = tf.nn.dropout(relu,dropout_rate,name='drop_out');
		batch_norm = tf.contrib.layers.batch_norm(drop_out,decay=0.1, epsilon = 1e-5,
				updates_collections = None, scale=True,
				is_training=phase,scope='batch_norm');
		output = batch_norm;
		if use_pool:
			pool = tf.nn.max_pool(batch_norm,[1,2,2,1],[1,2,2,1],'VALID',name='pool');
			output = pool;
		return output;

def dense_layer(scope, input, filter_size, weight_decay, dropout_rate):
	# filter_size [input_channel, output_channel]
	with tf.variable_scope(scope) as var_scope:
		w = _variable_with_weight_decay('w', filter_size, weight_decay);
		b = _variable_with_weight_decay('b', filter_size[-1], weight_decay);
		dense = tf.nn.bias_add(tf.matmul(input,w,name='dense'), b);
		relu = tf.nn.relu(dense,'relu');
		drop_out = tf.nn.dropout(relu,dropout_rate,name='drop_out')
	return drop_out;

def inference(X,phase=False,dropout_rate=0.8,n_classes=10,weight_decay=1e-4):
    # logits should be of dimension (batch_size, n_classes)
    # X [batch_size, height, width, num_channel] [b,32,32,3]
    #if not phase:
    #	dropout_rate = 1.0; #test
    batch_size = tf.shape(X);
    batch_size = batch_size[0];
    
    conv1 = conv_layer('layer1',X, [5,5,3,56],weight_decay,dropout_rate,phase,use_pool=True);
    # [batch_size, 16,16,56]
    conv2 = conv_layer('layer2',conv1, [5,5,56,28],weight_decay,dropout_rate,phase,use_pool=True);
    # [batch_size, 8,8,28]
    conv3 = conv_layer('layer3',conv2, [5,5,28,14], weight_decay,dropout_rate,phase,use_pool=True);
    # [batch_size, 4,4,14]
    conv4 = conv_layer('layer4',conv3, [5,5,14,7], weight_decay,dropout_rate,phase,use_pool=False);
    # [batch_size, 4,4,7]
    conv4 = tf.reshape(conv4,[batch_size,112]);
    # [batch_size, 112]
    dense5 = dense_layer('layer5', conv4,[112,n_classes],weight_decay,dropout_rate);
    # [batch_size,10]
    logits = dense5;

    return logits
