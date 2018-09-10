import os
import config
import numpy as np
import tensorflow as tf

from model import Model
from sklearn.utils import shuffle
from tensorflow import keras as K
from keras.metrics import binary_accuracy


batch_shape = (64, 400)
batch_unl = 96
training_epochs = 10

adversarial = True
virt_adversarial = False
combo = False
if adversarial and virt_adversarial:
	combo = True
	adversarial = False
	virt_adversarial = False

decay = False
clipping = True

display_epoch = 1
save = False
if os.path.exists("logs/"):
	logs_path = "logs/"
else:
	os.makedirs("logs/")
if os.path.exists("Model/"):
	model_path = "Model/model.ckpt"
else:
	os.makedirs("Model/")

padding_post = False


def test(sess, model):
	X_test = np.load(os.path.join(config.dataset_base_dir, "X_test_{}.npz".format(config.dataset_name)))['arr_0']
	y_test = np.load(os.path.join(config.dataset_base_dir, "y_test_{}.npz".format(config.dataset_name)))['arr_0']
	y_test = y_test.reshape(y_test.shape[0], 1)

	# Padding
	if not padding_post:
		X_test = K.preprocessing.sequence.pad_sequences(X_test, maxlen=batch_shape[1])
	else:
		X_test = K.preprocessing.sequence.pad_sequences(X_test, maxlen=batch_shape[1], padding='post')

	y = tf.placeholder(tf.float32, shape=(None, 1), name='test_labels')
	X = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='test_input')

	pred, _ = model.get_pred_and_emb(X)
	accuracy = tf.reduce_mean(binary_accuracy(y, pred))

	test_accuracies = list()
	total_batch = int(X_test.shape[0] / batch_shape[0])
	X_test, y_test = shuffle(X_test, y_test)
	avg_test_acc = 0
	for i in range(total_batch):
		offset = (i * batch_shape[0]) % (y_test.shape[0] - batch_shape[0])
		# Generate a minibatch.
		batch_data = X_test[offset:offset + batch_shape[0]]
		batch_labels = y_test[offset:offset + batch_shape[0]]
		# test mode
		fd = {X: batch_data, y: batch_labels, K.backend.learning_phase(): 0}
		acc_test = sess.run(accuracy, feed_dict=fd)
		test_accuracies.append(acc_test)
		avg_test_acc += acc_test / total_batch

	np.savez_compressed(
		'Plot/{}_test_accuracies_adv_{}_vadv_{}_decay_{}_clip_{}.npz'.format(config.dataset_name, adversarial, virt_adversarial,
																		decay,
																		clipping), test_accuracies)
	print("\nAverage accuracy on test set  is {:.3f}".format(avg_test_acc))


def validation(sess, model, x, y):
	y_val = tf.placeholder(tf.float32, shape=(None, 1), name='validation_labels')
	X_val = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='validation_inputs')

	pred, _ = model.get_pred_and_emb(X_val)
	accuracy = tf.reduce_mean(binary_accuracy(y_val, pred))

	total_batch = int(x.shape[0] / batch_shape[0])
	avg_val_acc = 0
	x, y = shuffle(x, y)
	for i in range(total_batch):
		offset = (i * batch_shape[0]) % (y.shape[0] - batch_shape[0])
		# Generate a minibatch.
		batch_data = x[offset:offset + batch_shape[0]]
		batch_labels = y[offset:offset + batch_shape[0]]
		fd = {X_val: batch_data, y_val: batch_labels, K.backend.learning_phase(): 0}
		acc_val = sess.run(accuracy, feed_dict=fd)
		avg_val_acc += acc_val / total_batch

	print("Average accuracy on validation is {:.3f}".format(avg_val_acc))
	return avg_val_acc


def train(sess, model):

	print("\nGetting data...")
	X_train = np.load(os.path.join(config.dataset_base_dir, "X_train_{}.npz".format(config.dataset_name)))['arr_0']
	y_train = np.load(os.path.join(config.dataset_base_dir, "y_train_{}.npz".format(config.dataset_name)))['arr_0']
	if virt_adversarial or combo:
		X_unl_train = np.load(os.path.join(config.dataset_base_dir, 'unlab_{}.npz'.format(config.dataset_name)))['arr_0']
	else:
		X_unl_train = None

	# Padding
	if not padding_post:
		X_train = K.preprocessing.sequence.pad_sequences(X_train, maxlen=batch_shape[1])
		if X_unl_train is not None:
			X_unl_train = K.preprocessing.sequence.pad_sequences(X_unl_train, maxlen=batch_shape[1])
	else:
		X_train = K.preprocessing.sequence.pad_sequences(X_train, maxlen=batch_shape[1], padding='post')
		if X_unl_train is not None:
			X_unl_train = K.preprocessing.sequence.pad_sequences(X_unl_train, maxlen=batch_shape[1], padding='post')

	# Split the data into a training set and a validation set
	print("Splitting data...")
	indices = np.arange(X_train.shape[0])
	np.random.shuffle(indices)
	X_train = X_train[indices]
	y_train = y_train[indices]
	num_validation_samples = int(0.05 * X_train.shape[0])
	print('{} elements in validation set'.format(num_validation_samples))

	X_train = X_train[:-num_validation_samples]
	y_train = y_train[:-num_validation_samples]
	X_val = X_train[-num_validation_samples:]
	y_val = y_train[-num_validation_samples:]

	# Reshaping
	print("Reshaping data...\n")
	y_val = y_val.reshape(y_val.shape[0], 1)
	y_train = y_train.reshape(y_train.shape[0], 1)

	y = tf.placeholder(tf.float32, shape=(None, 1), name='train_labels')
	X = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='train_inputs')
	X_unl = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='train_unl_inputs')

	with tf.name_scope('Model'):
		# Build model
		# Predictions and Embedding
		pred, _ = model.get_pred_and_emb(X)

	with tf.name_scope('Accuracy'):
		# Accuracy
		accuracy = tf.reduce_mean(binary_accuracy(y, pred))
		"""accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))"""

	with tf.name_scope('Loss'):
		# Binary Cross entropy (cost function)
		loss, emb = model.loss_fn(pred, y)
		if adversarial:
			loss += model.adversarial_loss(X, y, emb, loss)
		elif virt_adversarial:
			loss += model.virtual_adversarial_loss(X_unl)
		elif combo:
			loss += model.combo_loss(X, X_unl, y, emb, loss)

	# Create Optimizer.
	with tf.name_scope('Opt'):
		if not decay:
			opt = model.get_optimizer(lr=0.001)
			if not clipping:
				train_op = opt.minimize(loss)
			else:
				train_op = model.gradient_clipping(opt, loss)
		else:
			opt, global_step = model.get_optimizer(lr=0.001, decay=decay)
			if not clipping:
				train_op = opt.minimize(loss, global_step=global_step)
			else:
				train_op = model.gradient_clipping(opt, loss, global_step=global_step)

	# Initialize the variables
	init = [var.initializer for var in tf.global_variables() if not ('embedding' in var.name)]
	# init_g = tf.global_variables_initializer

	with tf.name_scope("summaries"):
		# Create a summary to monitor cost tensor
		tf.summary.scalar("loss", loss)
		# Create a summary to monitor accuracy tensor
		tf.summary.scalar("accuracy", accuracy)
		# Create summaries to visualize learning rate
		"""if decay:
			tf.summary.scalar('learning_rate', opt._lr)"""
		# Merge all summaries into a single op
		merged_summary_op = tf.summary.merge_all()

	# 'Saver' op to save and restore all the variables
	if save:
		saver = tf.train.Saver()

	# Initialize variables
	print("\nInitializing variables...")
	sess.run(init)

	# op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())  # self.sess.graph

	print("\nRun the command line:\n"
		"--> tensorboard --logdir=logs/ "
		"\nThen open the link printed in terminal into your web browser")

	print("\nTraining...")
	final_losses = list()
	final_accuracies = list()
	final_val_accuracies = list()
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0
		avg_acc = 0

		X_train, y_train = shuffle(X_train, y_train)
		if X_unl_train is not None:
			X_unl_train = shuffle(X_unl_train)
		total_batch = int(X_train.shape[0] / batch_shape[0])
		for i in range(total_batch):
			# Pick an offset within the training data, which has been randomized.
			offset = (i * batch_shape[0]) % (y_train.shape[0] - batch_shape[0])
			# Generate a minibatch.
			batch_data = X_train[offset:offset + batch_shape[0]]
			batch_labels = y_train[offset:offset + batch_shape[0]]
			if virt_adversarial or combo:
				offset_unl = (i * batch_unl) % (X_unl_train.shape[0] - batch_unl)
				batch_data_unl = X_unl_train[offset_unl:offset_unl + batch_unl]
				fd = {X: batch_data, y: batch_labels, X_unl: np.concatenate((batch_data_unl, batch_data), axis=0), K.backend.learning_phase(): 1}
			else:
				fd = {X: batch_data, y: batch_labels, K.backend.learning_phase(): 1}

			_, acc_val, loss_val, summary = sess.run([train_op, accuracy, loss, merged_summary_op],
														feed_dict=fd)

			# Write logs at every iteration
			summary_writer.add_summary(summary, i)

			# Compute average loss and accuracy
			avg_cost += loss_val / total_batch
			avg_acc += acc_val / total_batch

		# saving accuracies and losses
		final_accuracies.append(avg_acc)
		final_losses.append(avg_cost)

		# Display logs per epoch step
		if (epoch + 1) % display_epoch == 0:
			print("Epoch:", '%04d' % (epoch + 1), "loss =", "{:.9f}".format(avg_cost), "acc =",
				"{:.9f}".format(avg_acc))

		# Validate
		print("Validate...")
		final_val_accuracies.append(validation(sess, model, X_val, y_val))

	summary_writer.close()
	print("\nOptimization Finished!")

	try:
		import matplotlib.pyplot as plt
	except ImportError:
		print("\nImport matplotlib failed!!")
		if not os.path.exists("Plot/"):
			os.makedirs("Plot")

		np.savez_compressed(
			'Plot/{}_train_losses_adv_{}_vadv_{}_decay_{}_clip_{}.npz'.format(config.dataset_name, adversarial, virt_adversarial,
																		decay, clipping), final_losses)
		np.savez_compressed(
			'Plot/{}_train_accuracies_adv_{}_vadv_{}_decay_{}_clip_{}.npz'.format(config.dataset_name, adversarial, virt_adversarial, decay,
																			clipping), final_accuracies)
		np.savez_compressed(
			'Plot/{}_val_accuracies_adv_{}_vadv_{}_decay_{}_clip_{}.npz'.format(config.dataset_name, adversarial, virt_adversarial, decay,
																			clipping), final_val_accuracies)
		print("Losses and Accuracies saved!")
	else:
		plt.plot([np.asarray(l).mean() for l in final_losses], color='red', linestyle='solid', marker='o', linewidth=2)
		plt.plot([np.asarray(a).mean() for a in final_accuracies], color='blue', linestyle='solid', marker='o',
				 linewidth=2)
		plt.savefig('Plot/{}_train_e{}_m{}_l{}_adv_{}_vadv_{}_decay_{}_clip_{}.png'.format(config.dataset_name, training_epochs, batch_shape[0], batch_shape[1],
																						adversarial, virt_adversarial,
																		decay, clipping))

	# Save model weights to disk
	if save:
		try:
			save_path = saver.save(sess, model_path)
			print("Model saved in file: %s" % save_path)
		except IOError as err:
			print("Error while saving file.\nException says: {}".format(err))

	# Test
	print("\nEvaluate...")
	test(sess, model)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	configur = tf.ConfigProto()
	configur.gpu_options.allow_growth = True
	session = tf.Session(config=configur)

	print("\nDataset: {}".format(config.dataset_name))

	model = Model(session)
	embedding_matrix_filename = config.dataset_name + '_embedding_matrix_' + str(config.EMBEDDING_DIM) + '.npz'
	embedding_matrix = np.load(os.path.join(config.embedding_base_dir, embedding_matrix_filename))['arr_0']
	model.build(embedding_matrix=embedding_matrix)

	train(session, model)

	model.clear_session()
