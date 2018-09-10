import tensorflow as tf

from tensorflow import keras as K
from keras.losses import binary_crossentropy

batch_shape = (64, 400)
batch_unl = 96
power_iteration = 1
clipping = True


class Model:
	def __init__(self, session):
		# Session
		self.set_session(session)

	def set_session(self, session):
		self.sess = session
		#  Keras will use the session we registered
		# to initialize all variables that it creates internally.
		K.backend.set_session(self.sess)

	def clear_session(self):
		K.backend.clear_session()

	def build(self, embedding_matrix=None, lstm_units=512, dense_units=30):
		# Define layers of Model
		self.embedding = K.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
											weights=[embedding_matrix], trainable=False, name='Embedding')
		self.dropout = K.layers.Dropout(rate=0.5, seed=91, name='Dropout')
		self.lstm = K.layers.LSTM(lstm_units, name='lstm')
		self.dense = K.layers.Dense(dense_units, activation='relu', name='Dense')
		self.prob = K.layers.Dense(1, activation='sigmoid', name='Prob')
		# Optimizer
		self.optimizer = None
		# Scalar and Summaries
		self.loss = None
		self.accuracy = None
		self.merged_summary_op = None

	def get_pred_and_emb(self, x, perturbation=None):
		emb = self.embedding(x)
		drop = self.dropout(emb)
		if perturbation is not None:
			drop += perturbation
		lstm = self.lstm(drop)
		dense = self.dense(lstm)
		out = self.prob(dense)
		return out, emb

	def get_optimizer(self, lr=0.0005, decay=False):
		initial_learning_rate = lr
		if not decay:
			self.optimizer = tf.train.AdamOptimizer(initial_learning_rate)
			return self.optimizer
		else:
			# Creates a variable to hold the global_step.
			global_step = tf.Variable(0, trainable=False, name='global_step')
			# global_step = tf.train.get_or_create_global_step()
			# Exponential Decay learning rate
			learning_rate = tf.train.exponential_decay(
				initial_learning_rate,
				global_step,
				1,
				0.9998,
				staircase=True
			)
			# Passing global_step to minimize() will increment it at each step.
			self.optimizer = tf.train.AdamOptimizer(learning_rate)
			return self.optimizer, global_step

	@staticmethod
	def gradient_clipping(opt, loss, global_step=None):
		# Extract trainable variable
		t_vars = tf.trainable_variables()
		# Compute the gradients and clip it.
		grads, _ = tf.clip_by_global_norm(
			tf.gradients(loss, t_vars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N), 1)
		# Apply the clipped gradients. Op to update all variables according to their gradient
		# Apply the optimizer to the variables / gradients tuple.
		if global_step is not None:
			train_op = opt.apply_gradients(zip(grads, t_vars), global_step=global_step)
		else:
			train_op = opt.apply_gradients(zip(grads, t_vars))

		return train_op

	@staticmethod
	def kl_divergence(prob_p, prob_q):
		p = tf.distributions.Bernoulli(probs=prob_p)
		q = tf.distributions.Bernoulli(probs=prob_q)
		return tf.distributions.kl_divergence(p, q, allow_nan_stats=False)

	def loss_fn(self, x, labels):
		pred, emb = self.get_pred_and_emb(x)
		# Minimize error using cross entropy
		self.loss = tf.reduce_mean(binary_crossentropy(labels, pred))
		return self.loss, emb

	def adversarial_loss(self, x, labels, embedded, loss):
		"""Adds gradient to embedding and recomputes classification loss."""
		grad, = tf.gradients(loss, embedded, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
		grad = tf.stop_gradient(grad)
		# Perturbations
		#alpha = tf.reduce_max(tf.abs(grad), axis=(1, 2), keep_dims=True) + 1e-12
		#l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(grad / alpha, 2), axis=(1, 2), keep_dims=True) + 1e-6)
		# shape(grad) = (batch, num_timesteps, emb_dim)
		# Scale over the full sequence, dims (1, 2)
		#l2_norm = tf.sqrt(tf.reduce_sum(tf.pow(grad, 2), axis=(1, 2), keep_dims=True) + 1e-6)
		#grad_unit = grad / l2_norm
		#perturb = 0.02 * grad_unit
		perturb = 0.02 * tf.nn.l2_normalize(grad, dim=1)
		pred, _ = self.get_pred_and_emb(x, perturbation=perturb)
		adv_loss = tf.reduce_mean(binary_crossentropy(labels, pred))
		return adv_loss

	def virtual_adversarial_loss(self, x):
		# Get prob and Embedding
		pred, embedded = self.get_pred_and_emb(x)
		pred = tf.clip_by_value(pred, 1e-7, 1. - 1e-7)
		# # Initialize perturbation with random noise.
		d = tf.random_normal(shape=tf.shape(embedded), dtype=tf.float32)
		for i in range(power_iteration):
			# Normalize random vector
			#l2_norm = tf.sqrt(tf.reduce_sum(tf.pow(d, 2), axis=(1, 2), keep_dims=True))
			#d_norm = d / l2_norm
			d_norm = tf.nn.l2_normalize(d, dim=1)
			d = 0.02 * d_norm
			# Get prob with perturbation
			pert_pred, _ = self.get_pred_and_emb(x, perturbation=d)
			pert_pred = tf.clip_by_value(pert_pred, 1e-7, 1. - 1e-7)
			# Compute KL divergence
			kl = self.kl_divergence(pred, pert_pred)
			# Compute gradients
			grad, = tf.gradients(kl, d, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
			d = tf.stop_gradient(grad)
		#d = 0.02 * (d / tf.sqrt(tf.reduce_sum(tf.pow(d, 2), axis=(1, 2), keep_dims=True)))
		d = 0.02 * tf.nn.l2_normalize(d, dim=1)
		pred = tf.stop_gradient(pred)
		# Compute Virtual Adversarial Loss
		pert_pred, _ = self.get_pred_and_emb(x, perturbation=d)
		pert_pred = tf.clip_by_value(pert_pred, 1e-7, 1. - 1e-7)
		virt_adv_loss = tf.reduce_mean(self.kl_divergence(pred, pert_pred))
		return virt_adv_loss

	def combo_loss(self, x, x_unl, labels, embedded, loss):
		return self.adversarial_loss(x, labels, embedded, loss) + self.virtual_adversarial_loss(x_unl)
