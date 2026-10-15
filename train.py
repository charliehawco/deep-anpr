#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#	 The above copyright notice and this permission notice shall be included
#	 in all copies or substantial portions of the Software.
#
#	 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#	 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#	 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#	 NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#	 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#	 USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines for training the network.

"""


__all__ = (
	'train',
)


import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
import gen
import model


def code_to_vec(p, code):
	print("calling procedure 'code_to_vec({},{})'".format(p, code))
	def char_to_vec(c):
		print("calling procedure 'char_to_vec({})'".format(c))
		y = numpy.zeros((len(common.CHARS),))
		y[common.CHARS.index(c)] = 1.0
		return y

	c = numpy.vstack([char_to_vec(c) for c in code])

	return numpy.concatenate([[1. if p else 0], c.flatten()])


def read_data(img_glob):
	print("calling procedure 'read_data({})'".format('img_glob'))
	for fname in sorted(glob.glob(img_glob)):
		im = (cv2.imread(fname)[:, :, 0].astype(numpy.float32)) / 255
		pName = fname.split("\\")[1].split(".")[0]
		file_id = pName.split("_")[0]
		number_plate = pName.split("_")[1]
		presence = pName.split("_")[2]

		print('file_id = {}'.format(file_id))
		print('number_plate = {}'.format(number_plate))
		print('presence = {}'.format(presence))

		p = presence == '1'
		yield im, code_to_vec(p, number_plate)


def unzip(b):
	print("calling procedure 'unzip({})'".format('b'))
	xs, ys = zip(*b)
	xs = numpy.array(xs)
	ys = numpy.array(ys)
	return xs, ys


def batch(it, batch_size):
	print("calling procedure 'batch({},{})'".format(it, batch_size))
	out = []
	for x in it:
		out.append(x)
		if len(out) == batch_size:
			yield out
			out = []
	if out:
		yield out


def mpgen(f):
	print("calling procedure 'mpgen({})'".format(f))
	def main(q, args, kwargs):
		print("calling procedure 'mpgen.main({},{},{})'".format(q, args, kwargs))
		try:
			for item in f(*args, **kwargs):
				q.put(item)
		finally:
			q.close()

	@functools.wraps(f)
	def wrapped(*args, **kwargs):
		print("calling procedure 'wrapped({},{})'".format(args, kwargs))
		q = multiprocessing.Queue(3)
		print('test1')
		print(q)
		testing = main(q,(50,),{})
		print('performed testing')
		proc = multiprocessing.Process(target=main, args=(q, args, kwargs))
		for x in q:
			print(x)
		print('test2')
		proc.start()
		print('test3')
		try:
			while True:
				item = q.get()
				yield item
		finally:
			proc.terminate()

			last_weights = [p.eval() for p in params]
			numpy.savez("weights.npz", *last_weights)
			print('saved!')

			proc.join()

		print('testing')
	return wrapped


@mpgen
def read_batches(batch_size):
	print("calling procedure 'read_batches({})'".format(batch_size))
	g = gen.generate_ims()
	def gen_vecs():
		print("calling procedure 'read_batches.gen_vecs()'")
		for im, c, p in itertools.islice(g, batch_size):
			yield im, code_to_vec(p, c)

	while True:
		yield unzip(gen_vecs())


def get_loss(y, y_):
	# Calculate the loss from digits being incorrect.  Don't count loss from
	# digits that are in non-present plates.
	print("calling procedure 'get_loss({},{})'".format(y, y_))
	digits_loss = tf.nn.softmax_cross_entropy_with_logits(
										  labels=tf.reshape(y[:, 1:],
													[-1, len(common.CHARS)]),
										  logits=tf.reshape(y_[:, 1:],
													 [-1, len(common.CHARS)]))
	digits_loss = tf.reshape(digits_loss, [-1, 7])
	digits_loss = tf.reduce_sum(digits_loss, 1)
	digits_loss *= (y_[:, 0] != 0)
	digits_loss = tf.reduce_sum(digits_loss)

	# Calculate the loss from presence indicator being wrong.
	presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
														  labels=y[:, :1], logits=y_[:, :1])
	presence_loss = 7 * tf.reduce_sum(presence_loss)

	return digits_loss, presence_loss, digits_loss + presence_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None):
	"""
	Train the network.

	The function operates interactively: Progress is reported on stdout, and
	training ceases upon `KeyboardInterrupt` at which point the learned weights
	are saved to `weights.npz`, and also returned.

	:param learn_rate:
		Learning rate to use.

	:param report_steps:
		Every `report_steps` batches a progress report is printed.

	:param batch_size:
		The size of the batches used for training.

	:param initial_weights:
		(Optional.) Weights to initialize the network with.

	:return:
		The learned network weights.

	"""
	print("calling procedure 'train({},{},{},{})'".format(learn_rate, report_steps, batch_size, initial_weights))
	x, y, params = model.get_training_model()

	y_ = tf.compat.v1.placeholder(tf.float32, [None, 7 * len(common.CHARS) + 1])

	digits_loss, presence_loss, loss = get_loss(y, y_)
	train_step = tf.compat.v1.train.AdamOptimizer(learn_rate).minimize(loss)

	best = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(common.CHARS)]), 2)
	correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 7, len(common.CHARS)]), 2)

	if initial_weights is not None:
		assert len(params) == len(initial_weights)
		assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

	init = tf.compat.v1.global_variables_initializer()

	def vec_to_plate(v):
		print("calling procedure 'train.vec_to_plate({})'".format(v))
		return "".join(common.CHARS[i] for i in v)

	def do_report():
		print("calling procedure 'train.do_report()'")
		r = sess.run([best,
					  correct,
					  tf.greater(y[:, 0], 0),
					  y_[:, 0],
					  digits_loss,
					  presence_loss,
					  loss],
					 feed_dict={x: test_xs, y_: test_ys})
		num_correct = numpy.sum(
						numpy.logical_or(
							numpy.all(r[0] == r[1], axis=1),
							numpy.logical_and(r[2] < 0.5,
											  r[3] < 0.5)))
		r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
		for b, c, pb, pc in zip(*r_short):
			print("{} {} <-> {} {}".format(vec_to_plate(c), pc,
										   vec_to_plate(b), float(pb)))
		num_p_correct = numpy.sum(r[2] == r[3])

		print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} "
			   "(digits: {}, presence: {}) |{}|").format(
			batch_idx,
			100. * num_correct / (len(r[0])),
			100. * num_p_correct / len(r[2]),
			r[6],
			r[4],
			r[5],
			"".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
										   for b, c, pb, pc in zip(*r_short)))

	def do_batch():
		print("calling procedure 'train.do_batch()'")
		sess.run(train_step,
				 feed_dict={x: batch_xs, y_: batch_ys})
		if batch_idx % report_steps == 0:
			do_report()

	gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95)
	with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(init)
		if initial_weights is not None:
			sess.run(assign_ops)

		test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

		try:
			last_batch_idx = 0
			last_batch_time = time.time()
			batch_iter = enumerate(read_batches(batch_size))
			for batch_idx, (batch_xs, batch_ys) in batch_iter:
				print('batch_idx = {}'.format(batch_idx))
				do_batch()
				if batch_idx % report_steps == 0:
					batch_time = time.time()
					if last_batch_idx != batch_idx:
						print("time for 60 batches {}".format(
							60 * (last_batch_time - batch_time) /
											(last_batch_idx - batch_idx)))
						last_batch_idx = batch_idx
						last_batch_time = batch_time

		except KeyboardInterrupt:
			last_weights = [p.eval() for p in params]
			numpy.savez("weights.npz", *last_weights)
			return last_weights

		finally:
			last_weights = [p.eval() for p in params]
			numpy.savez("weights.npz", *last_weights)
			return last_weights


if __name__ == "__main__":
##	if len(sys.argv) > 1:
##		f = numpy.load(sys.argv[1])
##		initial_weights = [f[n] for n in sorted(f.files,
##												key=lambda s: int(s[4:]))]
##	else:
##		initial_weights = None
	f = numpy.load('weights.npz')
	initial_weights = [f[n] for n in sorted(f.files,
												key=lambda s: int(s[4:]))]
##	initial_weights = None

	train(learn_rate=0.001,
		  report_steps=20,
		  batch_size=50,
		  initial_weights=initial_weights)

