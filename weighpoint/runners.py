"""Contains training/validation loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from absl import logging
import gin
from weighpoint import callbacks as cb
from weighpoint.meta import builder as b
from tqdm import tqdm


def _unpack_dataset_values(values):
    assert(isinstance(values, tuple))
    if len(values) == 2:
        features, labels = values
        weights = None
    else:
        features, labels, weights = values
    return features, labels, weights


@gin.configurable
def train(
        model_dir, problem, batch_size, epochs, logits_meta_fn, optimizer,
        callbacks=None, verbose=True, checkpoint_freq=None,
        summary_freq=None, save_config=True, lr_schedule=None):
    # need to run in graph mode for reinitializable iterators
    assert(not tf.executing_eagerly())
    with problem:
        if model_dir is not None:
            model_dir = os.path.expanduser(model_dir)
        datasets = {k: problem.get_dataset(
                        split=k, batch_size=None, prefetch=False)
                    for k in ('train', 'validation')}

        builder = b.MetaNetworkBuilder()
        with builder:
            inputs, labels, weights = _unpack_dataset_values(
                builder.prebatch_inputs_from(datasets['train']))
            logits = logits_meta_fn(inputs, problem.output_spec())
            labels, weights = problem.preprocess_labels(labels, weights)
        if isinstance(logits, tf.RaggedTensor):
            assert(logits.ragged_rank == 1)
            logits = logits.values

        preprocessor = builder.preprocessor(labels, weights)
        model = builder.model((logits,))
        model.compile(
            optimizer=optimizer,
            loss=problem.loss,
            metrics=problem.metrics)

        tf.compat.v1.summary.scalar('lr', model.optimizer.lr)
        custom_summary = tf.compat.v1.summary.merge_all()

        train_steps = problem.examples_per_epoch('train') // batch_size
        validation_steps = problem.examples_per_epoch(
            'validation') // batch_size

        def preprocess_dataset(dataset):
            num_parallel_calls = tf.data.experimental.AUTOTUNE
            return preprocessor.map_and_batch(
                dataset.repeat(),
                batch_size=batch_size,
                num_parallel_calls=num_parallel_calls).prefetch(
                    tf.data.experimental.AUTOTUNE)

        datasets = tf.nest.map_structure(preprocess_dataset, datasets)

        iters = tf.nest.map_structure(
            tf.compat.v1.data.make_initializable_iterator, datasets)

        callbacks, initial_epoch = cb.get_callbacks(
            model,
            callbacks=callbacks,
            checkpoint_freq=checkpoint_freq,
            summary_freq=summary_freq,
            save_config=save_config,
            model_dir=model_dir,
            custom_summary=custom_summary,
            train_steps_per_epoch=train_steps,
            val_steps_per_epoch=validation_steps,
            lr_schedule=lr_schedule,
            train_iter=iters['train'],
            val_iter=iters['validation'],
        )
        model.fit(
            iters['train'],
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=iters['validation'],
            steps_per_epoch=train_steps,
            validation_steps=validation_steps,
            initial_epoch=initial_epoch,
        )


@gin.configurable
def evaluate(
        model_dir, problem, batch_size, logits_meta_fn, optimizer):
    with problem:
        if model_dir is not None:
            model_dir = os.path.expanduser(model_dir)
        dataset = problem.get_dataset(
            split='validation', batch_size=None, prefetch=False)

        builder = b.MetaNetworkBuilder()
        with builder:
            inputs, labels, weights = _unpack_dataset_values(
                builder.prebatch_inputs_from(dataset))
            logits = logits_meta_fn(inputs, problem.output_spec())
            labels, weights = problem.preprocess_labels(labels, weights)
        if isinstance(logits, tf.RaggedTensor):
            assert(logits.ragged_rank == 1)
            logits = logits.values

        preprocessor = builder.preprocessor(labels, weights)
        model = builder.model((logits,))
        model.compile(
            optimizer=optimizer,
            loss=problem.loss,
            metrics=problem.metrics)

        validation_steps = problem.examples_per_epoch(
            'validation') // batch_size

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint is not None:
            saver.restore(tf.keras.backend.get_session(), checkpoint)

        def preprocess_dataset(dataset):
            num_parallel_calls = tf.data.experimental.AUTOTUNE
            return preprocessor.map_and_batch(
                dataset.repeat(),
                batch_size=batch_size,
                num_parallel_calls=num_parallel_calls).prefetch(
                    tf.data.experimental.AUTOTUNE)

        dataset = preprocess_dataset(dataset)
        model.evaluate(dataset, steps=validation_steps)


@gin.configurable
def confusion(
        model_dir, problem, batch_size, logits_meta_fn,
        overwrite=False, split='train'):
    # need to run in graph mode for reinitializable iterators
    with problem:
        if model_dir is not None:
            model_dir = os.path.expanduser(model_dir)
        path = os.path.join(model_dir, 'confusion-%s.npy' % split)
        if not overwrite and os.path.isfile(path):
            logging.info('Found existing confusion matrix at %s' % path)
            return np.load(path)

        logging.info('Computing confusion matrix...')

        assert(not tf.executing_eagerly())
        if model_dir is not None:
            model_dir = os.path.expanduser(model_dir)
        # datasets = {k: problem.get_dataset(
        #                 split=k, batch_size=None, prefetch=False)
        #             for k in ('train', 'validation')}
        dataset = problem.get_dataset(
            split=split, batch_size=None, prefetch=False)

        builder = b.MetaNetworkBuilder()
        with builder:
            inputs, labels, weights = _unpack_dataset_values(
                builder.prebatch_inputs_from(dataset))
            logits = logits_meta_fn(inputs, problem.output_spec())
            labels, weights = problem.preprocess_labels(labels, weights)
        if isinstance(logits, tf.RaggedTensor):
            assert(logits.ragged_rank == 1)
            logits = logits.values

        preprocessor = builder.preprocessor(labels, weights)
        model = builder.model((logits,))

        train_steps_per_epoch = \
            problem.examples_per_epoch('train') // batch_size
        examples = problem.examples_per_epoch(split)
        steps = examples // batch_size
        if examples % batch_size > 0:
            steps += 1
        saver_callback = cb.SaverCallback(model_dir, train_steps_per_epoch)

        def preprocess_dataset(dataset):
            num_parallel_calls = tf.data.experimental.AUTOTUNE
            return preprocessor.map_and_batch(
                dataset,
                batch_size=batch_size,
                num_parallel_calls=num_parallel_calls).prefetch(
                    tf.data.experimental.AUTOTUNE)

        num_classes = problem.output_spec().shape[-1]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        dataset = preprocess_dataset(dataset)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        initializer = iterator.initializer
        inputs, labels = iterator.get_next()
        logits = model(tf.nest.flatten(inputs))
        pred = tf.argmax(logits, axis=-1)

        sess = tf.compat.v1.keras.backend.get_session()
        saver_callback.restore()
        if saver_callback.last_saved_epoch() is None:
            raise RuntimeError('No saved data found in %s' % model_dir)

        try:
            sess.run(initializer)
            tf.keras.backend.set_learning_phase(0)
            for i in tqdm(range(steps)):
                pred_np, labels_np = sess.run([pred, labels])
                for p, l in zip(pred_np, labels_np):
                    confusion[p, l] += 1
            sess.run([pred, labels])
            raise RuntimeError('step count incorrect')
        except tf.errors.OutOfRangeError:
            pass
        logging.info(
            'Confusion matrix calculation complete. Saving to %s' % path)
        np.save(path, confusion)
        return confusion
