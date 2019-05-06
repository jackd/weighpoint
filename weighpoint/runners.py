"""Contains training/validation loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import gin
from weighpoint import callbacks as cb
from weighpoint.meta import builder as b


@gin.configurable
def train(
        model_dir, problem, batch_size, epochs, logits_meta_fn, optimizer,
        callbacks=None, verbose=True, checkpoint_freq=None,
        summary_freq=None, save_config=True, lr_schedule=None):
    # need to run in graph mode for reinitializable iterators
    if model_dir is not None:
        model_dir = os.path.expanduser(model_dir)
    datasets = {k: problem.get_dataset(
                    split=k, batch_size=None, prefetch=False)
                for k in ('train', 'validation')}

    builder = b.MetaNetworkBuilder()
    with builder:
        inputs = builder.prebatch_inputs_from(datasets['train'])
        logits = logits_meta_fn(inputs, problem.output_spec())
    if isinstance(logits, tf.RaggedTensor):
        assert(logits.ragged_rank == 1)
        logits = logits.values

    preprocessor = builder.preprocessor()
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
    if model_dir is not None:
        model_dir = os.path.expanduser(model_dir)
    dataset = problem.get_dataset(
        split='validation', batch_size=None, prefetch=False)

    builder = b.MetaNetworkBuilder()
    with builder:
        inputs = builder.prebatch_inputs_from(dataset)
        logits = logits_meta_fn(inputs, problem.output_spec())
    if isinstance(logits, tf.RaggedTensor):
        assert(logits.ragged_rank == 1)
        logits = logits.values

    preprocessor = builder.preprocessor()
    model = builder.model((logits,))
    model.compile(
        optimizer=optimizer,
        loss=problem.loss,
        metrics=problem.metrics)

    validation_steps = problem.examples_per_epoch('validation') // batch_size

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
    exit()
    model.evaluate(dataset, steps=validation_steps)
