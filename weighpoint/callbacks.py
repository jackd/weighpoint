from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin.tf
from weighpoint.tf_compat import is_v1
import os

GinConfigSaverCallback = gin.config.external_configurable(
    gin.tf.GinConfigSaverCallback)


@gin.configurable
class SaverCallback(tf.keras.callbacks.Callback):
    """Callback using a standard `tf.train.Saver`."""

    def __init__(
            self, model_dir, train_steps_per_epoch, checkpoint_freq=1,
            max_to_keep=5, **saver_kwargs):
        if not is_v1:
            raise RuntimeError(
                'SaverCallback is only usable in tensorflow v1 - see '
                'CheckpointManagerCallback for substitute')
        self._model_dir = model_dir
        self._train_steps_per_epoch = train_steps_per_epoch
        self._checkpoint_freq = checkpoint_freq
        self._saver = tf.train.Saver(max_to_keep=max_to_keep, **saver_kwargs)
        self._last_save = None
        self._epoch = None

    def last_saved_epoch(self):
        latest_checkpoint = tf.train.latest_checkpoint(self._model_dir)
        if latest_checkpoint is None:
            return None
        else:
            _, filename = os.path.split(latest_checkpoint)
            steps = int(filename.split('.')[0].split('-')[1])
            assert(steps % self._train_steps_per_epoch == 0)
            return steps // self._train_steps_per_epoch

    def restore(self, checkpoint=None):
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(self._model_dir)
            if checkpoint is None:
                return
        self._saver.restore(
            tf.keras.backend.get_session(), checkpoint)

    def on_train_begin(self, logs=None):
        self.restore()

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        self._epoch = epoch
        if epoch % self._checkpoint_freq == 0 and self._last_save != epoch:
            self._save()

    def on_train_end(self, logs=None):
        self._save()

    def _save(self):
        if self._epoch is not None and (
                self._last_save is None or self._last_save < self._epoch):
            self._saver.save(
                tf.keras.backend.get_session(),
                os.path.join(self._model_dir, 'model'),
                global_step=self._train_steps_per_epoch * self._epoch)
            self._last_save = self._epoch


@gin.configurable
class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """
    Callback wraping `tf.train.CheckpointManager`.

    Restores previous checkpoint `on_train_begin`

    Example usage:
    ```python
    model = get_model(...)
    model.compile(optimizer=optimizer, ...)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, '/tmp/my_model', max_to_keep=5)
    callback = CheckpointManagerCallback(checkpoint, manager, period=1)

    model.fit(..., callbacks=[callbacks])
    ```
    """
    def __init__(self, checkpoint, manager, period=1, save_on_train_end=True):
        if is_v1:
            raise RuntimeError(
                'CheckpointManagerCallback is not usable in tensorflow v1 - '
                'see SaverCallback for substitute')
        self._manager = manager
        self._checkpoint = checkpoint
        self._period = period
        self._save_on_train_end = save_on_train_end
        self._restored = False
        self._epoch_count = None
        self._last_save = None

    def _on_begin(self):
        if not self._restored:
            self.restore()

    def restore(self, save_path=None):
        if save_path is None:
            save_path = self._manager.latest_checkpoint
        self._checkpoint.restore(save_path)
        self._restored = True

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def on_epoch_end(self, epoch, logs=None):
        epochs_finished = epoch + 1
        self._epoch_count = epochs_finished
        if epochs_finished % self._period == 0:
            self._save()

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save()

    def _save(self):
        if self._epoch_count is None:
            return
        if self._last_save != self._epoch_count:
            self._manager.save(self._epoch_count)
            self._last_save = self._epoch_count


@gin.configurable
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, custom_summary, *args, **kwargs):
        self.__custom_summary = custom_summary
        self.__last_write = 0
        super(CustomTensorBoard, self).__init__(*args, **kwargs)

    def __write_custom_summary(self, summary_val):
        if self._total_batches_seen - self.__last_write >= self.update_freq:
            self.writer.add_summary(summary_val, self._total_batches_seen)
            self.__last_write = self._total_batches_seen

    def on_train_begin(self, logs=None):
        self.model._fit_function.fetches.append(self.__custom_summary)
        self.model._fit_function.fetch_callbacks[
            self.__custom_summary] = self.__write_custom_summary
        super(CustomTensorBoard, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.model._fit_function.fetches.remove(self.__custom_summary)
        self.model._fit_function.fetch_callbacks.pop(self.__custom_summary)
        super(CustomTensorBoard, self).on_train_end(logs)


def initializer_callback(train_iter, val_iter):
    kwargs = {}

    sess = tf.compat.v1.keras.backend.get_session()
    train_initializer = train_iter.initializer
    val_initializer = val_iter.initializer
    initializers = [train_initializer, val_initializer]

    kwargs['on_epoch_begin'] = lambda _, __: sess.run(train_initializer)
    kwargs['on_epoch_end'] = lambda _, __: sess.run(val_initializer)
    kwargs['on_train_begin'] = lambda _: sess.run(initializers)
    return tf.keras.callbacks.LambdaCallback(**kwargs)


@gin.configurable
def exponential_decay_lr_schedule(lr0, factor):
    def f(epoch):
        return lr0 * (factor ** epoch)
    return f


def get_callbacks(
        model,
        callbacks=None,
        checkpoint_freq=None,
        summary_freq=None,
        save_config=True, model_dir=None,
        custom_summary=None,
        train_steps_per_epoch=None,
        val_steps_per_epoch=None,
        lr_schedule=None,
        train_iter=None,
        val_iter=None,
        ):
    if callbacks is None:
        callbacks = []
    else:
        callbacks = list(callbacks)

    initial_epoch = 0
    if checkpoint_freq is not None:
        checkpoint = tf.train.Checkpoint(
                optimizer=model.optimizer, model=model)
        manager = tf.train.CheckpointManager(
            checkpoint, model_dir, max_to_keep=5)
        saver_callback = CheckpointManagerCallback(
            checkpoint, manager, period=checkpoint_freq)
        chkpt = manager.latest_checkpoint
        if chkpt is not None:
            for substr in chkpt.split('.')[-1::-1]:
                try:
                    last_step = int(substr)
                    assert(last_step % train_steps_per_epoch == 0)
                    initial_epoch = last_step // train_steps_per_epoch
                    break
                except Exception:
                    pass
            else:
                raise RuntimeError('Unrecognized checkpoint prefix %s' % chkpt)

        callbacks.append(saver_callback)

    if summary_freq:
        kwargs = dict(
            write_graph=False, log_dir=model_dir, update_freq=summary_freq)
        if custom_summary is None or not is_v1:
            tb_callback = tf.keras.callbacks.TensorBoard(**kwargs)
        else:
            tb_callback = CustomTensorBoard(
                custom_summary=custom_summary, **kwargs)
        if train_steps_per_epoch is not None:
            initial_train_steps = initial_epoch*train_steps_per_epoch
            tb_callback._total_batches_seen = initial_train_steps
            tb_callback._samples_seen = initial_train_steps
        if val_steps_per_epoch is not None:
            initial_val_steps = initial_epoch*val_steps_per_epoch
            tb_callback._total_val_batches_seen = initial_val_steps

        callbacks.append(tb_callback)

    if lr_schedule is not None:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

    if save_config:
        callbacks.append(GinConfigSaverCallback(model_dir))

    assert((train_iter is None) == (val_iter is None))
    if not any(it is None for it in (train_iter, val_iter)):
        callbacks.append(initializer_callback(train_iter, val_iter))
    return callbacks, initial_epoch
