import os
import time

import numpy as np
import tensorflow as tf

import create_pretraining_data
import metrics
import modeling
import optimization

# hyperparameters
data_dir = '../MSMARCO_tfrecord/'
bert_config_file = '../uncased_L-12_H-768_A-12/bert_config.json'
bert_config_file_large = '../uncased_L-24_H-1024_A-16/bert_config.json'
init_checkpoint = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
init_checkpoint_large = '../uncased_L-24_H-1024_A-16/bert_model.ckpt'
output_dir = './model_dense_multi_GPU/'

num_gpu = 1
is_training = True
is_fine_tuning = False

num_labels = 2
max_seq_length = 512
query_length = 64
document_length = 448 # 512 - 64 = 448
train_batch_size = 8
learning_rate = 1e-5
num_train_steps = 400000
num_warmup_steps = 0
save_checkpoints_steps = 500
iterations_per_loop = 1000

num_warmup_steps = num_warmup_steps * iterations_per_loop
num_train_steps = num_train_steps * iterations_per_loop

class BertBaseModel():
    def __init__(self, batch_size, learning_rate, num_train_steps, num_warmup_steps, num_labels,
                 seq_length, query_length, bert_config, use_one_hot_embeddings, is_training, is_fine_tuning, gpu_num):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.query_length = query_length
        self.bert_config = bert_config
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.is_training = is_training
        self.is_fine_tuning = is_fine_tuning
        self.gpu_num = gpu_num
        self.gpu_step = int(self.batch_size / self.gpu_num)
        self.initialize = True
        
        self.input_layer()
        self.loss()
        
        devices = self.get_available_gpus()
        print("Available Device:", devices)
        
    def input_layer(self):
        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length])
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length])
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length])
        self.label_ids = tf.placeholder(tf.int32, [None])
        
    # Source:
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    def get_available_gpus(self):
        """
            Returns a list of the identifiers of all visible GPUs.
        """
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    # see https://github.com/tensorflow/tensorflow/issues/9517
    def assign_to_device(self, device, ps_device):
        """Returns a function to place variables on the ps_device.

        Args:
            device: Device for everything but variables
            ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

        If ps_device is not set then the variables will be placed on the default device.
        The best device for shared varibles depends on the platform as well as the
        model. Start with CPU:0 and then test GPU:0 to see if there is an
        improvement.
        """
        PS_OPS = [
            'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
            'MutableHashTableOfTensors', 'MutableDenseHashTable'
        ]
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in PS_OPS:
                return ps_device
            else:
                return device
        return _assign
    
    def build(self, gpu_idx):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_fine_tuning,
            input_ids=self.input_ids[gpu_idx*self.gpu_step:(gpu_idx+1)*self.gpu_step],
            input_mask=self.input_mask[gpu_idx*self.gpu_step:(gpu_idx+1)*self.gpu_step],
            token_type_ids=self.segment_ids[gpu_idx*self.gpu_step:(gpu_idx+1)*self.gpu_step],
            use_one_hot_embeddings=self.use_one_hot_embeddings)
        
        print("GPU:", gpu_idx)
        
        # [batch_size, hidden_size]
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        # applied dropout if training
        output_layer = tf.layers.dropout(inputs=output_layer, rate=0.1, training=self.is_training)
        logits = tf.layers.dense(inputs=output_layer, units=self.num_labels, name='dense', reuse=tf.AUTO_REUSE)
        self.log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(self.label_ids[gpu_idx*self.gpu_step:(gpu_idx+1)*self.gpu_step], depth=self.num_labels, dtype=tf.float32)
        
        per_example_loss = -tf.reduce_sum(one_hot_labels * self.log_probs, axis=-1)
        self.total_loss = tf.reduce_mean(per_example_loss)
      
    def loss(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_op, self.update_loss = self.create_parallel_optimization(optimizer)
        
    def create_parallel_optimization(self, optimizer, controller="/cpu:0"):
        # This function is defined below; it returns a list of device ids like
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        # 0 represent 2
        # 1 represent 3
        # `['/gpu:0', '/gpu:1']`
        devices = self.get_available_gpus()

        # This list keeps track of the gradients per tower and the losses
        tower_grads = []
        losses = []
        
        # Get the current variable scope so we can reuse all variables we need once we get
        # to the second iteration of the loop below
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for i, id in enumerate(devices):
                # custom number of GPU
                if i >= self.gpu_num:
                    break
                name = 'tower_{}'.format(i)
                # Use the assign_to_device function to ensure that variables are created on the
                # controller.
                with tf.device(self.assign_to_device(id, controller)), tf.name_scope(name):
                    # Compute loss and gradients, but don't apply them yet
                    if self.initialize:
                        self.build(i)
                    loss = self.total_loss
            
                    with tf.name_scope("compute_gradients"):
                        # `compute_gradients` returns a list of (gradient, variable) pairs
                        tvars = tf.trainable_variables()
                        tvars = list(filter(lambda x: "bert/pooler" not in x.name, tvars))
                        grads = optimizer.compute_gradients(loss, var_list=tvars)
                        tower_grads.append(grads)
                    losses.append(loss)

                # After the first iteration, we want to reuse the variables.
                outer_scope.reuse_variables()
                
        self.initialize = False
        # Apply the gradients on the controlling device
        with tf.name_scope("apply_gradients"), tf.device(controller):
            # Note that what we are doing here mathematically is equivalent to returning the
            # average loss over the towers and compute the gradients relative to that.
            # Unfortunately, this would place all gradient-computations on one device, which is
            # why we had to compute the gradients above per tower and need to average them here.

            # This function is defined below; it takes the list of (gradient, variable) lists
            # and turns it into a single (gradient, variables) list.
            gradients = self.average_gradients(tower_grads)
            global_step = tf.train.get_or_create_global_step()
            gradients, norm_summary_ops = self.clip_grads(gradients, 1.0, True, global_step)
            apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
            avg_loss = tf.reduce_mean(losses)
            
        return apply_gradient_op, avg_loss
    
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
            over the devices. The inner list ranges over the different variables.
        Returns:
                List of pairs of (gradient, variable) where the gradient has been averaged
                across all towers.
        """
        # calculate average gradient for each shared variable across all GPUs
        average_grads = []
        
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            # We need to average the gradients across each GPU.
            
            g0, v0 = grad_and_vars[0]
            if isinstance(g0, tf.IndexedSlices):
                # If the gradient is type IndexedSlices then this is a sparse
                # gradient with attributes indices and values.
                # To average, need to concat them individually then create
                # a new IndexedSlices object.
                indices = []
                values = []
                for g, v in grad_and_vars:
                    indices.append(g.indices)
                    values.append(g.values)
                all_indices = tf.concat(indices, 0)
                avg_values = tf.concat(values, 0) / len(grad_and_vars)
                # deduplicate across indices
                av, ai = self._deduplicate_indexed_slices(avg_values, all_indices)
                grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

            else:
                # a normal tensor can just do a simple average
                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                grads = [g for g, _ in grad_and_vars]
                grad = tf.reduce_mean(grads, 0)

            # the Variables are redundant because they are shared
            # across towers. So.. just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    
    def _deduplicate_indexed_slices(self, values, indices):
        """Sums `values` associated with any non-unique `indices`.
        Args:
          values: A `Tensor` with rank >= 1.
          indices: A one-dimensional integer `Tensor`, indexing into the first
          dimension of `values` (as in an IndexedSlices object).
        Returns:
          A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
          de-duplicated version of `indices` and `summed_values` contains the sum of
          `values` slices associated with each unique index.
        """
        unique_indices, new_index_positions = tf.unique(indices)
        summed_values = tf.unsorted_segment_sum(values, new_index_positions, tf.shape(unique_indices)[0])
        return (summed_values, unique_indices)
    
    def clip_by_global_norm_summary(self, t_list, clip_norm, norm_name, variables):
        # wrapper around tf.clip_by_global_norm that also does summary ops of norms

        # compute norms
        # use global_norm with one element to handle IndexedSlices vs dense
        norms = [tf.global_norm([t]) for t in t_list]

        # summary ops before clipping
        summary_ops = []
        for ns, v in zip(norms, variables):
            name = 'norm_pre_clip/' + v.name.replace(":", "_")
            summary_ops.append(tf.summary.scalar(name, ns))

        # clip
        clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

        # summary ops after clipping
        norms_post = [tf.global_norm([t]) for t in clipped_t_list]
        for ns, v in zip(norms_post, variables):
            name = 'norm_post_clip/' + v.name.replace(":", "_")
            summary_ops.append(tf.summary.scalar(name, ns))

        summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

        return clipped_t_list, tf_norm, summary_ops


    def clip_grads(self, grads, all_clip_norm_val, do_summaries, global_step):
        # grads = [(grad1, var1), (grad2, var2), ...]
        def _clip_norms(grad_and_vars, val, name):
            # grad_and_vars is a list of (g, v) pairs
            grad_tensors = [g for g, v in grad_and_vars]
            vv = [v for g, v in grad_and_vars]
            scaled_val = val
            if do_summaries:
                clipped_tensors, g_norm, so = self.clip_by_global_norm_summary(
                    grad_tensors, scaled_val, name, vv)
            else:
                so = []
                clipped_tensors, g_norm = tf.clip_by_global_norm(
                    grad_tensors, scaled_val)

            ret = []
            for t, (g, v) in zip(clipped_tensors, grad_and_vars):
                ret.append((t, v))

            return ret, so

        ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')
        assert len(ret) == len(grads)
        return ret, summary_ops
    
    def train(self, dataset, ckpt_dir, output_dir, save_checkpoints_step):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # we can load "bert" variable only, the rest should be trained by our self
        tvars = tf.trainable_variables()
        tvars = list(filter(lambda x: "bert" in x.name, tvars))
        saver = tf.train.Saver(var_list=tvars)
        saver2 = tf.train.Saver()
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_dir)
#             saver2.restore(sess, ckpt_dir)
            
            train_iterator = dataset.make_one_shot_iterator()
            next_element = train_iterator.get_next()

            loss = []
            min_loss = 1
            step = 0
            print("Start training!")
            
            for i in range(self.num_train_steps):
                features = sess.run(next_element)
                train_loss_steps, _ = sess.run(
                    [self.update_loss, self.update_op], 
                    feed_dict={
                        self.input_ids: features["input_ids"],
                        self.input_mask: features["input_mask"],
                        self.segment_ids: features["segment_ids"],
                        self.label_ids: features["label_ids"]
                    })
                loss.append(train_loss_steps)
                step += 1

                if i % save_checkpoints_step==0:
                    loss = np.sum(loss)
                    loss /= step;

                    time_tuple = time.localtime()
                    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time_tuple)
                    print(time_string + " {:d} step loss: {:.4f}".format(i, loss))

                    with open(output_dir + "loss.txt", "a") as text_file:
                        text_file.write(time_string + " {:d} step loss: {:.4f}\n".format(i, loss))

                    saver2.save(sess, output_dir + 'bert-model.ckpt')
                    loss = []
                    step = 0
                    print("save checkpoint in {}".format(output_dir + 'bert-model.ckpt-' + str(i)))
        
    def predict(self, dataset, ckpt_dir, output_dir, batch_size, max_eval_examples):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        saver = tf.train.Saver()
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_dir)
            
            train_iterator = dataset.make_one_shot_iterator()
            next_element = train_iterator.get_next()
            
            step = max_eval_examples / batch_size
            log_probs = np.zeros((1, num_labels))
            labels = np.zeros((1, 1))
            
            print("Start predicting!")
            
            for i in range(int(step)):
                features = sess.run(next_element)
                log_prob = sess.run(
                    self.log_probs, 
                    feed_dict={
                        self.input_ids: features["input_ids"],
                        self.input_mask: features["input_mask"],
                        self.segment_ids: features["segment_ids"]
                    })
                
                log_probs = np.concatenate((log_probs, log_prob), axis=0)
                labels = np.concatenate((labels, features["label_ids"].reshape(-1, 1)), axis=0)
                
                if i % 100 == 0:
                    time_tuple = time.localtime()
                    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time_tuple)
                    print(time_string + " {:d} step".format(i))
                    
                    with open(output_dir + "/eval_step.txt", "a") as text_file:
                        text_file.write(time_string + " {:d} step\n".format(i))
                        
            print("End predicting!")
            
            log_probs = np.delete(log_probs, 0, 0)
            labels = np.delete(labels, 0, 0)
            
        return log_probs, labels
    
if __name__ == "__main__":
    tf.reset_default_graph()
    
    bert_config = modeling.BertConfig.from_json_file(bert_config_file_large)
    dataset = create_pretraining_data.input_builder(dataset_path=data_dir + "/dataset_train.tf", seq_length=max_seq_length,
                                                    batch_size=train_batch_size, is_training=is_training)
    
    Bert = BertBaseModel(batch_size=train_batch_size, learning_rate=learning_rate, num_train_steps=num_train_steps, 
                         num_warmup_steps=num_warmup_steps, num_labels=num_labels, seq_length=max_seq_length, 
                         query_length=query_length, bert_config=bert_config, use_one_hot_embeddings=False, 
                         is_training=is_training, is_fine_tuning=is_fine_tuning, gpu_num=num_gpu)
    # training
    Bert.train(dataset=dataset, ckpt_dir=init_checkpoint_large, output_dir=output_dir,
               save_checkpoints_step=save_checkpoints_steps)