import numpy as np
import tensorflow as tf

def input_builder(dataset_path, seq_length, batch_size, is_training, max_eval_examples=None):
    output_buffer_size = batch_size * 1000
    
    def extract_fn(data_record):
        features = {
            "query_ids": tf.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
            "doc_ids": tf.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
            "label": tf.FixedLenFeature([], tf.int64),
        }

        sample = tf.parse_single_example(data_record, features)
        
        query_ids = tf.cast(sample["query_ids"], tf.int64)
        doc_ids = tf.cast(sample["doc_ids"], tf.int64)
        label_ids = tf.cast(sample["label"], tf.int64)
        input_ids = tf.concat((query_ids, doc_ids), 0)
        
        query_segment_id = tf.zeros_like(query_ids)
        doc_segment_id = tf.ones_like(doc_ids)
        segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

        input_mask = tf.ones_like(input_ids)

        features = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "input_mask": input_mask,
            "label_ids": label_ids
        }
        return features

    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=1000000)
    else:
        if max_eval_examples:
            # Use at most this number of examples (debugging only).
            dataset = dataset.take(max_eval_examples)
            # pass
            
    dataset = dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(
            batch_size=batch_size,
            padded_shapes={
                "input_ids": [seq_length],
                "segment_ids": [seq_length],
                "input_mask": [seq_length],
                "label_ids": []
            },
            padding_values={
                "input_ids": tf.cast(0, dtype=tf.int64),
                "segment_ids": tf.cast(0, dtype=tf.int64),
                "input_mask":tf.cast(0, dtype=tf.int64),
                "label_ids": tf.cast(0, dtype=tf.int64)
            }))

    return dataset