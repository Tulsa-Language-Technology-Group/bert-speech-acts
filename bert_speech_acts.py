# Fine-tune BERT with DailyDialog annotations.

import os
import sys
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from sklearn.model_selection import train_test_split

sys.path.append('models')
df = pd.read_csv('equalized.csv')

train_df, remaining = train_test_split(df, random_state=42, train_size=0.75, stratify=df.label.values)
valid_df, _ = train_test_split(remaining, random_state=42, train_size=0.075, stratify=remaining.label.values)

# efficient ingest pipeline
with tf.device('/cpu:0'):
    train_data = tf.data.Dataset.from_tensor_slices((train_df['text'].values, train_df['label'].values))
    valid_data = tf.data.Dataset.from_tensor_slices((valid_df.text.values, valid_df.label.values))
    for text, label in train_data.take(1):
        print(text)
        print(label)

# multiclass setup
label_list = [1, 2, 3, 4]
max_seq_length = 128
train_batch_size = 32

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
    example = classifier_data_lib.InputExample(guid=None,
                                               text_a=text.numpy(),
                                               text_b=None,
                                               label=label.numpy())
    feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)
    return feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id


# map featurized input to tf Dataset with Dataset.map(), which runs in graph mode
# turn graph tensors (that do not have a value) to regular tensors to enable eager execution
def to_feature_map(text, label):
    input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label],
                                                                  Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    label_id.set_shape([])

    x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }

    return x, label_id


# complete tf pipeline with per element mappings
with tf.device('/cpu:0'):
    train_data = (train_data.map(to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                  .shuffle(1000)
                  .batch(32, drop_remainder=True)
                  .prefetch(tf.data.experimental.AUTOTUNE))

    valid_data = (valid_data.map(to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                  .batch(32, drop_remainder=True)
                  .prefetch(tf.data.experimental.AUTOTUNE))


def create_model():
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_type_ids")
    # the pooled output is the contextualized input
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
    drop = tf.keras.layers.Dropout(0.4)(pooled_output)
    output = tf.keras.layers.Dense(20, activation='softmax', name='output')(drop)

    model = tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=output
    )
    return model


# compile model
model = create_model()

# multiclass setup
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

print(model.summary())

# save callbacks
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# save checkpoint
epochs = 2
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=1)

model.save_weights('./checkpoints/checkpoint')
