#! /usr/bin/env python
#--eval_train --checkpoint_dir="./runs/1547379309/checkpoints/"
import tensorflow as tf
import numpy as np
from collections import Counter
import os
import data_helpers
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("label1_datas", "./test_datas_label1_final", "Data source for the label1 data.")
# tf.flags.DEFINE_string("label2_datas", "./test_datas_label2_final", "Data source for the label2 data.")
# tf.flags.DEFINE_string("label3_datas", "./test_datas_label3_final", "Data source for the label3 data.")

# tf.flags.DEFINE_string("train_label1_datas", "./data_label1_new", "Data source for the label1 data.")
# tf.flags.DEFINE_string("train_label2_datas", "./data_label2_new", "Data source for the label2 data.")
# tf.flags.DEFINE_string("train_label3_datas", "./data_label3_new", "Data source for the label3 data.")


tf.flags.DEFINE_string("train_data", "./final_2/train_gene_1.txt", "Data source for the label1 data.")
tf.flags.DEFINE_string("test_data", "./final_2/train_gene_test.txt", "Data source for the label1 data.")
tf.flags.DEFINE_integer("slot_1", 15, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")
tf.flags.DEFINE_integer("slot_0", 10, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1523649466/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("Eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.Eval_train:
    x_raw, y = data_helpers.load_data_and_labels(FLAGS.train_data)
    y_test = np.argmax(y, axis=1)
else:
    x_raw = ["我要金百万.", "昨天定的那一单"]
    y_test = [0, 1]

# x_train, y_trian = data_helpers.load_data_and_labels(FLAGS.train_label1_datas, FLAGS.train_label2_datas, FLAGS.train_label3_datas)
# print(wordlist.shape)

# Map data into vocabulary

voca = []
voca_slot = []
for x in x_raw:
    x_split = str(x).replace('\"', '').replace('\'', '').replace('[', '').replace(']', '').split(' ')
    spli_str = ""
    for i in x_split:
        spli_str = spli_str + " " + i
    # print(spli_str)
    voca.append(spli_str)
    # (x_split[0])

#     voca.append(spli_str)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(voca)))
# for i in voca_slot:
#     if FLAGS.slot_1 in i:
#         i.append(1)
#     else:
#         i.append(0)
print(np.array(x_test).shape)

print("\nEvaluating...\n")


# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print(checkpoint_file)
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # zero_num = graph.get_operation_by_name("zero_num").outputs[0]
        # embedding_W = graph.get_operation_by_name("embedding/embedding_W").outputs[0]
        # embedded_chars = graph.get_operation_by_name("embedding/embedded_chars").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        score = graph.get_operation_by_name("output/scores").outputs[0]
        # drop = graph.get_operation_by_name("pool").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        label1_pre = []
        label2_pre = []
        label3_pre = []
        for x_test_batch in batches:
            batch_predictions, sco = sess.run([predictions, score], {input_x: x_test_batch,
                                                                     dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
label1_ri = 0
label2_ri = 0
label3_ri = 0
if y_test is not None:
    print(all_predictions)
    output = open("test_reply", "a", encoding="UTF-8")
    for i in range(len(all_predictions)):
        # print(x_raw[i], y_test[i])
        # print(all_predictions[i])
        output.write(str(x_raw[i]))
        output.write("###")
        output.write(str(all_predictions[i]))
        output.write("\n")
    correct_predictions = float(sum(all_predictions == y_test))

    pre = 0
    rec = 0
    count_1 =  Counter(all_predictions)
    count_2 = Counter(y_test)
    for i in range(24):
        tp_fp = count_2[i*1.0]+1
        tp_fn = count_1[i*1.0]+1
        tp = 0
        weight = count_2[i*1.0]/(len(y_test))
        for j in range(len(y_test)):
            if all_predictions[j] == y_test[j] and y_test[j] == i:
                tp += 1
        print('precision of ' + str(i) +' '+ str(tp/tp_fp))
        print('recall of ' + str(i) +' '+ str(tp/tp_fn))
        pre += (tp/tp_fp)*weight
        rec += (tp/tp_fn)*weight
    print('precision of all ' + str(pre))
    print('recall of all ' + str(rec))

    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# count = Counter(all_predictions)
# if count[1.0] == 0: count[1.0] = 1
# if count[2.0] == 0: count[2.0] = 1


# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', encoding="utf-8") as f:
    csv.writer(f).writerows(predictions_human_readable)
