	V-���a@V-���a@!V-���a@	�r3Pˮ?�r3Pˮ?!�r3Pˮ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-���a@�~j�t��?A��Q��a@Yj�t��?*	      ^@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��~j�t�?!������?@)��~j�t�?1������?@:Preprocessing2F
Iterator::ModelJ+��?!�����jD@)���S㥛?1     �6@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat���S㥛?!     �6@)�~j�t��?1      4@:Preprocessing2S
Iterator::Model::ParallelMap�I+��?!UUUUUU2@)�I+��?1UUUUUU2@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�x?!      @)�~j�t�x?1      @:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!      @)�~j�t�h?1      @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�~j�t��?�~j�t��?!�~j�t��?      ��!       "      ��!       *      ��!       2	��Q��a@��Q��a@!��Q��a@:      ��!       B      ��!       J	j�t��?j�t��?!j�t��?R      ��!       Z	j�t��?j�t��?!j�t��?JCPU_ONLY