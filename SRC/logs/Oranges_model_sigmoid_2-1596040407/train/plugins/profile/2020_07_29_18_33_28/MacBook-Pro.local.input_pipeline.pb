	���(\OW@���(\OW@!���(\OW@	[y����?[y����?![y����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���(\OW@y�&1��?AB`��"CW@Y�MbX9�?*	      Z@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatX9��v��?!��N���=@)y�&1��?1�N��N�:@:Preprocessing2F
Iterator::Modelj�t��?!�N��N�D@)9��v���?1      9@:Preprocessing2S
Iterator::Model::ParallelMap�� �rh�?!��؉�X0@)�� �rh�?1��؉�X0@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�� �rh�?!��؉�X0@)�~j�t��?1;�;�'@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�Zd;�?!;�;�SM@);�O��n�?1��N��N!@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{�G�zt?!�;�;@){�G�zt?1�;�;@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap/�$��?!;�;14@)����Mbp?1O��N��@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!;�;�@)�~j�t�h?1;�;�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	y�&1��?y�&1��?!y�&1��?      ��!       "      ��!       *      ��!       2	B`��"CW@B`��"CW@!B`��"CW@:      ��!       B      ��!       J	�MbX9�?�MbX9�?!�MbX9�?R      ��!       Z	�MbX9�?�MbX9�?!�MbX9�?JCPU_ONLY