	�z�G�W@�z�G�W@!�z�G�W@	�œD%x�?�œD%x�?!�œD%x�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�z�G�W@��ʡE��?A�I+�W@YP��n��?*	      m@2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateJ+��?!�=��E@)+��η?1��K�C@:Preprocessing2F
Iterator::Model�~j�t��?!�Τ�љD@){�G�z�?1�V���*A@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�Q���?!�W0��
.@)�� �rh�?1�yi�>/-@:Preprocessing2S
Iterator::Model::ParallelMap����Mb�?!��x�w@)����Mb�?1��x�w@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{�G�zt?!�V���*@){�G�zt?1�V���*@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�&1��?!xO�n�E@)����Mbp?1��x�w�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����MbP?!��x�w�?)����MbP?1��x�w�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ʡE��?��ʡE��?!��ʡE��?      ��!       "      ��!       *      ��!       2	�I+�W@�I+�W@!�I+�W@:      ��!       B      ��!       J	P��n��?P��n��?!P��n��?R      ��!       Z	P��n��?P��n��?!P��n��?JCPU_ONLY