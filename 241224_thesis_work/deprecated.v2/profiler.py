import tensorflow as tf

# First create a profiler. See profiler tutorials for more details.
profiler = tf.profiler.Profiler(sess.graph)
run_meta = config_pb2.RunMetadata()
_ = sess.run(r1,
             options=config_pb2.RunOptions(
                 trace_level=config_pb2.RunOptions.FULL_TRACE),
             run_metadata=run_meta)
profiler.add_step(1, run_meta)

# Then Start advise.
profiler.advise()

# For one-shot API
tf.profiler.advise(
    sess.graph, run_meta=run_metadata)