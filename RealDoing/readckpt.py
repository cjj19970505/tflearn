from tensorflow.python.tools import inspect_checkpoint as chkp
a = chkp.print_tensors_in_checkpoint_file(".\\RealDoing\\saved\\HalfWayModel.ckpt", tensor_name='', all_tensors=True)