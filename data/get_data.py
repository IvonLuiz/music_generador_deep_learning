from pathlib import Path
import tensorflow as tf


data_dir = Path('GTZAN')

if not data_dir.exists():
  tf.keras.utils.get_file(
      'music.zip',
      origin='https://osf.io/drjhb/download',
      extract=True,
      cache_dir='.', cache_subdir='GTZAN',
  )

