## batch_generator.py
##
## 
##
## History:
##    1.0    29-Jun-2016     Initial version
##    1.1    12-Aug-2016     Changed input / output to key / value pairs
##                           Changed class from Dataset to Batch
##    1.2    30-Sep-2016     Changed class from Batch to BatchGenerator
##                           Added option to split batch generator into multiple
##                           batch generators

import random
import numpy as np

class BatchGenerator:
   """
   Object which produces batches from a provided dataset.
   """

   def __init__(self, shape_dict):
      """
      Setup a new generator for producing batches
      """

      self._shape_dict = shape_dict

      self._data = []
      self._data_keys = shape_dict.keys()
      self._shapes = {}

      for k in self._data_keys:
         self._shapes[k] = shape_dict[k]

      self._current_index = 0


   def add_sample(self, sample_dict):
      """
      Add a sample to the Dataset
      """

      self._data.append(sample_dict)


   def shuffle(self):
      """
      Shuffle the data
      """

      random.shuffle(self._data)


   def reset(self):
      """
      Wrap back around to the start of the list
      """

      self._current_index = 0


   def split(self, distribution):
      """
      Split the dataset in the batch generator into multiple generators

      distribution - Percentage of dataset for each batch generator.
                     This is assumed to sum to 1.0
      """

      # Create new batch generators
      batch_generators = [BatchGenerator(self._shape_dict) for _ in distribution]

      # Add each sample in the dataset to a random generator, as appropriate
      for sample in self._data:
         rnd = random.random()
         idx = 0

         # Determine which batch to add this to
         while rnd > distribution[idx]:
            rnd = rnd - distribution[idx]
            idx += 1
 
            # Just in case, assign to the last generator if needed
            if idx == len(distribution):
               idx = len(distribution) - 1
               break

         batch_generators[idx].add_sample(sample)

      return batch_generators


   def get_current_index(self):
      """
      Get the current position in the batch
      """

      return self._current_index


   def set_index(self, index):
      """
      """

      self._current_index = index


   def get_batch(self, batch_size):
      """
      Return an batch of input / output pairs
      """

      size = min(len(self._data) - self._current_index, batch_size)

      data = {}
      for k in self._data_keys:
         data[k] = np.zeros((size,) + self._shapes[k])

         for i in range(size):
            data[k][i,:] = self._data[i + self._current_index][k][:]

      self._current_index = self._current_index + size

      data['batch_size'] = size

      return data


   def num_samples(self):
      """
      The total number of samples in the batch
      """

      return len(self._data)


