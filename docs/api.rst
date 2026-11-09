API Reference
=============

Paradigms
---------

.. currentmodule:: fnirsatlas.paradigms

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BaseFNIRS
   FNIRSClassification
   Emotion
   MotorExecution
   MotorImagery
   NBack
   FingerTapping
   MentalArithmetic

Datasets
--------

Specific datasets are documented in :doc:`Datasets </generated/datasets>`.

.. currentmodule:: fnirsatlas.datasets

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BaseFNIRSDataset
   BaseBIDSFNIRSDataset

Transformations
---------------

.. currentmodule:: fnirsatlas.pipelines

.. autosummary::
   :toctree: generated/
   :template: class.rst

   WindowedFeatureExtractor
   StandardScalerND
   ChannelSplitter
   UnsqueezeLayer

Deep learning
-------------

.. currentmodule:: fnirsatlas.pipelines.dl

.. autoclass:: FNIRSClassifier
   :show-inheritance:
