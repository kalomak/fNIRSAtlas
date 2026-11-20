"""
==============================
Export preprocessed fNIRS data
==============================

Load preprocessed fNIRS epochs as NumPy arrays for use outside MOABB
evaluations.
"""

# Authors: Kalle Mäkelä
#
# License: MIT

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification

# %%
# Load the data
# -------------
#
# The paradigm preprocesses the recordings and returns the epochs ``X``, the
# labels ``y``, and a metadata table with one row per epoch.

dataset = Bnci2014FnirsMA(subjects=[1])
paradigm = FNIRSClassification()
X, y, metadata = paradigm.get_data(dataset, subjects=[1])

# %%
# Inspect the arrays
# ------------------
#
# ``X`` is a NumPy array shaped ``(trials, chromophores, channels, timepoints)``.
# ``metadata`` is a pandas DataFrame with the subject, session, and run of each
# trial.

print(type(X).__name__, X.dtype, X.shape)
print(y[:5])
print(metadata.head())
