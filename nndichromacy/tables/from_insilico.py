import datajoint as dj

from insilico_stimuli.stimuli import GaborSet
import numpy as np

from insilico_stimuli.tables.templates import ExperimentPerUnitTemplate
from . import schema
from .from_mei import MEISelector, TrainedEnsembleModel


@schema
class OptimalMouseGaborPerUnitExperiment(ExperimentPerUnitTemplate, dj.Computed):
    """MEI table template.
    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    trained_model_table = TrainedEnsembleModel
    unit_table = MEISelector


@schema
class IsoResponseGabors(ExperimentPerUnitTemplate, dj.Computed):
    trained_model_table = TrainedEnsembleModel
    unit_table = MEISelector
    previous_experiment_table = OptimalMouseGaborPerUnitExperiment

    def get_stimulus_set(self, key):
        return GaborSet(**(self & key).fetch1("output")["full_gabor_config"])
