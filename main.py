from de_module import data_engineering_phase
from fe_module import feature_engineering_phase
from mne_module import modelling_and_evaluation_phase

de_phase_trigger = False
fe_phase_trigger = True
mae_phase_trigger = False


if de_phase_trigger:
    data_engineering_phase()

if fe_phase_trigger:
    feature_engineering_phase()

if mae_phase_trigger:
    modelling_and_evaluation_phase()
