from data.datasets.univariate import WorldSpaceDensityData, VolumeDataStorage


class InputDataEmulator(object):

    def __init__(self, volume_data_storage: VolumeDataStorage, training_data: WorldSpaceDensityData, validation_data: WorldSpaceDensityData):
        self.volume_data_storage = volume_data_storage
        self.data = {'train': training_data, 'val': validation_data}

    def num_timekeyframes(self):
        # Currently definition of timesteps for volume data storage and latent-space generation seems to be coupled
        # might change in a later version of the code
        return self.volume_data_storage.num_timesteps()

    def num_ensembles(self):
        return self.volume_data_storage.num_members()

    def num_timesteps(self, mode):
        return self.data[mode].num_timesteps()

    def compute_actual_time_and_ensemble(self, timestep: int, ensemble:int, mode:str):
        actual_ensemble = self.volume_data_storage.ensemble_index[ensemble]
        actual_timestep = self.data[mode].timestep_index[timestep]
        return actual_timestep, actual_ensemble


