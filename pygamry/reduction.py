import pandas as pd
import numpy as np
import time
import os

from .dtaq import DtaqOcv


class DtaqReduction(DtaqOcv):
    def __init__(self, config_file, **init_kw):
        # Read config file
        self.config_file = config_file
        self.config_df = pd.read_csv(config_file, index_col=0)

        #
        self.red_step_index = None
        self.red_step_start_time = None
        self.red_step_params = None
        self.reduction_complete = None

        self.flag_file_path = None

        super().__init__(**init_kw)

    def start_new_red_step(self):
        """Reset statuses etc. for new reduction step"""
        self.red_step_index += 1
        self.red_step_start_time = time.time()
        if self.red_step_index < len(self.config_df):
            self.red_step_params = self.config_df.loc[self.red_step_index]
        else:
            self.reduction_complete = True

    def evaluate_slope(self):
        """Get OCV slope"""
        t_sample = self.signal_params['t_sample']
        sample_window = int(60 * self.red_step_params['SlopeWindowMinutes'] / t_sample)
        # Get recent voltage within sample window
        v_sample = (self.data_array[-sample_window:, self.cook_columns.index('Vf')]).astype(float) * 1000  # mV
        t = (self.data_array[-sample_window:, self.cook_columns.index('Time')]).astype(float) / 3600  # hours
        # Linear fit
        fit = np.polyfit(t, v_sample, deg=1)

        return fit[0]

    def check_reduction_status(self):
        status = False
        elapsed_minutes = (time.time() - self.red_step_start_time) / 60
        if elapsed_minutes >= self.red_step_params['MinWaitTimeMinutes']:
            # If wait time has passed, check current OCV
            if self.data_array[-1, self.cook_columns.index('Vf')] >= self.red_step_params['MinimumOCV'] or \
                    elapsed_minutes >= self.red_step_params['MaxWaitTimeMinutes']:
                # If current OCV above threshold OR max time has passed, check slope
                slope = self.evaluate_slope()
                if slope < self.red_step_params['SlopeThresholdmVh']:
                    # If slope is below threshold, consider reduction step complete
                    status = True

        return status

    def _IGamryDtaqEvents_OnDataAvailable(self, this):
        super()._IGamryDtaqEvents_OnDataAvailable(this)

        # Check if reduction step is complete
        red_step_complete = self.check_reduction_status()

        if red_step_complete:
            print(f'step {self.red_step_index} complete')
            # Write empty flag file for LabVIEW
            flag_file = os.path.join(self.flag_file_path, f'Reduction_Step{self.red_step_index}_COMPLETE.txt')
            with open(flag_file, 'w+'):
                pass

            self.start_new_red_step()

        # If all steps complete, close handle to terminate PumpEvents
        if self.reduction_complete:
            # Fudge new_count for final write - only matters for write_mode continuous, in which case new_count will be
            # equal to total_points - last_write_index
            self.write_to_files(self.total_points - self._last_write_index, True)  # final write
            self.close_connection()

    def run(self, pstat, duration, t_sample, flag_file_path, max_iter=3, **run_kw):
        self.reduction_complete = False
        self.red_step_index = -1
        self.start_new_red_step()

        self.flag_file_path = flag_file_path

        # Run until reduction is complete
        iteration = 0
        while not self.reduction_complete:
            if iteration == 0:
                append_to_file = False
            else:
                append_to_file = True

            super().run(pstat, duration, t_sample, append_to_file=append_to_file, **run_kw)

            iteration += 1
            if iteration >= max_iter:
                break
