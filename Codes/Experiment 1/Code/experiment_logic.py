import random
import csv
import os
import datetime
import itertools

class ExperimentLogic:
    STIMULI = ["DLM_2", "DLM_3", "ULM_L", "LM_L", "LM_C"]
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.participant_id = "test"
        self.current_block_index = 0
        self.block_order = []
        self.trials = []
        self.current_trial_index = 0
        self.results = []
        
        # Mapping from Stimulus Name to Device Demo Index
        # This will be populated by scanning or manual setting
        self.stimulus_indices = {} 

    def start_experiment(self, pid):
        self.participant_id = pid
        # Randomize block order
        blocks = ["Intensity", "Spatial"]
        random.shuffle(blocks)
        self.block_order = blocks
        self.current_block_index = 0
        self.generate_current_block_trials()

    def generate_current_block_trials(self):
        """
        Generates a sequence of 20 trials (all permutations of 2 stimuli).
        Ensures perfect balance of (A, B) and (B, A).
        """
        # Generate all ordered pairs (permutations) of length 2
        # For 5 stimuli, this is 5 * 4 = 20 pairs.
        full_trials = list(itertools.permutations(self.STIMULI, 2))
        
        # Shuffle the sequence so order is randomized
        random.shuffle(full_trials)
        
        self.trials = full_trials
        self.current_trial_index = 0
        self.results = []

    def get_current_block_type(self):
        if 0 <= self.current_block_index < len(self.block_order):
            return self.block_order[self.current_block_index]
        return None

    def is_experiment_complete(self):
        return self.current_block_index >= len(self.block_order)

    def next_block(self):
        self.current_block_index += 1
        if not self.is_experiment_complete():
            self.generate_current_block_trials()
            return True
        return False

    def save_trial_data(self, stimulus_a, stimulus_b, chosen_intensity, chosen_clarity, rt):
        filename = os.path.join(self.data_dir, f"{self.participant_id}.csv")
        file_exists = os.path.exists(filename)
        
        current_block_type = self.get_current_block_type()
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['ParticipantID', 'Timestamp', 'BlockIndex', 'BlockType', 'Trial', 'StimulusA', 'StimulusB', 'Chosen_Intensity', 'Chosen_Clarity', 'ReactionTime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'ParticipantID': self.participant_id,
                'Timestamp': datetime.datetime.now().isoformat(),
                'BlockIndex': self.current_block_index + 1,
                'BlockType': current_block_type,
                'Trial': self.current_trial_index + 1,
                'StimulusA': stimulus_a,
                'StimulusB': stimulus_b,
                'Chosen_Intensity': chosen_intensity,
                'Chosen_Clarity': chosen_clarity,
                'ReactionTime': rt
            })

    def get_current_trial(self):
        if self.current_trial_index < len(self.trials):
            return self.trials[self.current_trial_index]
        return None

    def next_trial(self):
        self.current_trial_index += 1
        return self.get_current_trial()

    def set_participant(self, pid):
        self.participant_id = pid
