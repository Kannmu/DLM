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
        self.current_block = 1
        self.trials = []
        self.current_trial_index = 0
        self.results = []
        
        # Mapping from Stimulus Name to Device Demo Index
        # This will be populated by scanning or manual setting
        self.stimulus_indices = {} 

    def generate_sequence(self):
        """
        Generates a sequence of 20 trials (10 pairs * 2 orders).
        """
        pairs = list(itertools.combinations(self.STIMULI, 2))
        # pairs is [(A,B), (A,C), ...] length 10
        
        # Create full list of 20 trials (AB and BA)
        full_trials = []
        for a, b in pairs:
            full_trials.append((a, b))
            full_trials.append((b, a))
            
        # Shuffle
        random.shuffle(full_trials)
        self.trials = full_trials
        self.current_trial_index = 0
        self.results = []

    def save_trial_data(self, stimulus_a, stimulus_b, chosen_intensity, chosen_clarity, rt):
        filename = os.path.join(self.data_dir, f"{self.participant_id}.csv")
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['ParticipantID', 'Timestamp', 'Block', 'Trial', 'StimulusA', 'StimulusB', 'Chosen_Intensity', 'Chosen_Clarity', 'ReactionTime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'ParticipantID': self.participant_id,
                'Timestamp': datetime.datetime.now().isoformat(),
                'Block': self.current_block,
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
