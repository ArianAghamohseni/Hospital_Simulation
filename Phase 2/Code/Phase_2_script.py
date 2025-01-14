import random
import math
import pandas as pd
import numpy as np
import time
from collections import deque
import heapq
import logging
import scipy.stats as st
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, filename='hospital_simulation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class HospitalSimulation:
    def __init__(self, simulation_time,
                 arrival_rate_urgent=4/60,
                 arrival_rate_regular=1/60,  # Ensure this parameter is present
                 prob_hard_surgery=0.05,
                 surgery_duration_mean={'s': 30.222215, 'm': 74.54242222, 'h': 242.0312},
                 surgery_duration_std={'s': 4.958886914, 'm': 9.953222836, 'h': 63.27455578},
                 preoperative_beds=25,
                 general_ward_beds=40):
        self.simulation_time = simulation_time
        self.clock = 0
        self.future_event_list = []    # FEL (Priority Queue)
        self.trace_list = []           # Trace for Excel

        # State variables
        self.state = {
            'LEn': preoperative_beds,  # **Set from parameter**
            'LQn': 0,                   # Queue for Preoperative Admission
            'LEe': 10,                  # Emergency Room beds
            'LQe': 0,                   # Queue for Emergency Room (ambulance queue up to 10)
            'lel': 3,                   # Laboratory beds
            'lqln': 0,                  # Queue for Lab from Preoperative
            'lqle': 0,                  # Queue for Lab from ER
            'LEs': 50,                  # Operating Rooms beds
            'lqsn': 0,                  # Queue for OR regular patients
            'lqse': 0,                  # Queue for OR urgent patients
            'nd': 0,                    # Number of deaths in OR
            'E': 1,                     # Power status (1 = on, 0 = off)
            'lec': 5,                   # CCU beds
            'lqc': 0,                   # Queue for CCU
            'lei': 10,                  # ICU beds
            'lqi': 0,                   # Queue for ICU
            'leg': general_ward_beds,   # **Set from parameter**
            'lqg': 0,                   # Queue for General Ward
            'ns': 0,                    # Number of discharged patients
            'R': 0,                     # Number of reoperations
            'NOLE': 0,                  # Number of patients unable to enter Lab from ER
            'NOLN': 0,                  # Number of patients unable to enter Lab from Preoperative Admission
        }

        self.next_patient_id = 1

        # Data collection structures
        self.data = {
            'patients': {},
            'statistics': {
                'length_of_stay': [],
                'ER_full_count': 0,
                'ER_total': 0,
                'queues': {
                    'ER': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'Preoperative': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'Lab_Preoperative': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'Lab_ER': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'OR_Regular': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'OR_Urgent': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'ICU': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'CCU': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                    'General_Ward': {'max': 0, 'total': 0, 'count': 0, 'total_wait_time': 0, 'count_wait_time': 0},
                },
                'reoperations': 0,
                'complex_surgeries': 0,
                'bed_utilization_time': {
                    'Preoperative': 0,
                    'ER': 0,
                    'Lab': 0,
                    'OR': 0,
                    'General_Ward': 0,
                    'ICU': 0,
                    'CCU': 0
                },
                'bed_utilization': {},
                'Num_deaths': 0,
                'NOLE': 0,  # Total number of times Lab from ER was full
                'NOLN': 0,  # Total number of times Lab from Preoperative Admission was full
            },
            'trace': []  # For Excel output
        }

        # Queues
        self.queues = {
            'ER': deque(),
            'Preoperative': deque(),
            'Lab_Preoperative': deque(),
            'Lab_ER': deque(),
            'OR_Regular': deque(),
            'OR_Urgent': deque(),
            'ICU': deque(),
            'CCU': deque(),
            'General_Ward': deque()
        }

        self.queue_arrival_times = {
            'ER': deque(),
            'Preoperative': deque(),
            'Lab_Preoperative': deque(),
            'Lab_ER': deque(),
            'OR_Regular': deque(),
            'OR_Urgent': deque(),
            'ICU': deque(),
            'CCU': deque(),
            'General_Ward': deque()
        }

        # Bed usage tracker
        self.bed_usage = {
            'Preoperative': 0,
            'ER': 0,
            'Lab': 0,
            'OR': 0,
            'General_Ward': 0,
            'ICU': 0,
            'CCU': 0
        }

        # Tracking last event time for bed utilization
        self.last_event_time = 0

        # Schedule initial events
        self.schedule_event('HE', self.clock, None)
        self.schedule_power_outages()

        # Variable Parameters for Sensitivity Analysis
        self.arrival_rate_urgent = arrival_rate_urgent
        self.arrival_rate_regular = arrival_rate_regular
        self.prob_hard_surgery = prob_hard_surgery
        self.surgery_duration_mean = surgery_duration_mean
        self.surgery_duration_std = surgery_duration_std

    #####################################################################################################################################
    # Distributions
    @staticmethod
    def exponential(rate: float) -> float:
        """
        Generates a random variate from an exponential distribution.

        Args:
            rate (float): Rate parameter (λ) of the exponential distribution.

        Returns:
            float: A random variate following the exponential distribution.
        """
        if rate <= 0:
            raise ValueError("Rate parameter must be positive for exponential distribution.")
        return np.random.exponential(scale=1/rate)

    @staticmethod
    def triangular_dist(low: float, mode: float, high: float) -> float:
        """
        Generates a random variate from a triangular distribution.

        Args:
            low (float): Lower limit of the distribution.
            mode (float): The peak or most likely value.
            high (float): Upper limit of the distribution.

        Returns:
            float: A random variate following the triangular distribution.
        """
        if not (low <= mode <= high):
            raise ValueError("Ensure that low <= mode <= high for triangular distribution.")
        return np.random.triangular(left=low, mode=mode, right=high)

    @staticmethod
    def uniform_dist(low: float, high: float) -> float:
        """
        Generates a random variate from a uniform distribution.

        Args:
            low (float): Lower bound of the distribution.
            high (float): Upper bound of the distribution.

        Returns:
            float: A random variate following the uniform distribution.
        """
        if low > high:
            raise ValueError("Lower bound must be less than or equal to upper bound for uniform distribution.")
        return np.random.uniform(low=low, high=high)

    @staticmethod
    def normal_dist(mean: float, std: float) -> float:
        """
        Generates a random variate from a normal (Gaussian) distribution.

        Args:
            mean (float): Mean of the distribution.
            std (float): Standard deviation of the distribution.

        Returns:
            float: A random variate following the normal distribution.
        """
        if std < 0:
            raise ValueError("Standard deviation must be non-negative for normal distribution.")
        return np.random.normal(loc=mean, scale=std)

    @staticmethod
    def poisson_arrival_time(rate: float) -> float:
        """
        Generates inter-arrival time for a Poisson process using an exponential distribution.

        Args:
            rate (float): Rate parameter (λ) of the Poisson process.

        Returns:
            float: Inter-arrival time following an exponential distribution.
        """
        return HospitalSimulation.exponential(rate)

    #####################################################################################################################################
    # Helper functions

    def run_simulation(self):
        # Writing this function for the repetitions to see where it crashed
        start_time = time.time()
        try:
            while self.future_event_list and self.clock < self.simulation_time:
                event_time, event_type, patient_id = heapq.heappop(self.future_event_list)

                # Update bed utilization before processing the event
                self.update_bed_utilization(event_time)

                self.clock = event_time
                logging.info(f"Processing event '{event_type}' for patient {patient_id} at time {self.clock}")
                self.process_event(event_type, patient_id)
                self.collect_statistics()

            # Final update for the last interval
            self.update_bed_utilization(self.simulation_time)
            self.calculate_bed_utilization_time()
            end_time = time.time()
            logging.info(f"Simulation completed in {end_time - start_time:.2f} seconds.")
            print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
        except RecursionError as e:
            logging.critical(f"Simulation crashed due to recursion error: {e}")
            print(f"Simulation crashed due to recursion error: {e}")
        except Exception as e:
            logging.critical(f"Simulation crashed: {e}")
            print(f"Simulation crashed: {e}")

    def update_bed_utilization(self, current_time):
        elapsed_time = current_time - self.last_event_time
        for bed_type in self.data['statistics']['bed_utilization_time']:
            self.data['statistics']['bed_utilization_time'][bed_type] += self.bed_usage[bed_type] * elapsed_time
        self.last_event_time = current_time

    def process_event(self, event_type, patient_id):
        try:
            if event_type == 'HE':
                self.event_HE(patient_id)
            elif event_type == 'LE_ER':
                self.event_LE_ER(patient_id)
            elif event_type == 'LE_Preoperative':
                self.event_LE_Preoperative(patient_id)
            elif event_type == 'LL':
                self.event_LL(patient_id)
            elif event_type == 'SE':
                self.event_SE(patient_id)
            elif event_type == 'SE_COMPLETED':
                self.event_SE_COMPLETED(patient_id)
            elif event_type == 'CE':
                self.event_CE(patient_id)
            elif event_type == 'IE':
                self.event_IE(patient_id)
            elif event_type == 'GE':
                self.event_GE(patient_id)
            elif event_type == 'GL':
                self.event_GL(patient_id)
            elif event_type == 'PO':
                self.event_PO()
            elif event_type == 'PI':
                self.event_PI()
            elif event_type == 'ME':
                self.event_ME(patient_id)
            elif event_type == 'ICU_Discharge':
                self.event_ICU_Discharge(patient_id)
            elif event_type == 'CCU_Discharge':
                self.event_CCU_Discharge(patient_id)
            else:
                logging.warning(f"Unknown event type: {event_type}")
        except Exception as e:
            logging.error(f"Error processing event '{event_type}' for patient {patient_id}: {e}")

    #####################################################################################################################################
    # Start of events

    def event_HE(self, patient_id=None):
        """
        Hospital Entry Event.
        Handles both new patient arrivals and re-entries of existing patients.

        Args:
            patient_id (int, optional): Existing patient ID for re-entry. Defaults to None.
        """
        if patient_id is None:
            # Handle new patient arrivals
            if random.random() < 0.25:  # 25% urgent
                patient_type = 'urgent'
                arrival_rate = 4 / 60
            else:
                patient_type = 'regular'
                arrival_rate = 1 / 60

            # Determine if patient arrives as a group
            group_probability = 0.005
            if patient_type == 'urgent' and random.random() < group_probability:
                group_size = random.randint(2, 5)
                logging.info(f"Group arrival of size {group_size} for {patient_type} patients")
                for _ in range(group_size):
                    self.add_patient(patient_type, group=True)
            else:
                # Single patient
                self.add_patient(patient_type, group=False)
        else:
            # Handle re-entry of existing patient
            patient = self.data['patients'][patient_id]
            patient_type = patient['type']  # Should be 'urgent' after reclassification
            group = patient['group']  # Preserve group status if needed

            self.add_patient(patient_type, group, patient_id=patient_id)

        # Schedule next HE event for new arrivals only
        if patient_id is None:
            inter_arrival = self.poisson_arrival_time(arrival_rate)
            next_HE_time = self.clock + inter_arrival
            self.schedule_event('HE', next_HE_time, None)

    def add_patient(self, patient_type, group, patient_id=None):
        """
        Adds a new patient to the system or re-enters an existing patient.

        Args:
            patient_type (str): Type of the patient ('urgent' or 'regular').
            group (bool): Whether the patient arrives as part of a group.
            patient_id (int, optional): Existing patient ID for re-entry. Defaults to None.
        """
        if patient_id is None:
            # Assign a new patient ID
            patient_id = self.next_patient_id
            self.next_patient_id += 1
            is_new_patient = True
        else:
            is_new_patient = False  # Re-entering existing patient

        # Initialize or update patient attributes
        if is_new_patient:
            patient = {
                'id': patient_id,
                'type': patient_type,
                'group': group,
                'arrival_time': self.clock,
                'L': None,  # Surgery complexity: 's', 'm', 'h'
                'M': None,  # Destination after surgery
                'O': None,  # Operation type, e.g., 'HS' for open-heart
                'w': 0,      # Condition worsening flag: 1 if worsens, else 0
                'Q': None,
                'start_labs_time': None,
                'end_labs_time': None,
                'surgery_time': None,
                'surgery_type': None,
                'reoperation': False,
                'death': False,
                'discharge_time': None,
                'reoperation_time': None
            }
            self.data['patients'][patient_id] = patient
            logging.info(f"Added new patient {patient_id}: Type={patient_type}, Group={group}")
        else:
            # Re-entering existing patient
            patient = self.data['patients'][patient_id]
            patient['type'] = patient_type
            patient['group'] = group
            patient['arrival_time'] = self.clock
            patient['w'] = 0
            patient['death'] = False
            logging.info(f"Re-entered patient {patient_id}: Type={patient_type}, Group={group}")

        if patient_type == 'urgent':
            if self.state['LEe'] > 0:
                self.state['LEe'] -= 1
                self.bed_usage['ER'] += 1
                logging.info(f"Patient {patient_id} assigned to ER. Remaining ER beds: {self.state['LEe']}")
                # Schedule LE_ER (Leave ER event)
                admin_delay = 10
                self.schedule_event('LE_ER', self.clock + admin_delay, patient_id)
            else:
                if self.state['LQe'] < 10:
                    self.state['LQe'] += 1
                    logging.info(f"Patient {patient_id} queued in ER. Queue length: {self.state['LQe']}")
                    self.queue_arrival_times['ER'].append(self.clock)  # For waiting time calculation
                    self.queues['ER'].append(patient_id)  # Enqueue to ER queue
                    # Update statistics
                    if self.state['LQe'] > self.data['statistics']['queues']['ER']['max']:
                        self.data['statistics']['queues']['ER']['max'] = self.state['LQe']
                else:
                    # Queue is full, patient exits
                    self.state['ns'] += 1
                    logging.warning(f"Patient {patient_id} exited due to full ER queue.")
        else:
            if self.state['LEn'] > 0:
                self.state['LEn'] -= 1
                self.bed_usage['Preoperative'] += 1
                logging.info(f"Patient {patient_id} assigned to Preoperative Admission. Remaining beds: {self.state['LEn']}")
                # Schedule LE_Preoperative
                admin_delay = 60
                self.schedule_event('LE_Preoperative', self.clock + admin_delay, patient_id)
            else:
                # Increment preoperative queue
                self.queues['Preoperative'].append(patient_id)
                self.state['LQn'] += 1
                logging.info(f"Patient {patient_id} queued in Preoperative Admission. Queue length: {self.state['LQn']}")
                self.queue_arrival_times['Preoperative'].append(self.clock)  # For waiting time calculation
                # Update statistics
                if self.state['LQn'] > self.data['statistics']['queues']['Preoperative']['max']:
                    self.data['statistics']['queues']['Preoperative']['max'] = self.state['LQn']

    def release_bed(self, bed_type, patient_id):
        if self.bed_usage[bed_type] > 0:
            self.bed_usage[bed_type] -= 1
            if bed_type == 'ER':
                self.state['LEe'] += 1
            elif bed_type == 'Preoperative':
                self.state['LEn'] += 1
            elif bed_type == 'Lab':
                self.state['lel'] += 1
            elif bed_type == 'OR':
                self.state['LEs'] += 1
            elif bed_type == 'General_Ward':
                self.state['leg'] += 1
            elif bed_type == 'ICU':
                self.state['lei'] += 1
            elif bed_type == 'CCU':
                self.state['lec'] += 1
            logging.info(f"Patient {patient_id} released from {bed_type}. Available {bed_type} beds: "
                        f"{self.state['LEe'] if bed_type == 'ER' else self.state['LEn'] if bed_type == 'Preoperative' else self.state['lel'] if bed_type == 'Lab' else self.state['LEs'] if bed_type == 'OR' else self.state['leg'] if bed_type == 'General_Ward' else self.state['lei'] if bed_type == 'ICU' else self.state['lec']}.")
        else:
            logging.error(f"Attempted to release {bed_type} bed for patient {patient_id}, but no beds are occupied.")

    def event_LE_ER(self, patient_id):
        """Lab Entry from ER Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Patient {patient_id} is processing LE_ER (Lab Entry from ER) event at time {self.clock}")

        # Handle ER to Lab transition
        patient['start_labs_time'] = self.clock
        logging.info(f"Patient {patient_id} entered Lab from ER at time {self.clock}")

        if self.state['lel'] > 0:
            self.state['lel'] -= 1
            self.bed_usage['Lab'] += 1  # Assign Lab bed
            self.state['NOLE'] += 1
            logging.info(f"Patient {patient_id} assigned to Lab. Remaining Lab beds: {self.state['lel']}")
            # Schedule LL (Leave Lab event)
            test_duration = self.uniform_dist(28, 32)
            self.schedule_event('LL', self.clock + test_duration, patient_id)
        else:
            # Increment lab queue from ER
            self.queues['Lab_ER'].append(patient_id)
            self.state['lqle'] += 1
            logging.info(f"Patient {patient_id} queued in Lab_ER. Queue length: {self.state['lqle']}.")
            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['Lab_ER'].append(self.clock)
            # Update statistics
            if self.state['lqle'] > self.data['statistics']['queues']['Lab_ER']['max']:
                self.data['statistics']['queues']['Lab_ER']['max'] = self.state['lqle']

    def event_LE_Preoperative(self, patient_id):
        """Lab Entry from Preoperative Admission Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Patient {patient_id} is processing LE_Preoperative (Lab Entry from Preoperative Admission) event at time {self.clock}")

        # Handle Preoperative Admission to Lab transition
        patient['start_labs_time'] = self.clock
        logging.info(f"Patient {patient_id} entered Lab from Preoperative Admission at time {self.clock}")

        if self.state['lel'] > 0:
            self.state['lel'] -= 1
            self.bed_usage['Lab'] += 1
            self.state['NOLN'] += 1
            logging.info(f"Patient {patient_id} assigned to Lab. Remaining Lab beds: {self.state['lel']}")
            # Schedule LL (Leave Lab event)
            test_duration = self.uniform_dist(28, 32)
            self.schedule_event('LL', self.clock + test_duration, patient_id)
        else:
            # Increment lab queue from Preoperative
            self.queues['Lab_Preoperative'].append(patient_id)
            self.state['lqln'] += 1
            logging.info(f"Patient {patient_id} queued in Lab_Preoperative. Queue length: {self.state['lqln']}.")
            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['Lab_Preoperative'].append(self.clock)
            # Update statistics
            if self.state['lqln'] > self.data['statistics']['queues']['Lab_Preoperative']['max']:
                self.data['statistics']['queues']['Lab_Preoperative']['max'] = self.state['lqln']

    def event_LL(self, patient_id):
        """Leave Lab Event"""
        patient = self.data['patients'][patient_id]
        patient['end_labs_time'] = self.clock
        self.bed_usage['Lab'] -= 1
        self.state['lel'] += 1
        logging.info(f"Patient {patient_id} left Lab at time {self.clock}")

        if patient['type'] == 'urgent':
            delay = self.triangular_dist(5, 75, 100)
            self.schedule_event('SE', self.clock + delay, patient_id)
            logging.info(f"Patient {patient_id} scheduled for Surgery Entry at {self.clock + delay}")
        else:
            delay = 60
            self.schedule_event('SE', self.clock + delay, patient_id)
            logging.info(f"Patient {patient_id} scheduled for Surgery Entry at {self.clock + delay}")

        # Priority: Lab_ER > Lab_Preoperative
        if self.queues['Lab_ER']:
            next_patient_id = self.queues['Lab_ER'].popleft()
            self.state['lqle'] -= 1
            self.bed_usage['Lab'] += 1
            logging.info(f"Patient {next_patient_id} moved from Lab_ER to Lab. Remaining Lab_ER queue: {self.state['lqle']}")
            # Calculate waiting time
            if self.queue_arrival_times['Lab_ER']:
                enqueue_time = self.queue_arrival_times['Lab_ER'].popleft()
                wait_time = self.clock - enqueue_time
                self.data['statistics']['queues']['Lab_ER']['total_wait_time'] += wait_time
                self.data['statistics']['queues']['Lab_ER']['count_wait_time'] += 1
                logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in Lab_ER queue.")
            else:
                logging.error(f"Lab_ER arrival times deque empty for patient {next_patient_id}.")
            # Schedule LL for the next Lab_ER patient
            test_duration = self.uniform_dist(28, 32)
            self.schedule_event('LL', self.clock + test_duration, next_patient_id)
        elif self.queues['Lab_Preoperative']:
            next_patient_id = self.queues['Lab_Preoperative'].popleft()
            self.state['lqln'] -= 1
            self.bed_usage['Lab'] += 1
            logging.info(f"Patient {next_patient_id} moved from Lab_Preoperative to Lab. Remaining Lab_Preoperative queue: {self.state['lqln']}")
            # Calculate waiting time
            if self.queue_arrival_times['Lab_Preoperative']:
                enqueue_time = self.queue_arrival_times['Lab_Preoperative'].popleft()
                wait_time = self.clock - enqueue_time
                self.data['statistics']['queues']['Lab_Preoperative']['total_wait_time'] += wait_time
                self.data['statistics']['queues']['Lab_Preoperative']['count_wait_time'] += 1
                logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in Lab_Preoperative queue.")
            else:
                logging.error(f"Lab_Preoperative arrival times deque empty for patient {next_patient_id}.")
            # Schedule LL for the next Lab_Preoperative patient
            test_duration = self.uniform_dist(28, 32)
            self.schedule_event('LL', self.clock + test_duration, next_patient_id)

    def event_SE(self, patient_id):
        """Surgery Entry Event"""
        patient = self.data['patients'][patient_id]
        # Determine surgery type based on historical data, phase 2
        surgery_type_prob = random.random()
        if surgery_type_prob < 0.5:
            patient['L'] = 's'  # Simple
            surgery_duration = self.normal_dist(30.222215, 4.958886914)
        elif surgery_type_prob < 0.95:
            patient['L'] = 'm'  # Moderate
            surgery_duration = self.normal_dist(74.54242222, 9.953222836)
        else:
            patient['L'] = 'h'  # Hard
            surgery_duration = self.normal_dist(242.0312, 63.27455578)
            self.data['statistics']['complex_surgeries'] += 1
            logging.info(f"Patient {patient_id} assigned to hard surgery.")

        # Assign operation type 'O' based on surgery type
        if patient['L'] == 'h':
            # 25% chance of open-heart surgery
            patient['O'] = 'HS' if random.random() < 0.25 else 'NHS'
        else:
            patient['O'] = None

        # Ensure surgery_duration is positive
        surgery_duration = max(surgery_duration, 1)

        patient['surgery_time'] = surgery_duration
        patient['surgery_type'] = patient['L']
        logging.info(f"Patient {patient_id} undergoing surgery type {patient['L']} with duration {surgery_duration:.2f} minutes")

        # Check if Operating Room is available
        if self.state['LEs'] > 0:
            self.state['LEs'] -= 1
            self.bed_usage['OR'] += 1
            logging.info(f"Patient {patient_id} assigned to Operating Room. Remaining OR beds: {self.state['LEs']}")
            # Schedule surgery completion
            self.schedule_event('SE_COMPLETED', self.clock + surgery_duration, patient_id)
        else:
            # Increment surgery queue
            if patient['type'] == 'urgent':
                self.queues['OR_Urgent'].append(patient_id)
                self.state['lqse'] += 1
                logging.info(f"Patient {patient_id} queued in OR_Urgent. Queue length: {self.state['lqse']}")
                # For waiting time calculation
                self.queue_arrival_times['OR_Urgent'].append(self.clock)
                # Update statistics
                if self.state['lqse'] > self.data['statistics']['queues']['OR_Urgent']['max']:
                    self.data['statistics']['queues']['OR_Urgent']['max'] = self.state['lqse']
            else:
                self.queues['OR_Regular'].append(patient_id)
                self.state['lqsn'] += 1
                logging.info(f"Patient {patient_id} queued in OR_Regular. Queue length: {self.state['lqsn']}")
                # For waiting time calculation
                self.queue_arrival_times['OR_Regular'].append(self.clock)
                # Update statistics
                if self.state['lqsn'] > self.data['statistics']['queues']['OR_Regular']['max']:
                    self.data['statistics']['queues']['OR_Regular']['max'] = self.state['lqsn']

    def event_SE_COMPLETED(self, patient_id):
        """Surgery Completed Event"""
        patient = self.data['patients'][patient_id]
        self.release_bed('OR', patient_id)

        # Determine post-surgery destination based on surgery type
        if patient['L'] == 's':
            # Simple surgery: transfer to General Ward
            if self.state['leg'] > 0:
                self.state['leg'] -= 1
                self.bed_usage['General_Ward'] += 1
                logging.info(f"Patient {patient_id} transferred to General Ward. Remaining beds: {self.state['leg']}")
                # Schedule discharge after stay
                discharge_time = self.clock + 2880  # 2 days in minutes
                self.schedule_event('GL', discharge_time, patient_id)
            else:
                # Increment General Ward queue
                self.queues['General_Ward'].append(patient_id)
                self.state['lqg'] += 1
                logging.info(f"Patient {patient_id} queued in General_Ward. Queue length: {self.state['lqg']}")
                # Record queue entry time for waiting time calculation
                self.queue_arrival_times['General_Ward'].append(self.clock)
                # Update statistics
                if self.state['lqg'] > self.data['statistics']['queues']['General_Ward']['max']:
                    self.data['statistics']['queues']['General_Ward']['max'] = self.state['lqg']
        elif patient['L'] == 'm':
            # Moderate surgery: transfer based on probabilities
            transfer_prob = random.random()
            if transfer_prob < 0.7:
                destination = 'General_Ward'
            elif transfer_prob < 0.8:
                destination = 'ICU'
            else:
                destination = 'CCU'

            patient['M'] = destination
            logging.info(f"Patient {patient_id} surgery type 'm' transferred to {destination}")

            if destination == 'General_Ward':
                if self.state['leg'] > 0:
                    self.state['leg'] -= 1
                    self.bed_usage['General_Ward'] += 1
                    # Schedule discharge after stay
                    stay_duration = self.exponential(1/50)
                    discharge_time = self.clock + stay_duration
                    self.schedule_event('GL', discharge_time, patient_id)
                    logging.info(f"Patient {patient_id} scheduled to discharge at {discharge_time}")
                else:
                    # Increment General Ward queue
                    self.queues['General_Ward'].append(patient_id)
                    self.state['lqg'] += 1
                    logging.info(f"Patient {patient_id} queued in General_Ward. Queue length: {self.state['lqg']}")
                    # Record queue entry time for waiting time calculation
                    self.queue_arrival_times['General_Ward'].append(self.clock)
                    # Update statistics
                    if self.state['lqg'] > self.data['statistics']['queues']['General_Ward']['max']:
                        self.data['statistics']['queues']['General_Ward']['max'] = self.state['lqg']
            elif destination == 'ICU':
                if self.state['lei'] > 0:
                    self.state['lei'] -= 1
                    self.bed_usage['ICU'] += 1
                    # Schedule ICU discharge
                    stay_duration = self.exponential(1/25)
                    discharge_time = self.clock + stay_duration
                    self.schedule_event('ICU_Discharge', discharge_time, patient_id)
                    logging.info(f"Patient {patient_id} scheduled to discharge from ICU at {discharge_time}")
                else:
                    # Increment ICU queue
                    self.queues['ICU'].append(patient_id)
                    self.state['lqi'] += 1
                    logging.info(f"Patient {patient_id} queued in ICU. Queue length: {self.state['lqi']}")
                    # Record queue entry time for waiting time calculation
                    self.queue_arrival_times['ICU'].append(self.clock)
                    # Update statistics
                    if self.state['lqi'] > self.data['statistics']['queues']['ICU']['max']:
                        self.data['statistics']['queues']['ICU']['max'] = self.state['lqi']
            elif destination == 'CCU':
                if self.state['lec'] > 0 and self.state['E'] == 1:
                    self.state['lec'] -= 1
                    self.bed_usage['CCU'] += 1
                    # Schedule CCU discharge
                    stay_duration = self.exponential(1/25)
                    discharge_time = self.clock + stay_duration
                    self.schedule_event('CCU_Discharge', discharge_time, patient_id)
                    logging.info(f"Patient {patient_id} scheduled to discharge from CCU at {discharge_time}")
                else:
                    # Increment CCU queue
                    self.queues['CCU'].append(patient_id)
                    self.state['lqc'] += 1
                    logging.info(f"Patient {patient_id} queued in CCU. Queue length: {self.state['lqc']}")
                    # Record queue entry time for waiting time calculation
                    self.queue_arrival_times['CCU'].append(self.clock)
                    # Update statistics
                    if self.state['lqc'] > self.data['statistics']['queues']['CCU']['max']:
                        self.data['statistics']['queues']['CCU']['max'] = self.state['lqc']
        elif patient['L'] == 'h':
            # Hard surgery: Handle death and condition worsening
            death_prob = 0.1
            if random.random() < death_prob:
                # Patient dies
                patient['death'] = True
                self.state['nd'] += 1
                self.data['statistics']['Num_deaths'] += 1
                logging.warning(f"Patient {patient_id} died during hard surgery.")
                # Schedule Morgue Entry
                s8 = self.uniform_dist(60, 120)
                self.schedule_event('ME', self.clock + s8, patient_id)
                logging.info(f"Patient {patient_id} scheduled to enter morgue at {self.clock + s8}")

                # Handle queue logic based on patient type
                if patient['type'] == 'urgent':
                    if self.state['LQe'] < 10:
                        self.state['LQe'] += 1
                        self.schedule_event('HE', self.clock, patient_id)
                        logging.info(f"Patient {patient_id} re-entered hospital as urgent after death.")
                    else:
                        self.state['LEe'] += 1
                        self.state['ns'] += 1
                        logging.warning(f"ER queue full. Patient {patient_id} exited after death.")
                else:
                    if self.state['LQn'] > 0:
                        self.state['LQn'] -= 1
                        self.schedule_event('HE', self.clock, patient_id)
                        logging.info(f"Patient {patient_id} re-entered hospital as regular after death.")
                    else:
                        self.state['LEn'] += 1
                        self.state['ns'] += 1
                        logging.warning(f"Preoperative queue empty. Patient {patient_id} exited after death.")
            else:
                # Patient survives hard surgery, check for condition worsening
                if patient['O'] == 'HS':
                    worsening_prob = 0
                else:
                    worsening_prob = 0.01

                if random.random() < worsening_prob:
                    # Condition worsens
                    patient['w'] = 1
                    patient['type'] = 'urgent'
                    self.data['statistics']['reoperations'] += 1  # Increment reoperations
                    logging.info(f"Patient {patient_id}'s condition worsened after hard surgery. Scheduling reoperation.")

                    # Release OR bed and schedule reoperation
                    self.release_bed('OR', patient_id)
                    # Schedule Surgery Entry Event after s8*
                    s8 = self.uniform_dist(60, 120)
                    self.schedule_event('SE', self.clock + s8, patient_id)
                    logging.info(f"Patient {patient_id} scheduled for reoperation at {self.clock + s8}")

                    # Handle queue logic based on patient type (now urgent)
                    if patient['type'] == 'urgent':
                        if self.state['LQe'] < 10:
                            self.state['LQe'] += 1
                            self.schedule_event('HE', self.clock, patient_id)
                            logging.info(f"Patient {patient_id} re-entered hospital as urgent for reoperation.")
                        else:
                            self.state['LEe'] += 1
                            self.state['ns'] += 1
                            logging.warning(f"ER queue full. Patient {patient_id} exited after condition worsening.")
                    else:
                        if self.state['LQn'] > 0:
                            self.state['LQn'] -= 1
                            self.schedule_event('HE', self.clock, patient_id)
                            logging.info(f"Patient {patient_id} re-entered hospital as regular for reoperation.")
                        else:
                            self.state['LEn'] += 1
                            self.state['ns'] += 1
                            logging.warning(f"Preoperative queue empty. Patient {patient_id} exited after condition worsening.")
                else:
                    # No condition worsening, transfer to appropriate ward
                    if patient['O'] == 'HS':
                        # Open-heart surgery: transfer to CCU
                        destination = 'CCU'
                    else:
                        # Other hard surgeries: transfer to ICU
                        destination = 'ICU'

                    patient['M'] = destination
                    logging.info(f"Patient {patient_id} transferred to {destination} after hard surgery.")

                    if destination == 'CCU':
                        if self.state['lec'] > 0 and self.state['E'] == 1:
                            self.state['lec'] -= 1
                            self.bed_usage['CCU'] += 1
                            # Schedule CCU discharge
                            stay_duration = self.exponential(1/25)
                            discharge_time = self.clock + stay_duration
                            self.schedule_event('CCU_Discharge', discharge_time, patient_id)
                            logging.info(f"Patient {patient_id} scheduled to discharge from CCU at {discharge_time}")
                        else:
                            # Increment CCU queue
                            self.queues['CCU'].append(patient_id)
                            self.state['lqc'] += 1
                            logging.info(f"Patient {patient_id} queued in CCU. Queue length: {self.state['lqc']}")
                            # Record queue entry time for waiting time calculation
                            self.queue_arrival_times['CCU'].append(self.clock)
                            # Update statistics
                            if self.state['lqc'] > self.data['statistics']['queues']['CCU']['max']:
                                self.data['statistics']['queues']['CCU']['max'] = self.state['lqc']
                    elif destination == 'ICU':
                        if self.state['lei'] > 0:
                            self.state['lei'] -= 1
                            self.bed_usage['ICU'] += 1
                            # Schedule ICU discharge
                            stay_duration = self.exponential(1/25)
                            discharge_time = self.clock + stay_duration
                            self.schedule_event('ICU_Discharge', discharge_time, patient_id)
                            logging.info(f"Patient {patient_id} scheduled to discharge from ICU at {discharge_time}")
                        else:
                            # Increment ICU queue
                            self.queues['ICU'].append(patient_id)
                            self.state['lqi'] += 1
                            logging.info(f"Patient {patient_id} queued in ICU. Queue length: {self.state['lqi']}")
                            # Record queue entry time for waiting time calculation
                            self.queue_arrival_times['ICU'].append(self.clock)
                            # Update statistics
                            if self.state['lqi'] > self.data['statistics']['queues']['ICU']['max']:
                                self.data['statistics']['queues']['ICU']['max'] = self.state['lqi']

    def event_CE(self, patient_id):
        """CCU Entry Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Processing CCU Entry for Patient {patient_id} at time {self.clock}")

        if self.state['lec'] > 0 and self.state['E'] == 1:
            # Assign patient to CCU
            self.state['lec'] -= 1
            self.bed_usage['CCU'] += 1
            logging.info(f"Patient {patient_id} entered CCU. Remaining CCU beds: {self.state['lec']}")

            # Update patient state
            patient['current_location'] = 'CCU'
            patient['ccu_entry_time'] = self.clock

            # Schedule CCU discharge
            stay_duration = self.exponential(1/25)
            discharge_time = self.clock + stay_duration
            self.schedule_event('CCU_Discharge', discharge_time, patient_id)
            logging.info(f"Patient {patient_id} scheduled to discharge from CCU at {discharge_time}")
        else:
            # Handle CCU bed unavailability
            self.queues['CCU'].append(patient_id)
            self.state['lqc'] += 1
            logging.info(f"Patient {patient_id} queued in CCU. Queue length: {self.state['lqc']}")

            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['CCU'].append(self.clock)

            # Update patient state to reflect queue wait
            patient['current_location'] = 'CCU_Queue'
            patient['ccu_queue_entry_time'] = self.clock

            # Update statistics
            if self.state['lqc'] > self.data['statistics']['queues']['CCU']['max']:
                self.data['statistics']['queues']['CCU']['max'] = self.state['lqc']
                logging.info(f"New maximum queue length for CCU: {self.state['lqc']}")

            # Increment NOLE only if the queue exceeds capacity
            if len(self.queues['CCU']) > 10:
                self.data['statistics']['NOLN'] += 1
                logging.warning(f"CCU queue exceeded capacity. NOLN incremented to {self.data['statistics']['NOLN']}.")

    def event_IE(self, patient_id):
        """ICU Entry Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Processing ICU Entry for Patient {patient_id} at time {self.clock}")

        if self.state['lei'] > 0:
            # A bed is available in the ICU
            self.state['lei'] -= 1
            self.bed_usage['ICU'] += 1
            logging.info(f"Patient {patient_id} entered ICU. Remaining ICU beds: {self.state['lei']}")

            # Update patient state to reflect ICU admission
            patient['current_location'] = 'ICU'
            patient['icu_entry_time'] = self.clock

            # Schedule ICU discharge
            stay_duration = self.exponential(1/25)
            discharge_time = self.clock + stay_duration
            self.schedule_event('ICU_Discharge', discharge_time, patient_id)
            logging.info(f"Patient {patient_id} scheduled to discharge from ICU at {discharge_time}")
        else:
            # No ICU bed available, add patient to the ICU queue
            self.queues['ICU'].append(patient_id)
            self.state['lqi'] += 1
            logging.info(f"Patient {patient_id} queued in ICU. Queue length: {self.state['lqi']}")

            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['ICU'].append(self.clock)

            # Update patient state to reflect queue wait
            patient['current_location'] = 'ICU_Queue'
            patient['icu_queue_entry_time'] = self.clock

            # Update statistics
            if self.state['lqi'] > self.data['statistics']['queues']['ICU']['max']:
                self.data['statistics']['queues']['ICU']['max'] = self.state['lqi']
                logging.info(f"New maximum ICU queue length: {self.state['lqi']}")

            # If the queue exceeds capacity, increment NOLN
            if len(self.queues['ICU']) > 10:
                self.data['statistics']['NOLN'] += 1
                logging.warning(f"ICU queue exceeded capacity. NOLN incremented to {self.data['statistics']['NOLN']}.")

    def event_GE(self, patient_id):
        """General Ward Entry Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Processing General Ward Entry for Patient {patient_id} at time {self.clock}")

        if self.state['leg'] > 0:
            # Assign patient to General Ward
            self.state['leg'] -= 1
            self.bed_usage['General_Ward'] += 1
            logging.info(f"Patient {patient_id} entered General Ward. Remaining beds: {self.state['leg']}")

            # Update patient state
            patient['current_location'] = 'General_Ward'
            patient['general_ward_entry_time'] = self.clock

            # Schedule discharge after stay (2 days = 2880 minutes)
            discharge_time = self.clock + 2880
            self.schedule_event('GL', discharge_time, patient_id)
            logging.info(f"Patient {patient_id} scheduled to discharge from General Ward at {discharge_time}")
        else:
            # Handle bed unavailability by queuing the patient
            self.queues['General_Ward'].append(patient_id)
            self.state['lqg'] += 1
            logging.info(f"Patient {patient_id} queued in General_Ward. Queue length: {self.state['lqg']}")

            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['General_Ward'].append(self.clock)

            # Update patient state to reflect queue status
            patient['current_location'] = 'General_Ward_Queue'
            patient['general_ward_queue_entry_time'] = self.clock

            # Update statistics
            if self.state['lqg'] > self.data['statistics']['queues']['General_Ward']['max']:
                self.data['statistics']['queues']['General_Ward']['max'] = self.state['lqg']
                logging.info(f"New maximum queue length for General Ward: {self.state['lqg']}")

            # Increment NOLN only if the queue exceeds capacity
            if len(self.queues['General_Ward']) > 10:
                self.data['statistics']['NOLN'] += 1
                logging.warning(f"General_Ward queue exceeded capacity. NOLN incremented to {self.data['statistics']['NOLN']}.")

    def event_GL(self, patient_id):
        """General Leave Event (Discharge)"""
        patient = self.data['patients'][patient_id]
        patient['discharge_time'] = self.clock
        self.state['leg'] += 1
        self.bed_usage['General_Ward'] -= 1
        self.state['ns'] += 1
        logging.info(f"Patient {patient_id} discharged from General Ward at time {self.clock}")
        # Record length of stay
        length_of_stay = self.clock - patient['arrival_time']
        self.data['statistics']['length_of_stay'].append(length_of_stay)

        # Check if there are patients waiting in General Ward queue
        if self.queues['General_Ward']:
            next_patient_id = self.queues['General_Ward'].popleft()
            self.state['lqg'] -= 1
            self.state['leg'] -= 1
            self.bed_usage['General_Ward'] += 1
            logging.info(f"Patient {next_patient_id} moved from General_Ward queue to General Ward.")
            # Calculate waiting time
            if self.queue_arrival_times['General_Ward']:
                enqueue_time = self.queue_arrival_times['General_Ward'].popleft()
                wait_time = self.clock - enqueue_time
                self.data['statistics']['queues']['General_Ward']['total_wait_time'] += wait_time
                self.data['statistics']['queues']['General_Ward']['count_wait_time'] += 1
                logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in General_Ward queue.")
            else:
                logging.error(f"General_Ward arrival times deque empty for patient {next_patient_id}.")
            # Schedule discharge for the next patient
            stay_duration = self.exponential(1/50)
            discharge_time = self.clock + stay_duration
            self.schedule_event('GL', discharge_time, next_patient_id)

    def event_PO(self):
        """Power Outage Event"""
        self.state['E'] = 0
        # Adjust bed capacities for ICU and CCU to 80%
        self.state['lei'] = min(10, math.floor(0.8 * 10))    # 8 beds
        self.state['lec'] = min(5, math.floor(0.8 * 5))      # 4 beds
        logging.warning(f"Power outage at time {self.clock}. ICU beds limited to {self.state['lei']}, CCU beds limited to {self.state['lec']}.")

        # If current beds exceed new capacity, handle excess
        # For ICU
        while self.bed_usage['ICU'] > self.state['lei']:
            self.bed_usage['ICU'] -= 1
            self.state['lei'] -= 1
            # Transfer a patient from ICU to General Ward or queue
            if self.queues['ICU']:
                next_patient_id = self.queues['ICU'].popleft()
                self.state['lqi'] -= 1
                self.queues['General_Ward'].append(next_patient_id)
                self.state['lqg'] += 1
                logging.info(f"Patient {next_patient_id} moved from ICU to General_Ward due to power outage.")
                # Calculate waiting time
                if self.queue_arrival_times['General_Ward']:
                    enqueue_time = self.queue_arrival_times['General_Ward'].popleft()
                    wait_time = self.clock - enqueue_time
                    self.data['statistics']['queues']['General_Ward']['total_wait_time'] += wait_time
                    self.data['statistics']['queues']['General_Ward']['count_wait_time'] += 1
                    logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in General_Ward queue.")
                else:
                    logging.error(f"General_Ward arrival times deque empty for patient {next_patient_id}.")
                # Schedule LL for the next Lab_Preoperative patient
                self.schedule_event('GL', self.clock + self.exponential(1/50), next_patient_id)
            else:
                # No patients in ICU queue, cannot transfer
                logging.warning(f"No patients in ICU queue to transfer during power outage.")

        # For CCU
        while self.bed_usage['CCU'] > self.state['lec']:
            self.bed_usage['CCU'] -= 1
            self.state['lec'] -= 1
            # Transfer a patient from CCU to General Ward or queue
            if self.queues['CCU']:
                next_patient_id = self.queues['CCU'].popleft()
                self.state['lqc'] -= 1
                self.queues['General_Ward'].append(next_patient_id)
                self.state['lqg'] += 1
                logging.info(f"Patient {next_patient_id} moved from CCU to General_Ward due to power outage.")
                # Calculate waiting time
                if self.queue_arrival_times['General_Ward']:
                    enqueue_time = self.queue_arrival_times['General_Ward'].popleft()
                    wait_time = self.clock - enqueue_time
                    self.data['statistics']['queues']['General_Ward']['total_wait_time'] += wait_time
                    self.data['statistics']['queues']['General_Ward']['count_wait_time'] += 1
                    logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in General_Ward queue.")
                else:
                    logging.error(f"General_Ward arrival times deque empty for patient {next_patient_id}.")
                # Schedule GL for the next patient
                stay_duration = self.exponential(1/50)
                discharge_time = self.clock + stay_duration
                self.schedule_event('GL', discharge_time, next_patient_id)
            else:
                # No patients in CCU queue, cannot transfer
                logging.warning(f"No patients in CCU queue to transfer during power outage.")

    def event_PI(self):
        """Power Restoration Event"""
        self.state['E'] = 1
        # Restore full bed capacities
        self.state['lei'] = 10
        self.state['lec'] = 5
        logging.info(f"Power restored at time {self.clock}. ICU beds: {self.state['lei']}, CCU beds: {self.state['lec']}.")

        # Check if queues can be served now
        self.check_queues_after_power()

    def check_queues_after_power(self):
        """Check and serve queues for ICU and CCU after power restoration"""
        # ICU
        while self.queues['ICU'] and self.state['lei'] > 0:
            next_patient_id = self.queues['ICU'].popleft()
            self.state['lqi'] -= 1
            self.state['lei'] -= 1
            self.bed_usage['ICU'] += 1
            logging.info(f"Patient {next_patient_id} moved from ICU queue to ICU.")
            # Calculate waiting time
            if self.queue_arrival_times['ICU']:
                enqueue_time = self.queue_arrival_times['ICU'].popleft()
                wait_time = self.clock - enqueue_time
                self.data['statistics']['queues']['ICU']['total_wait_time'] += wait_time
                self.data['statistics']['queues']['ICU']['count_wait_time'] += 1
                logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in ICU queue.")
            else:
                logging.error(f"ICU arrival times deque empty for patient {next_patient_id}.")
            # Schedule ICU discharge
            stay_duration = self.exponential(1/25)
            discharge_time = self.clock + stay_duration
            self.schedule_event('ICU_Discharge', discharge_time, next_patient_id)

        # CCU
        while self.queues['CCU'] and self.state['lec'] > 0:
            next_patient_id = self.queues['CCU'].popleft()
            self.state['lqc'] -= 1
            self.state['lec'] -= 1
            self.bed_usage['CCU'] += 1
            logging.info(f"Patient {next_patient_id} moved from CCU queue to CCU.")
            # Calculate waiting time
            if self.queue_arrival_times['CCU']:
                enqueue_time = self.queue_arrival_times['CCU'].popleft()
                wait_time = self.clock - enqueue_time
                self.data['statistics']['queues']['CCU']['total_wait_time'] += wait_time
                self.data['statistics']['queues']['CCU']['count_wait_time'] += 1
                logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in CCU queue.")
            else:
                logging.error(f"CCU arrival times deque empty for patient {next_patient_id}.")
            # Schedule CCU discharge
            stay_duration = self.exponential(1/25)
            discharge_time = self.clock + stay_duration
            self.schedule_event('CCU_Discharge', discharge_time, next_patient_id)

    def event_ME(self, patient_id):
        """Morgue Entry Event"""
        patient = self.data['patients'][patient_id]
        patient['death'] = True
        logging.warning(f"Patient {patient_id} entered the morgue at time {self.clock}.")

        # Release OR bed
        self.release_bed('OR', patient_id)

        # Check if there are patients waiting in OR queues
        self.check_OR_queues()

    def event_ICU_Discharge(self, patient_id):
        """ICU Discharge Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Processing ICU Discharge for Patient {patient_id} at time {self.clock}")

        # Release ICU bed
        self.state['lei'] += 1
        self.bed_usage['ICU'] -= 1
        logging.info(f"Patient {patient_id} discharged from ICU at time {self.clock}. Available ICU beds: {self.state['lei']}")

        # Update patient state
        patient['current_location'] = 'Discharged_from_ICU'
        patient['icu_discharge_time'] = self.clock

        # Transfer to General Ward
        if self.state['leg'] > 0:
            # Assign bed in General Ward
            self.state['leg'] -= 1
            self.bed_usage['General_Ward'] += 1
            logging.info(f"Patient {patient_id} transferred to General Ward from ICU. Remaining General Ward beds: {self.state['leg']}")

            # Update patient state
            patient['current_location'] = 'General_Ward'
            patient['general_ward_entry_time'] = self.clock

            # Schedule discharge from General Ward
            stay_duration = self.exponential(1/50)
            discharge_time = self.clock + stay_duration
            self.schedule_event('GL', discharge_time, patient_id)
            logging.info(f"Patient {patient_id} scheduled to discharge from General Ward at {discharge_time}")
        else:
            # Increment General Ward queue
            self.queues['General_Ward'].append(patient_id)
            self.state['lqg'] += 1
            logging.info(f"Patient {patient_id} queued in General_Ward from ICU. Queue length: {self.state['lqg']}")

            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['General_Ward'].append(self.clock)

            # Update patient state to reflect queue status
            patient['current_location'] = 'General_Ward_Queue'
            patient['general_ward_queue_entry_time'] = self.clock

            # Update statistics
            if self.state['lqg'] > self.data['statistics']['queues']['General_Ward']['max']:
                self.data['statistics']['queues']['General_Ward']['max'] = self.state['lqg']
                logging.info(f"New maximum queue length for General Ward: {self.state['lqg']}")

        self.check_ICU_queues()

    def event_CCU_Discharge(self, patient_id):
        """CCU Discharge Event"""
        patient = self.data['patients'][patient_id]
        logging.info(f"Processing CCU Discharge for Patient {patient_id} at time {self.clock}")

        # Release CCU bed
        self.state['lec'] += 1
        self.bed_usage['CCU'] -= 1
        logging.info(f"Patient {patient_id} discharged from CCU at time {self.clock}. Available CCU beds: {self.state['lec']}")

        # Update patient state
        patient['current_location'] = 'Discharged_from_CCU'
        patient['ccu_discharge_time'] = self.clock

        # Transfer to General Ward
        if self.state['leg'] > 0:
            self.state['leg'] -= 1
            self.bed_usage['General_Ward'] += 1
            logging.info(f"Patient {patient_id} transferred to General Ward from CCU. Remaining General Ward beds: {self.state['leg']}")

            patient['current_location'] = 'General_Ward'
            patient['general_ward_entry_time'] = self.clock

            stay_duration = self.exponential(1/50)  # Mean 50 minutes
            discharge_time = self.clock + stay_duration
            self.schedule_event('GL', discharge_time, patient_id)
            logging.info(f"Patient {patient_id} scheduled to discharge from General Ward at {discharge_time}")
        else:
            # Increment General Ward queue
            self.queues['General_Ward'].append(patient_id)
            self.state['lqg'] += 1
            logging.info(f"Patient {patient_id} queued in General_Ward from CCU. Queue length: {self.state['lqg']}")

            # Record queue entry time for waiting time calculation
            self.queue_arrival_times['General_Ward'].append(self.clock)

            # Update patient state to reflect queue status
            patient['current_location'] = 'General_Ward_Queue'
            patient['general_ward_queue_entry_time'] = self.clock

            # Update statistics
            if self.state['lqg'] > self.data['statistics']['queues']['General_Ward']['max']:
                self.data['statistics']['queues']['General_Ward']['max'] = self.state['lqg']
                logging.info(f"New maximum queue length for General Ward: {self.state['lqg']}")

        self.check_CCU_queues()

    def check_OR_queues(self):
        """Check and serve the next patient in OR queues based on priority"""
        if self.state['LEs'] > 0:
            # Prioritize urgent patients
            if self.queues['OR_Urgent']:
                next_patient_id = self.queues['OR_Urgent'].popleft()
                self.state['lqse'] -= 1
                self.state['LEs'] -= 1
                self.bed_usage['OR'] += 1
                surgery_duration = self.data['patients'][next_patient_id]['surgery_time']
                self.schedule_event('SE_COMPLETED', self.clock + surgery_duration, next_patient_id)
                logging.info(f"Patient {next_patient_id} moved from OR_Urgent to OR. Remaining OR beds: {self.state['LEs']}.")
                # Calculate waiting time
                if self.queue_arrival_times['OR_Urgent']:
                    enqueue_time = self.queue_arrival_times['OR_Urgent'].popleft()
                    wait_time = self.clock - enqueue_time
                    self.data['statistics']['queues']['OR_Urgent']['total_wait_time'] += wait_time
                    self.data['statistics']['queues']['OR_Urgent']['count_wait_time'] += 1
                    logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in OR_Urgent queue.")
                else:
                    logging.error(f"OR_Urgent arrival times deque empty for patient {next_patient_id}.")
            elif self.queues['OR_Regular']:
                next_patient_id = self.queues['OR_Regular'].popleft()
                self.state['lqsn'] -= 1
                self.state['LEs'] -= 1
                self.bed_usage['OR'] += 1
                surgery_duration = self.data['patients'][next_patient_id]['surgery_time']
                self.schedule_event('SE_COMPLETED', self.clock + surgery_duration, next_patient_id)
                logging.info(f"Patient {next_patient_id} moved from OR_Regular to OR. Remaining OR beds: {self.state['LEs']}.")
                # Calculate waiting time
                if self.queue_arrival_times['OR_Regular']:
                    enqueue_time = self.queue_arrival_times['OR_Regular'].popleft()
                    wait_time = self.clock - enqueue_time
                    self.data['statistics']['queues']['OR_Regular']['total_wait_time'] += wait_time
                    self.data['statistics']['queues']['OR_Regular']['count_wait_time'] += 1
                    logging.info(f"Patient {next_patient_id} waited {wait_time:.2f} minutes in OR_Regular queue.")
                else:
                    logging.error(f"OR_Regular arrival times deque empty for patient {next_patient_id}.")
        else:
            logging.warning("No available OR beds to serve queued patients.")

    def collect_statistics(self):
        """Collect statistics at each event"""
        if self.state['LQe'] == 10:
            self.data['statistics']['ER_full_count'] += 1
        self.data['statistics']['ER_total'] += 1

        for queue_name in ['ER', 'Preoperative', 'Lab_Preoperative', 'Lab_ER', 'OR_Regular', 'OR_Urgent', 'ICU', 'CCU', 'General_Ward']:
            queue_length = len(self.queues[queue_name])
            self.data['statistics']['queues'][queue_name]['total'] += queue_length
            self.data['statistics']['queues'][queue_name]['count'] += 1
            if queue_length > self.data['statistics']['queues'][queue_name]['max']:
                self.data['statistics']['queues'][queue_name]['max'] = queue_length

    def calculate_bed_utilization_time(self):
        """Calculate average bed utilization rates based on time-weighted data"""
        for bed_type, total_time in self.data['statistics']['bed_utilization_time'].items():
            max_beds = {
                'Preoperative': 25,
                'ER': 10,
                'Lab': 3,
                'OR': 50,
                'General_Ward': 40,
                'ICU': 10,
                'CCU': 5
            }[bed_type]
            utilization = (total_time / (self.simulation_time * max_beds)) * 100  # Percentage
            utilization = min(utilization, 100)  # Cap at 100%
            self.data['statistics']['bed_utilization'][bed_type] = utilization
            logging.info(f"Bed utilization for {bed_type}: {utilization:.2f}%")
        logging.info("Calculated time-weighted bed utilization rates with caps.")

    def generate_excel_output(self, filename):
        """Generate Excel file with trace information.

        Args:
            filename (str): The filename for the Excel output.
        """
        # Make trace_list into dataframe
        if not self.trace_list:
            logging.warning("No trace data to export.")
            return
        trace_df = pd.DataFrame(self.trace_list, columns=[
            'Step', 'Hour', 'Current Event', 'Patient ID', 'State Variables', 'Cumulative Statistics',
            'Upcoming Events', 'Event Times'
        ])
        trace_df.to_excel(filename, index=False)
        logging.info(f"Generated Excel output at '{filename}'.")

    def calculate_performance_metrics(self):
        """Calculate and return performance metrics"""
        # Average length of stay
        avg_length_of_stay = np.mean(self.data['statistics']['length_of_stay']) if self.data['statistics']['length_of_stay'] else 0

        # Probability of ER queue being full
        prob_ER_full = self.data['statistics']['ER_full_count'] / self.data['statistics']['ER_total'] if self.data['statistics']['ER_total'] else 0

        # Average queue lengths
        avg_queue_lengths = {}
        for queue in self.data['statistics']['queues']:
            count = self.data['statistics']['queues'][queue]['count']
            total = self.data['statistics']['queues'][queue]['total']
            avg = total / count if count > 0 else 0
            avg_queue_lengths[queue] = avg

        # Maximum queue lengths
        max_queue_lengths = {queue: self.data['statistics']['queues'][queue]['max'] for queue in self.data['statistics']['queues']}

        # Average waiting times in queues
        avg_waiting_times = {}
        for queue in self.data['statistics']['queues']:
            total_wait = self.data['statistics']['queues'][queue]['total_wait_time']
            count_wait = self.data['statistics']['queues'][queue]['count_wait_time']
            avg_wait = total_wait / count_wait if count_wait > 0 else 0
            avg_waiting_times[queue] = avg_wait

        # Bed utilization based on time-weighted data
        bed_utilization = self.data['statistics']['bed_utilization']

        # Number of reoperations
        num_reoperations = self.data['statistics']['reoperations']

        # Number of hard surgeries
        num_hard_surgeries = self.data['statistics']['complex_surgeries']

        # Number of deaths
        num_deaths = self.data['statistics']['Num_deaths']

        metrics = {
            'average_length_of_stay_minutes': avg_length_of_stay,
            'probability_ER_full': prob_ER_full,
            'average_queue_lengths': avg_queue_lengths,
            'average_waiting_times_minutes': avg_waiting_times,
            'maximum_queue_lengths': max_queue_lengths,
            'bed_utilization_percent': bed_utilization,
            'number_of_reoperations': num_reoperations,
            'number_of_hard_surgeries': num_hard_surgeries,
            'number_of_deaths': num_deaths
        }

        logging.info("Calculated performance metrics.")
        return metrics

    def trace_event(self, event_type, patient_id):
        """Record trace information for Excel output"""
        trace_entry = {
            'Step': len(self.trace_list) + 1,
            'Hour': self.clock / 60,
            'Current Event': event_type,
            'Patient ID': patient_id,
            'State Variables': self.state.copy(),
            'Cumulative Statistics': self.data['statistics'].copy(),
            'Upcoming Events': [f"{evt[1]} for patient {evt[2]}" if evt[2] is not None else f"{evt[1]} for no patient" for evt in self.future_event_list],
            'Event Times': [evt[0] for evt in self.future_event_list]
        }
        self.trace_list.append(trace_entry)

    #####################################################################################################################################
    # Scheduling and Power Outage

    def schedule_event(self, event_type, event_time, patient_id):
        """
        Schedule an event by adding it to the future event list (FEL).

        Args:
            event_type (str): The type of event.
            event_time (float): The simulation time at which the event occurs.
            patient_id (int or None): The patient ID associated with the event.
        """
        if event_time < self.clock:
            logging.error(f"Attempted to schedule event '{event_type}' in the past. Current time: {self.clock}, Event time: {event_time}")
            return
        heapq.heappush(self.future_event_list, (event_time, event_type, patient_id))
        logging.debug(f"Scheduled event '{event_type}' for patient {patient_id} at time {event_time}")

    def schedule_power_outages(self):
        """
        Schedule power outage and restoration events based on a Poisson process.
        For simplicity, assuming average time between outages is 720 minutes (12 hours).
        """
        outage_rate = 1 / 720  # Average one outage every 720 minutes
        current_time = self.clock
        while current_time < self.simulation_time:
            inter_outage_time = self.poisson_arrival_time(outage_rate)
            outage_time = current_time + inter_outage_time
            if outage_time >= self.simulation_time:
                break
            self.schedule_event('PO', outage_time, None)
            restoration_time = outage_time + 60
            if restoration_time < self.simulation_time:
                self.schedule_event('PI', restoration_time, None)
            current_time = outage_time + 60

    #####################################################################################################################################
    # Run the Simulation

def run_single_replication(simulation_time,
                           arrival_rate_urgent=4/60,
                           prob_hard_surgery=0.05,
                           surgery_duration_mean={'s': 30.222215, 'm': 74.54242222, 'h': 242.0312},
                           surgery_duration_std={'s': 4.958886914, 'm': 9.953222836, 'h': 63.27455578},
                           preoperative_beds=25,
                           general_ward_beds=40,
                           seed=None):
    """Run a single simulation replication and return the KPIs."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    hospital_sim = HospitalSimulation(simulation_time=simulation_time,
                                      arrival_rate_urgent=arrival_rate_urgent,
                                      arrival_rate_regular=1/60,  # Default or can be parameterized as needed
                                      prob_hard_surgery=prob_hard_surgery,
                                      surgery_duration_mean=surgery_duration_mean,
                                      surgery_duration_std=surgery_duration_std,
                                      preoperative_beds=preoperative_beds,
                                      general_ward_beds=general_ward_beds)
    original_process_event = hospital_sim.process_event

    def process_event_with_trace(event_type, patient_id):
        original_process_event(event_type, patient_id)
        hospital_sim.trace_event(event_type, patient_id)

    hospital_sim.process_event = process_event_with_trace
    hospital_sim.run_simulation()
    metrics = hospital_sim.calculate_performance_metrics()

    # Generate Excel output with a unique filename
    if seed is not None:
        filename = f'hospital_simulation_trace_replication_{seed}.xlsx'
    else:
        filename = 'hospital_simulation_trace.xlsx'
    hospital_sim.generate_excel_output(filename)

    return metrics

def sensitivity_analysis():
    """Perform Sensitivity Analysis on key performance metrics."""
    simulation_time = 30 * 24 * 60  # 30 days in minutes
    replications = 30
    baseline_params = {
        'arrival_rate_urgent': 4 / 60,  # 4 urgent patients per hour
        'prob_hard_surgery': 0.05,      # 5% surgeries are hard
        'surgery_duration_mean': {'s': 30.222215, 'm': 74.54242222, 'h': 242.0312},
        'surgery_duration_std': {'s': 4.958886914, 'm': 9.953222836, 'h': 63.27455578}
    }

    variation_levels = {
        'arrival_rate_urgent': [baseline_params['arrival_rate_urgent'] * factor for factor in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2]],  # ±20%
        'prob_hard_surgery': [baseline_params['prob_hard_surgery'] * factor for factor in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]],  # ±20%
        'surgery_duration_mean_h': [baseline_params['surgery_duration_mean']['h'] * factor for factor in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]],  # ±10%
        'preoperative_beds': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'general_ward_beds': [35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    }

    sensitivity_results = []

    # Sensitivity for Average Length of Stay (varying surgery_duration_mean['h'])
    print("Starting Sensitivity Analysis for Average Length of Stay...")
    for duration in variation_levels['surgery_duration_mean_h']:
        params = baseline_params.copy()
        params['surgery_duration_mean'] = baseline_params['surgery_duration_mean'].copy()
        params['surgery_duration_mean']['h'] = duration
        # Run simulations
        metrics_list = []
        for i in range(replications):
            metrics = run_single_replication(simulation_time,
                                            arrival_rate_urgent=params['arrival_rate_urgent'],
                                            prob_hard_surgery=params['prob_hard_surgery'],
                                            surgery_duration_mean=params['surgery_duration_mean'],
                                            surgery_duration_std=params['surgery_duration_std'],
                                            preoperative_beds=25,
                                            general_ward_beds=40,
                                            seed=i)
            metrics_list.append(metrics)
        # Collect average_length_of_stay_minutes
        avg_length_of_stay = [m['average_length_of_stay_minutes'] for m in metrics_list]
        mean = np.mean(avg_length_of_stay)
        ci_low, ci_high = st.t.interval(0.95, len(avg_length_of_stay)-1, loc=mean, scale=st.sem(avg_length_of_stay))
        sensitivity_results.append({
            'Parameter': 'Surgery Duration Mean (h)',
            'Parameter Value': duration,
            'Metric': 'Average Length of Stay',
            'Mean': mean,
            'CI Lower': ci_low,
            'CI Upper': ci_high
        })

    # Sensitivity for Probability of ER Queue Being Full (varying arrival_rate_urgent)
    print("Starting Sensitivity Analysis for Probability of ER Queue Being Full...")
    for rate in variation_levels['arrival_rate_urgent']:
        params = baseline_params.copy()
        params['arrival_rate_urgent'] = rate
        # Run simulations
        metrics_list = []
        for i in range(replications):
            metrics = run_single_replication(simulation_time,
                                            arrival_rate_urgent=params['arrival_rate_urgent'],
                                            prob_hard_surgery=params['prob_hard_surgery'],
                                            surgery_duration_mean=params['surgery_duration_mean'],
                                            surgery_duration_std=params['surgery_duration_std'],
                                            preoperative_beds=25,
                                            general_ward_beds=40,
                                            seed=i)
            metrics_list.append(metrics)
        # Collect probability_ER_full
        prob_er_full = [m['probability_ER_full'] for m in metrics_list]
        mean = np.mean(prob_er_full)
        ci_low, ci_high = st.t.interval(0.95, len(prob_er_full)-1, loc=mean, scale=st.sem(prob_er_full))
        sensitivity_results.append({
            'Parameter': 'Arrival Rate Urgent',
            'Parameter Value': rate,
            'Metric': 'Probability ER Full',
            'Mean': mean,
            'CI Lower': ci_low,
            'CI Upper': ci_high
        })

    print("Starting Sensitivity Analysis for Number of Hard Surgeries...")
    for prob in variation_levels['prob_hard_surgery']:
        params = baseline_params.copy()
        params['prob_hard_surgery'] = prob
        # Run simulations
        metrics_list = []
        for i in range(replications):
            metrics = run_single_replication(simulation_time,
                                            arrival_rate_urgent=params['arrival_rate_urgent'],
                                            prob_hard_surgery=params['prob_hard_surgery'],
                                            surgery_duration_mean=params['surgery_duration_mean'],
                                            surgery_duration_std=params['surgery_duration_std'],
                                            preoperative_beds=25,
                                            general_ward_beds=40,
                                            seed=i)
            metrics_list.append(metrics)
        # Collect number_of_hard_surgeries
        num_hard_surgeries = [m['number_of_hard_surgeries'] for m in metrics_list]
        mean = np.mean(num_hard_surgeries)
        ci_low, ci_high = st.t.interval(0.95, len(num_hard_surgeries)-1, loc=mean, scale=st.sem(num_hard_surgeries))
        sensitivity_results.append({
            'Parameter': 'Probability of Hard Surgery',
            'Parameter Value': prob,
            'Metric': 'Number of Hard Surgeries',
            'Mean': mean,
            'CI Lower': ci_low,
            'CI Upper': ci_high
        })

    print("Starting Sensitivity Analysis for Preoperative Admission Unit Beds...")
    for beds in variation_levels['preoperative_beds']:
        params = baseline_params.copy()
        # Keep other parameters at baseline
        metrics_list = []
        for i in range(replications):
            metrics = run_single_replication(simulation_time,
                                            arrival_rate_urgent=params['arrival_rate_urgent'],
                                            prob_hard_surgery=params['prob_hard_surgery'],
                                            surgery_duration_mean=params['surgery_duration_mean'],
                                            surgery_duration_std=params['surgery_duration_std'],
                                            preoperative_beds=beds,
                                            general_ward_beds=40,
                                            seed=i)
            metrics_list.append(metrics)
        # Collect average_queue_length for Preoperative and average_length_of_stay_minutes
        avg_queue_length_preop = [m['average_queue_lengths']['Preoperative'] for m in metrics_list]
        mean_queue = np.mean(avg_queue_length_preop)
        ci_low_queue, ci_high_queue = st.t.interval(0.95, len(avg_queue_length_preop)-1, loc=mean_queue, scale=st.sem(avg_queue_length_preop))

        avg_length_of_stay = [m['average_length_of_stay_minutes'] for m in metrics_list]
        mean_stay = np.mean(avg_length_of_stay)
        ci_low_stay, ci_high_stay = st.t.interval(0.95, len(avg_length_of_stay)-1, loc=mean_stay, scale=st.sem(avg_length_of_stay))

        sensitivity_results.append({
            'Parameter': 'Preoperative Beds',
            'Parameter Value': beds,
            'Metric': 'Average Preoperative Queue Length',
            'Mean': mean_queue,
            'CI Lower': ci_low_queue,
            'CI Upper': ci_high_queue
        })
        sensitivity_results.append({
            'Parameter': 'Preoperative Beds',
            'Parameter Value': beds,
            'Metric': 'Average Length of Stay',
            'Mean': mean_stay,
            'CI Lower': ci_low_stay,
            'CI Upper': ci_high_stay
        })

    print("Starting Sensitivity Analysis for General Ward Beds...")
    for beds in variation_levels['general_ward_beds']:
        params = baseline_params.copy()
        # Keep other parameters at baseline
        metrics_list = []
        for i in range(replications):
            metrics = run_single_replication(simulation_time,
                                            arrival_rate_urgent=params['arrival_rate_urgent'],
                                            prob_hard_surgery=params['prob_hard_surgery'],
                                            surgery_duration_mean=params['surgery_duration_mean'],
                                            surgery_duration_std=params['surgery_duration_std'],
                                            preoperative_beds=25,
                                            general_ward_beds=beds,
                                            seed=i)
            metrics_list.append(metrics)
        # Collect bed_utilization for General Ward and average_waiting_times_minutes for General Ward
        bed_utilization_general = [m['bed_utilization_percent']['General_Ward'] for m in metrics_list]
        mean_util = np.mean(bed_utilization_general)
        ci_low_util, ci_high_util = st.t.interval(0.95, len(bed_utilization_general)-1, loc=mean_util, scale=st.sem(bed_utilization_general))

        avg_waiting_time_general = [m['average_waiting_times_minutes']['General_Ward'] for m in metrics_list]
        mean_wait = np.mean(avg_waiting_time_general)
        ci_low_wait, ci_high_wait = st.t.interval(0.95, len(avg_waiting_time_general)-1, loc=mean_wait, scale=st.sem(avg_waiting_time_general))

        sensitivity_results.append({
            'Parameter': 'General Ward Beds',
            'Parameter Value': beds,
            'Metric': 'Bed Utilization (%)',
            'Mean': mean_util,
            'CI Lower': ci_low_util,
            'CI Upper': ci_high_util
        })
        sensitivity_results.append({
            'Parameter': 'General Ward Beds',
            'Parameter Value': beds,
            'Metric': 'Average Waiting Time in General Ward Queue',
            'Mean': mean_wait,
            'CI Lower': ci_low_wait,
            'CI Upper': ci_high_wait
        })

    sensitivity_df = pd.DataFrame(sensitivity_results)

    print("\n--- Sensitivity Analysis Results ---\n")
    print(sensitivity_df)

    sensitivity_df.to_csv('sensitivity_analysis_results.csv', index=False)
    print("\nSensitivity analysis results saved to 'sensitivity_analysis_results.csv'.")

    return sensitivity_df

def plot_sensitivity_results(sensitivity_df):
    """Plot the Sensitivity Analysis Results"""
    sns.set(style="whitegrid")

    # Plot for Average Length of Stay
    data_alos = sensitivity_df[sensitivity_df['Metric'] == 'Average Length of Stay']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_alos, marker='o')
    plt.fill_between(data_alos['Parameter Value'], data_alos['CI Lower'], data_alos['CI Upper'], alpha=0.2)
    plt.title('Sensitivity Analysis: Average Length of Stay')
    plt.xlabel('Mean Surgery Duration (h)')
    plt.ylabel('Average Length of Stay (minutes)')
    plt.tight_layout()
    plt.savefig('sensitivity_alos.png')
    plt.show()

    # Plot for Probability ER Full
    data_prob_er = sensitivity_df[sensitivity_df['Metric'] == 'Probability ER Full']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_prob_er, marker='o')
    plt.fill_between(data_prob_er['Parameter Value'], data_prob_er['CI Lower'], data_prob_er['CI Upper'], alpha=0.2)
    plt.title('Sensitivity Analysis: Probability ER Full')
    plt.xlabel('Arrival Rate Urgent (per minute)')
    plt.ylabel('Probability ER Full')
    plt.tight_layout()
    plt.savefig('sensitivity_prob_er_full.png')
    plt.show()

    # Plot for Number of Hard Surgeries
    data_hard_surg = sensitivity_df[sensitivity_df['Metric'] == 'Number of Hard Surgeries']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_hard_surg, marker='o')
    plt.fill_between(data_hard_surg['Parameter Value'], data_hard_surg['CI Lower'], data_hard_surg['CI Upper'], alpha=0.2)
    plt.title('Sensitivity Analysis: Number of Hard Surgeries')
    plt.xlabel('Probability of Hard Surgery')
    plt.ylabel('Number of Hard Surgeries')
    plt.tight_layout()
    plt.savefig('sensitivity_num_hard_surgeries.png')
    plt.show()

    # **New Plot 1: Preoperative Beds - Average Preoperative Queue Length**
    data_preop_queue = sensitivity_df[sensitivity_df['Metric'] == 'Average Preoperative Queue Length']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_preop_queue, marker='o', color='purple')
    plt.fill_between(data_preop_queue['Parameter Value'], data_preop_queue['CI Lower'], data_preop_queue['CI Upper'], alpha=0.2, color='purple')
    plt.title('Sensitivity Analysis: Average Preoperative Queue Length')
    plt.xlabel('Number of Preoperative Beds')
    plt.ylabel('Average Preoperative Queue Length')
    plt.tight_layout()
    plt.savefig('sensitivity_preoperative_queue_length.png')
    plt.show()

    # **New Plot 2: Preoperative Beds - Average Length of Stay**
    data_preop_stay = sensitivity_df[sensitivity_df['Metric'] == 'Average Length of Stay']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_preop_stay, marker='o', color='orange')
    plt.fill_between(data_preop_stay['Parameter Value'], data_preop_stay['CI Lower'], data_preop_stay['CI Upper'], alpha=0.2, color='orange')
    plt.title('Sensitivity Analysis: Average Length of Stay vs Preoperative Beds')
    plt.xlabel('Number of Preoperative Beds')
    plt.ylabel('Average Length of Stay (minutes)')
    plt.tight_layout()
    plt.savefig('sensitivity_alos_preoperative_beds.png')
    plt.show()

    # **New Plot 3: General Ward Beds - Bed Utilization**
    data_ward_util = sensitivity_df[sensitivity_df['Metric'] == 'Bed Utilization (%)']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_ward_util, marker='o', color='green')
    plt.fill_between(data_ward_util['Parameter Value'], data_ward_util['CI Lower'], data_ward_util['CI Upper'], alpha=0.2, color='green')
    plt.title('Sensitivity Analysis: General Ward Bed Utilization')
    plt.xlabel('Number of General Ward Beds')
    plt.ylabel('Bed Utilization (%)')
    plt.tight_layout()
    plt.savefig('sensitivity_general_ward_utilization.png')
    plt.show()

    # **New Plot 4: General Ward Beds - Average Waiting Time in Queue**
    data_ward_wait = sensitivity_df[sensitivity_df['Metric'] == 'Average Waiting Time in General Ward Queue']
    plt.figure(figsize=(10,6))
    sns.lineplot(x='Parameter Value', y='Mean', data=data_ward_wait, marker='o', color='red')
    plt.fill_between(data_ward_wait['Parameter Value'], data_ward_wait['CI Lower'], data_ward_wait['CI Upper'], alpha=0.2, color='red')
    plt.title('Sensitivity Analysis: Average Waiting Time in General Ward Queue')
    plt.xlabel('Number of General Ward Beds')
    plt.ylabel('Average Waiting Time (minutes)')
    plt.tight_layout()
    plt.savefig('sensitivity_general_ward_waiting_time.png')
    plt.show()

def run_replications(simulation_time, replications, seed_start=0):
    """Run multiple simulation replications and collect metrics."""
    metrics_list = []

    for i in range(replications):
        seed = seed_start + i
        print(f"Running replication {i+1}/{replications} with seed {seed}...")
        metrics = run_single_replication(simulation_time,
                                        arrival_rate_urgent=4/60,  # Or vary as needed
                                        prob_hard_surgery=0.05,    # Or vary as needed
                                        surgery_duration_mean={'s': 30.222215, 'm': 74.54242222, 'h': 242.0312},
                                        surgery_duration_std={'s': 4.958886914, 'm': 9.953222836, 'h': 63.27455578},
                                        preoperative_beds=25,
                                        general_ward_beds=40,
                                        seed=seed)
        metrics_list.append(metrics)

    df = pd.json_normalize(metrics_list)

    # Calculate 95% Confidence Intervals
    ci_dict = {}
    n = replications
    t_critical = st.t.ppf(0.975, df=n-1)  # Two-tailed t critical value for 95% CI

    ci_data = []

    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std(ddof=1)
        se = std / sqrt(n)
        ci_low = mean - t_critical * se
        ci_high = mean + t_critical * se
        ci_data.append({
            'Metric': column,
            'Mean': mean,
            'CI Lower': ci_low,
            'CI Upper': ci_high
        })

    ci_results = pd.DataFrame(ci_data)

    print("\n--- 95% Confidence Intervals for Performance Metrics ---\n")
    for index, row in ci_results.iterrows():
        print(f"{row['Metric']}: {row['Mean']:.2f} ± {t_critical * (df[row['Metric']].std()/sqrt(n)):.2f} "
              f"(95% CI: {row['CI Lower']:.2f} - {row['CI Upper']:.2f})")

    ci_results.to_csv('performance_metrics_confidence_intervals.csv', index=False)
    print("\nConfidence intervals saved to 'performance_metrics_confidence_intervals.csv'.")

    return df, ci_results

def main():
    """Main function to run simulation replications and perform sensitivity analysis."""
    simulation_time = 30 * 24 * 60  # Simulate for 30 days in minutes
    replications = 30
    seed_start = 0  # Starting seed for reproducibility

    print("Starting Hospital Simulation Replications...\n")
    df_metrics, ci_results = run_replications(simulation_time, replications, seed_start)
    print("\nAll replications completed.\n")

    sensitivity_df = sensitivity_analysis()
    plot_sensitivity_results(sensitivity_df)

    df_metrics.to_csv('simulation_replications_metrics.csv', index=False)
    print("\nAll replication metrics saved to 'simulation_replications_metrics.csv'.")

    bed_utilization = []
    for idx, row in ci_results.iterrows():
        if row['Metric'].startswith('bed_utilization_percent'):
            bed_type = row['Metric'].split('.')[-1]
            bed_utilization.append({
                'Bed Type': bed_type,
                'Utilization Mean (%)': row['Mean'],
                'CI Lower (%)': row['CI Lower'],
                'CI Upper (%)': row['CI Upper']
            })
    bed_utilization_df = pd.DataFrame(bed_utilization)
    bed_utilization_df.to_csv('bed_utilization_metrics.csv', index=False)
    print("Bed utilization metrics saved to 'bed_utilization_metrics.csv'.")

if __name__ == "__main__":
    main()
