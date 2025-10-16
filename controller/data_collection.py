import os
import pickle
import numpy as np
from datetime import datetime

class DataCollectionInterface:
    def __init__(self, folder_path='.', mode="pickle"):
        """
        Initialize the data collection interface.

        :param file_path: Path to save the data file (e.g., "data.pkl" or "data.npy").
        :param mode: Saving mode, either "pickle" or "npy".
        """
        self.folder_path = folder_path
        self.mode = mode.lower()
        self.data = []
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(
            self.folder_path, f"{timestamp}.{self.mode}"
        )
        os.makedirs(self.folder_path, exist_ok=True)


        if self.mode not in ["pickle", "npy"]:
            raise ValueError("Mode must be 'pickle' or 'npy'.")


    def add_data(self, new_data):
        """
        Add new data to the collection.

        :param new_data: Data to add (can be any Python object for pickle or numerical for npy).
        """
        self.data.append(new_data)

    def save(self):
        """
        Save the data to the specified file.
        """
        if self.mode == "pickle":
            with open(self.file_path, "ab") as file:
                for item in self.data:
                    pickle.dump(item, file)

        elif self.mode == "npy":
            np.save(self.file_path, np.array(self.data, dtype=object))

        print(f'data with {len(self.data)} is saved to {self.file_path}')
        self.data = []  # Clear saved data from memory

    def load(self):
        """
        Load all saved data from the file.

        :return: List of saved data.
        """
        if self.mode == "pickle":
            loaded_data = []
            with open(self.file_path, "rb") as file:
                while True:
                    try:
                        loaded_data.append(pickle.load(file))
                    except EOFError:
                        break
            return loaded_data
        elif self.mode == "npy":
            return list(np.load(self.file_path, allow_pickle=True))