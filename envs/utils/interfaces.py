from abc import ABC, abstractmethod


class StateInterface(ABC):
    @abstractmethod
    def get_state(self):
        pass