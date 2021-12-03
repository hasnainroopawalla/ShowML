from abc import ABC, abstractmethod

class Event(ABC):
    @abstractmethod
    def run_event():
        pass

class StartEvent(Event):
    def run_event():


