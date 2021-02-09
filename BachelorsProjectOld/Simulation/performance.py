

class Performance:
    def __init__(self):
        self.agent_deaths = 0
        self.amount_fires_isolated = 0
        self.cumulative_burnt = 0
        self.episode = 0

    def clear(self):
        self.agent_deaths = 0
        self.amount_fires_isolated = 0
        self.cumulative_burnt = 0
        self.episode = 0