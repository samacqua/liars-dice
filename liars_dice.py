"""Script to play agents against eachother in Liar's Dice."""

from collections import Counter
import random

CALL_LIAR = None


def is_correct_call(d1, d2, call):
    """Returns if the called number of dice is correct."""
    count, side = call
    return not bool(Counter({side: count}) - Counter(d1 + d2))


def score(d1, d2, hist):
    """Gets the score in {-1,1} relative to player 1."""

    assert hist and hist[-1] is CALL_LIAR

    # Player 2 called.
    if len(hist) % 2 == 0:
        res = is_correct_call(d1, d2, hist[-2])

    # Player 1 called.
    else:
        res = not is_correct_call(d1, d2, hist[-2])
    return int(res)*2 - 1


class Agent:
    def __init__(self, name):
        self.name = name

    def move(self, hist: tuple, dice: tuple) -> tuple:
        raise NotImplementedError("Please implement a move function.")
    
class SolvedAgent(Agent):

    def __init__(self, name, table):
        super().__init__(name)
        self.table = table

    def move(self, hist: tuple, dice: tuple):
        """Returns the move from the pre-calculated table."""

        if hist not in self.table[dice]:
            raise ValueError(f"History {hist} not in table for dice {dice}.")
        
        # List of tuples(move, prob). Sample from this list.
        moves, probs = zip(*self.table[dice][hist])
        move = random.choices(moves, weights=probs)[0]

        # print(hist, dice, move)
        return move


class Game:

    def __init__(self, agents, n_dice: int, n_sides: int = 6):
        if len(agents) != 2:
            raise NotImplementedError("Only implemented heads up as of now.")
        self.agents = agents
        self.n_dice = n_dice
        self.n_sides = n_sides

    def roll_dice(self, n_dice: int, n_sides: int = 6) -> tuple:
        """Rolls n_dice dice with n_sides sides each."""
        return tuple(random.randint(1, n_sides) for _ in range(n_dice))

    def play_round(self, agents_to_n_dice: dict) -> int:
        """Plays a round of liar's dice."""

        agents_to_dice = {agent: self.roll_dice(n_dice, self.n_sides) for agent, n_dice in agents_to_n_dice.items()}

        hist = tuple()
        while not (hist and hist[-1] == CALL_LIAR):
            for agent, dice in agents_to_dice.items():
                move = agent.move(hist, dice)
                hist += (move,)
                if hist[-1] == CALL_LIAR:
                    break

        score_ = score(agents_to_dice[self.agents[0]], agents_to_dice[self.agents[1]], hist)
        return score_

    def play(self) -> int:
        """Returns 0 if player 1 wins, 1 if player 2 wins."""

        agents_to_n_dice = {self.agents[0]: self.n_dice, self.agents[1]: self.n_dice}

        hist = tuple()
        while all((n_dice > 0 for n_dice in agents_to_n_dice.values())):
            round_result = self.play_round(agents_to_n_dice)
            if round_result == 1:
                agents_to_n_dice[self.agents[1]] -= 1
            else:
                agents_to_n_dice[self.agents[0]] -= 1

        # Whomever has dice left wins.
        for agent, n_dice in agents_to_n_dice.items():
            if n_dice > 0:
                # print(self.agents[0 if agent == self.agents[0] else 1].name, "won.")
                return 0 if agent == self.agents[0] else 1
    
    def play_n(self, n: int):
        """Plays n games and returns the results."""
        return [self.play() for _ in range(n)]
