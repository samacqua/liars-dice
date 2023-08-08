"""Solve endgame of liar's dice using linear programming.

Code based on https://github.com/thomasahle/snyd, concept based on 
http://www.sciencedirect.com/science/article/pii/089982569290035Q.
"""

from collections import Counter
from collections import defaultdict
import itertools
import sys
from ortools.linear_solver import pywraplp
import fractions
from functools import lru_cache
import time
from scipy.stats import binom_test


if len(sys.argv) < 4:
    print('Run {} [dice1] [dice2] [sides] mode'.format(sys.argv[0]))
    sys.exit()
else:
    DICE1 = int(sys.argv[1])
    DICE2 = int(sys.argv[2])
    SIDES = int(sys.argv[3])

NORMAL, JOKER, STAIRS, FONTROMEU = range(4)
if len(sys.argv) >= 5:
    mode = {'normal': NORMAL, 'joker': JOKER, 'stairs': STAIRS}[sys.argv[4]]
else:
    mode = NORMAL

################################################################
# Game definition
################################################################

if mode == NORMAL:
    CALLS = [(count, side)
            for count in range(1, DICE1+DICE2+1)
            for side in range(1, SIDES+1)]
elif mode == JOKER:
    # With jokers we can't call 1
    CALLS = [(count, side)
        for count in range(1, DICE1+DICE2+1)
        for side in range(2, SIDES+1)]
elif mode == STAIRS:
    # With stairs we can call up to four sixes...
    CALLS = [(count, side)
        for count in range(1, 2*(DICE1+DICE2)+1)
        for side in range(2, SIDES+1)]
elif mode == FONTROMEU:
    CALLS = [(count, side)
        for count in range(1, DICE1+DICE2+1)
        for side in range(1, SIDES+1)]

# All possible rolls.
ROLLS1 = list(itertools.product(range(1,SIDES+1), repeat=DICE1))
ROLLS2 = list(itertools.product(range(1,SIDES+1), repeat=DICE2))

CALL_LIAR = None  # Special token to indicate end of game.

def possible_calls(hist):
    """Yield all possible calls given a history."""

    # If first move, can play anything.
    if not hist:
        yield from CALLS
        return
    
    # If end of game, return.
    if hist[-1] is CALL_LIAR:
        return
    
    # Have to 1-up the last call.
    for call in CALLS:
        if call > hist[-1]:
            yield call

    # If can't 1-up, return.
    yield CALL_LIAR

def is_correct_call(d1, d2, call):
    """Returns if the called number of dice is correct."""

    count, side = call
    if mode == JOKER:
        d1 = tuple(side if d == 1 else d for d in d1)
        d2 = tuple(side if d == 1 else d for d in d2)
    if mode == STAIRS:
        if d1 == tuple(range(1,len(d1)+1)):
            d1 = (side,)*(len(d1)+1)
        if d2 == tuple(range(1,len(d2)+1)):
            d2 = (side,)*(len(d2)+1)

    return not bool(Counter({side: count}) - Counter(d1 + d2))

def is_leaf(hist):
    """Return if the current state is a leaf node."""

    assert not hist or hist[0] is not CALL_LIAR, "END_OF_GAME can't be first call"
    return hist and hist[-1] is CALL_LIAR

def histories(hist=()):
    """Keep yielding possible histories until end of game."""

    yield hist
    if not is_leaf(hist):
        for call in possible_calls(hist):
            yield from histories(hist+(call,))

# xs (ys) are the states after a move by player + the root.
# Each of these are given a variable, since they are either leafs or parents to leafs.
# This is of course game specific, so maybe it's a bad way to do it...
xs = lambda: (h for h in histories() if not h or len(h)%2==1)
ys = lambda: (h for h in histories() if not h or len(h)%2==0)
# The inners are those with children. Note the root is present in both.
inner_xs = lambda: (h for h in xs() if not is_leaf(h))
inner_ys = lambda: (h for h in ys() if not is_leaf(h))

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

################################################################
# Write as matrix
################################################################

def initSolver():
    """Initialize the linear program solver."""
    
    t = time.time()
    print('Creating variables...', file=sys.stderr)
    solver = pywraplp.Solver('', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Set up variables.
    # zvs = the possible scores for player 1.
    # xvs = the behavior profile for player 2.
    zvs = {(d2,x): solver.NumVar(-solver.infinity(), solver.infinity(),
        '<z d={} h={}>'.format(d2,stringify_hist(x))) for x in inner_xs() for d2 in ROLLS2}
    xvs = {(d1,x): solver.NumVar(0, 1,
        '<x d={} h={}>'.format(d1,stringify_hist(x))) for x in xs() for d1 in ROLLS1}
    print('Took {}s'.format(time.time()-t), file=sys.stderr)

    t = time.time()
    print('Setting constraints 1', file=sys.stderr)

    # The thing to maximize: f.T@z.
    # TODO: make sense of this.
    objective = solver.Objective()
    for d2 in ROLLS2:
        objective.SetCoefficient(zvs[d2,()], 1)
    objective.SetMaximization()

    print('Took {}s'.format(time.time()-t), file=sys.stderr)

    # E: (d1,x) -> (d1,y_inner)
    # F^T: (d2,x_inner) -> (d2,y)
    # A: (d2,y) -> (d1,x)
    # A^T: (d1,x) -> (d2,y)
    # |z| : d2*x_inner

    # A simple strategy.
    # TODO: make sense of this.
    simp = {(d1,x): 0 for d1 in ROLLS1 for x in xs()}
    for d1 in ROLLS1:
        simp[d1,()] = 1
        for y in inner_ys():
           call = next(possible_calls(y))
           simp[d1, y+(call,)] = simp[d1, y[:-1]]

    t = time.time()
    print('Setting constraints 2', file=sys.stderr)

    # Equalities: Ex = e.
    print('Equalities')
    for d1 in ROLLS1:
        # print('Roll', d1)

        # The root node of the game tree should have x_u = 1 (equation 3).
        root_constraint = solver.Constraint(1, 1)
        root_constraint.SetCoefficient(xvs[d1,()], 1)
        # print('1 =', xvs[d1,()].name())
        # print('1 =', simp[d1,()])

        # Make siblings sum to their parent for player 2 (equation 5).
        for hist in inner_ys():
            constraint = solver.Constraint(0, 0)
            # print('0 =', end=' ')
            constraint.SetCoefficient(xvs[d1,hist[:-1]], 1)
            # print(xvs[d1,hist[:-1]].name(), end=' ')

            for call in possible_calls(hist):
                # print('-', xvs[d1,hist+(call,)].name(), end=' ')
                constraint.SetCoefficient(xvs[d1,hist+(call,)], -1)
            # print('\n')

            # Check for simple strategy.
            simp_strategy_sum = simp[d1,hist[:-1]]
            for call in possible_calls(hist):
                simp_strategy_sum -= simp[d1,hist+(call,)]
            assert not simp_strategy_sum, "Simple strategy constraints not satisfied."


    print('Took {}s'.format(time.time()-t), file=sys.stderr)

    #F = np.zeros((len(inner_xs)+1, len(ys)))
    #for i, x in enumerate(inner_xs):
    #    if x == ():
    #        F[i, ys.index(())] = 1
    #    else:
    #        F[i, ys.index(x[:-1])] = 1
    #        for call in possible_calls(x):
    #            F[i, ys.index(x+(call,))] = -1
    #print(F)
    #print(F.T)

    # F and A must have equal number of collumns
    # This is true, they both have ROLLS x ys

    t = time.time()
    print('Setting constraints 3', file=sys.stderr)

    # Bound zT@F - xT@A <= 0
    # Bound F.T@z - A.Tx <= 0
    # z@F0 - x@A0 >= 0, ...
    print('Bounds')
    for d2 in ROLLS2:
        # print('Roll', d2)

        # Now the leaves.
        for hist in ys():
            constraint = solver.Constraint(-solver.infinity(), 0)
            # print('0 >= ', end='')

            # We have to take care of the root as well.
            if hist == ():

                # z@F:0
                # print(zvs[d2,()].name(), end=' ')
                constraint.SetCoefficient(zvs[d2,()], 1)
                for call in possible_calls(hist):
                    # print('+', zvs[d2,hist+(call,)].name(), end=' ')
                    constraint.SetCoefficient(zvs[d2,hist+(call,)], 1)
                # print()

                # A:0 is simply empty

                continue

            # z@F:i
            # I'm a y. To which internals am I a child, to which a parent?
            # We may not have any children, that's fine. If we are a leaf,
            # F will only have +1 entries for us.
            # print('-', zvs[d2,hist[:-1]].name(), end=' ')
            constraint.SetCoefficient(zvs[d2,hist[:-1]], -1)
            for call in possible_calls(hist):
                child = hist+(call,)
                if not is_leaf(child):
                    # print('+', zvs[d2,child].name(), end=' ')
                    constraint.SetCoefficient(zvs[d2,child], 1)

            # import pdb; pdb.set_trace()

            # -x@A:i
            lhist = hist+(CALL_LIAR,) if hist[-1] is not CALL_LIAR else hist
            xhist = hist+(CALL_LIAR,) if hist[-1] is not CALL_LIAR else hist[:-1]
            for d1 in ROLLS1:
                sign = '-' if -score(d1, d2, lhist) < 0 else '+'
                # print(sign, xvs[d1,xhist].name(), end=' ')
                # print(sign, simp[d1,xhist], end=' ')
                constraint.SetCoefficient(xvs[d1,xhist], -score(d1, d2, lhist))
            # print()

            # import pdb; pdb.set_trace()

    print('Took {}s'.format(time.time()-t), file=sys.stderr)

    return solver, xvs, zvs

# Formatting of solution
def stringify_call(call):
    """Returns a string representation of a call."""
    
    if call is CALL_LIAR:
        return "call"
    
    n_dice, face = call
    if n_dice == 1:
        return f"1 {face}"
    else:
        return f"{n_dice} {face}s"

def stringify_hist(hist):
    """Returns a string representation of the history."""
    return '[' + ','.join(map(stringify_call,hist)) + ']'

def stringify_fraction(val):
    """Returns a string representation of a fraction."""
    return str(fractions.Fraction.from_float(val).limit_denominator())

class CounterStrategy:

    def __init__(self, xvs):
        self.xvs = xvs

    @lru_cache(maxsize=10**5)
    def findCallProb(self, d1, hist):
        """Return the probability that player 1 did the last move of hist."""

        assert len(hist) % 2 == 1
        xhis = self.xvs[d1, hist].solution_value()
        xpar = self.xvs[d1, hist[:-2]].solution_value()

        return xhis/xpar if xpar > 1e-10 else 0

    @lru_cache(maxsize=10**5)
    def findP2Call(self, d2, hist):
        """Find the best call for p2, choosing the optimal deterministic counter strategy."""

        assert len(hist) % 2 == 1
        if sum(self.findCallProb(d1,hist) for d1 in ROLLS1) < 1e-6:
            #if d2 == (0,) and hist == ((1,1),):
            #    print('findP2Call called on impossible history')
            return next(possible_calls(hist))
        pd1s = self.estimateP1Rolls(hist)
        if d2 == (0,) and hist == ((1,1),):
            pass
            #print('pd1s', pd1s)
            #print('scores', [sum(p*stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS)) for p,d1 in zip(pd1s,ROLLS)])
        return min(possible_calls(hist), key=lambda call:
                sum(p*self.stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS1)))

    @lru_cache(maxsize=10**5)
    def stateValue(self, d1, d2, hist):
        """Return expected payoff for player 1."""

        if hist and hist[-1] is CALL_LIAR:
            res = score(d1, d2, hist)

        # Player 1.
        elif len(hist) % 2 == 0:
            res = sum(self.stateValue(d1, d2, hist+(call,))
                    * self.findCallProb(d1, hist+(call,))
                    for call in possible_calls(hist))

        # Player 2.
        elif len(hist) % 2 == 1:
            p2call = self.findP2Call(d2, hist)
            res = self.stateValue(d1, d2, hist+(p2call,))
        #print('stateValue({}, {}, {}) = {}'.format(d1, d2, hist, res))
        return res

    @lru_cache(maxsize=10**5)
    def estimateP1Rolls(self, hist):
        """TODO."""

        assert len(hist) % 2 == 1
        # Simple bayes
        prob_hist_given_d = [self.findCallProb(d1, hist) for d1 in ROLLS1]
        if sum(prob_hist_given_d) < 1e-10:
            return [1/len(ROLLS1) for _ in ROLLS1]
        return [p/sum(prob_hist_given_d) for p in prob_hist_given_d]


def printTrees(cs):
    """Print the trees for each possible roll."""

    print('Trees:')
    for d1 in ROLLS1:
        for hist in histories():

            # At root, print the roll value
            if not hist:
                avgValue = stringify_fraction(sum(cs.stateValue(d1, d2, ()) for d2 in ROLLS2)/len(ROLLS2))
                values = ', '.join(stringify_fraction(cs.stateValue(d1, d2, ())) for d2 in ROLLS2)
                print('Roll: {}, Expected: {}, Values: {}'.format(d1, avgValue, values))
                continue
            
            # If a parent has zero probability, don't go there
            hist_probs = [cs.findCallProb(d1, hist[:j])for j in range(1,len(hist)+1,2)]
            if any([h < 1e-8 for h in hist_probs]):
                continue
            # print(hist_probs)
            s = '|  '*len(hist) + (stringify_call(hist[-1]) if hist else 'root')

            # if len(hist) >= 2 and hist[-1] is CALL_LIAR:
            #     wl = [is_correct_call(d1, d2, hist[-2]) for d2 in ROLLS2]
            #     s += " " + " ".join("W" if w else "L" for w in wl)

            # import pdb; pdb.set_trace()
            if len(hist)%2==1:
                prob = stringify_fraction(cs.findCallProb(d1, hist))
                print('{} p={}'.format(s, prob))
            else:
                # Prints if player 2's best move is what was played (*) or not (_) 
                # (where index i == they have that die).
                tag = ''.join('_*'[hist[-1] == cs.findP2Call(d2,hist[:-1])] for d2 in ROLLS2)
                print(s, tag)


def main():

    print('Setting up linear program', file=sys.stderr)
    solver, xvs, zvs = initSolver()

    t = time.time()
    print('Solving', file=sys.stderr)

    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Status:', status, file=sys.stderr)
        print(zvs[(0,),()].solution_value())
        return

    print('Took {}s'.format(time.time()-t), file=sys.stderr)

    cs = CounterStrategy(xvs)
    printTrees(cs)
    print()

    # Print the probabilities of each move in every situation for player 1.
    dice_to_hist_to_move_p1 = {}
    for d1 in ROLLS1:
        print("\nDice: {}".format(d1))
        hist_to_move = {}

        for hist in histories():
            
            # Ignore histories where it isn't player 1's turn or the game is over.
            calls = list(possible_calls(hist))
            if len(hist) % 2 == 1 or not calls:
                continue

            # Ignore histories where the probability of getting there is zero.
            if hist and cs.findCallProb(d1, hist[:-1]) < 1e-10:
                continue

            print('\tHist: {}'.format(stringify_hist(hist)))
            for call in calls:
                prob = cs.findCallProb(d1, hist + (call,))
                if prob > 1e-10:    # Only print non-zero probability moves.
                    print(f'\t\t{stringify_call(call)} = {prob * 100}%')
                    hist_to_move.setdefault(hist, []).append((call, prob))

        dice_to_hist_to_move_p1[d1] = hist_to_move

    # Normalize probs.
    for d1 in ROLLS1:
        for hist in dice_to_hist_to_move_p1[d1]:
            total = sum(p for _, p in dice_to_hist_to_move_p1[d1][hist])
            dice_to_hist_to_move_p1[d1][hist] = [(c, p/total) for c, p in dice_to_hist_to_move_p1[d1][hist]]
    
    # Print the probabilities of each move in every situation for player 2.
    dice_to_hist_to_move_p2 = {}
    for d2 in ROLLS2:
        print("\nDice: {}".format(d2))

        hist_to_move = {}

        for hist in histories():
            
            # Ignore histories where it isn't player 2's turn or the game is over.
            calls = list(possible_calls(hist))
            if len(hist) % 2 == 0 or not calls:
                continue

            # Ignore histories where the probability of getting there is zero.
            # if hist == (())
            if hist and all((cs.findCallProb(d1, hist) < 1e-10 for d1 in ROLLS1)):
                continue

            print('\tHist: {}'.format(stringify_hist(hist)))
            print('\t\tPlayer 1 dice estimate:', end=' ')
            probs = cs.estimateP1Rolls(hist)
            for i, p in enumerate(probs, 1):
                if p > 1e-10:
                    print(f'{i} = {p * 100}%', end=' ')
            print()
            best_move = cs.findP2Call(d2, hist)
            print(f'\t\tBest move = {stringify_call(best_move)}')

            hist_to_move[hist] = [(best_move, 1)]

        dice_to_hist_to_move_p2[d2] = hist_to_move


    # Test feasability
    # print('Score by d2')
    # tot_exp = 0
    # for d2 in ROLLS2:
    #    expected_score = sum(cs.stateValue(d1, d2, ()) for d1 in ROLLS2)/len(ROLLS2)
    #    tot_exp += expected_score
    #    print('{} strat {} >= lambda N/A'.format(d2, expected_score))
    # print('Total expected score', tot_exp / len(ROLLS2))

    # import pdb; pdb.set_trace()

    # print('Zs', ', '.join(str(zv.solution_value()) for zv in zvs.values()))
    # print('Zs', ', '.join(sorted('{}x{}: {}'.format(d2, xinner, zv.solution_value()) for (d2, xinner), zv in zvs.items())))
    # for (d1, x), xv in sorted(xvs.items(), key=
    #        lambda a:(a[0][0], len(a[0][1]), str(a))):
    #    print(d1, x, xv.solution_value())
    # print('Xs', ', '.join(str(xv.solution_value()) for xv in xvs.values()))

    res = sum(zv.solution_value() for (_, hist), zv in zvs.items() if hist == ())
    res /= len(ROLLS1)*len(ROLLS2)
    print('Value:', stringify_fraction(res))

    res2 = sum(cs.stateValue(d1, d2, ()) for d1 in ROLLS1 for d2 in ROLLS2)/len(ROLLS1)/len(ROLLS2)
    print('Score:', stringify_fraction(res2))

    # from liars_dice import SolvedAgent, Game

    # class SolverAgent(SolvedAgent):
    #     def __init__(self, table):
    #         super().__init__(name='SolverAgent', table=table)

    # class CounterAgent(SolvedAgent):
    #     def __init__(self, table):
    #         super().__init__(name='CounterAgent', table=table)

    # print('Simulating ideal agent against agents of varying skill.')

    # import random
    # random.seed(0)
    # t = time.time()

    # corruption_levels = [0, 0.1, 0.5, 1.0]
    # for corruption_level in corruption_levels:

    #     print('\n\nSimulating ideal agent against agents with {}% corruption.'.format(corruption_level*100))

    #     # Create a new table with the given corruption level.
    #     table = dice_to_hist_to_move_p2.copy()
    #     for d2 in ROLLS2:
    #         dice_table = {}
    #         for hist in dice_to_hist_to_move_p2[d2]:
    #             dice_table[hist] = dice_to_hist_to_move_p2[d2][hist]

    #             # Randomly corrupt the table.
    #             if random.random() < corruption_level:
    #                 move_options = list(possible_calls(hist))
    #                 move = random.choice(move_options)
    #                 dice_table[hist] = [(move, 1)]
                
    #         table[d1] = dice_table

    #     game = Game(
    #         [SolverAgent(dice_to_hist_to_move_p1), 
    #          CounterAgent(table)], n_dice=1, n_sides=SIDES)
    #     res = game.play_n(10000)
    #     print('Took {}s'.format(time.time()-t))

    #     # Print the percent of games won by the solver.
    #     print('Solver win rate:', sum(res)/len(res))

    #     # Print the statistical significance of the solver's win rate.
    #     print('Statistical significance:', binom_test(sum(res), len(res), 0.5))


if __name__ == '__main__':
    main()

