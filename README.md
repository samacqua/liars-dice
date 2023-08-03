# Liar's Dice

## Resources

- Papers
    - [Regret Minimization in Games with Incomplete Information (2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)
    - [Monte Carlo Sampling for Regret Minimization in Extensive Games (2009)](http://mlanctot.info/files/papers/nips09mccfr_techreport.pdf)
    - [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)

- Rules of Liar's Dice
    - [wikipedia](https://dl.acm.org/doi/10.5555/1109557.1109570)

- Thomas Ahle project using deep-CFR to play Liar's Dice
    - [Medium Article](https://towardsdatascience.com/lairs-dice-by-self-play-3bbed6addde0)
    - [Project Github](https://github.com/thomasahle/liars-dice)
    - ways to improve
        - using policy network instead of just a value network
        - searching at inference time
        - better encoding of state for value network (positional encoding?)

- Thomas Ahle article on using linear programming to solve Liar's Dice endgame
    - [https://github.com/thomasahle/snyd](https://towardsdatascience.com/lairs-dice-by-self-play-3bbed6addde0)
    - ways to improve
        - calculate w/ sequential rationality (solved using [this technique](http://www.sciencedirect.com/science/article/pii/089982569290035Q) which isn't sequentially rational, but [this work](https://dl.acm.org/doi/10.5555/1109557.1109570) is)
        - calculate for states with higher dies as likely more compute available / more efficient linear program solvers than the time of this project (7 years ago)