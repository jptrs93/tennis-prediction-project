"""
Markov chain.

Functions which implement Markov Chain model for the outcome of a Tennis match based on player service
chances. Includes test and comparison against simulated implementation in order to confirm accuracy

"""

import numpy as np
import matplotlib.pyplot as plt


'----------------------------------------------------------------------------------------------------------------------'

'CLOSED FORM MARKOV CHAIN MODEL IMPLEMENTATION'

'----------------------------------------------------------------------------------------------------------------------'


def game_chance(serveChance):
    """
    Returns the probability of a player winning a service game based upon the probability of them winning a point
    on their serve. Assumes points are independent and identically distributed.

    Args:
        serveChance (float [0,1]) : Probability of the serving player winning a point
    Returns
        gameChance (float [0,1]) : Probability of the serving player winning a game
    """
    # Chance to win 40-0
    p40_0 = serveChance**4

    # Chance to win 40-15
    p40_15 = 4*(serveChance**4)*((1-serveChance)**1)

    # Chance to win 40-30
    p40_30 = 10*(serveChance**4)*((1-serveChance)**2)

    # Chance to win after deuce
    pend = 20*(serveChance**3)*((1-serveChance)**3)*(serveChance**2)*(1/(1-(2*serveChance*(1-serveChance))))

    return p40_0 + p40_15 + p40_30 + pend


def tie_chance(p1serveChance, p2serveChance):
    """
    Returns the probability a player winning a tie break based upon the probabilities of both players winning a
    point on their serve. Assumes points are independent and identically distributed.
    
    Args:
        p1serveChance (float [0,1]) : Probability of player 1 winning a point on their serve
        p2serveChance (float [0,1]) : Probability of player 2 winning a point on their serve
    Returns:
        p1tieChance (float [0,1]) : Probability of player 1 winning a tie break
    """

    # Chance to win 7-0
    p7_0 = (p1serveChance**3)*((1-p2serveChance)**4)

    # Chance to win 7-1
    p7_1 = 4*(p1serveChance**4)*((1-p2serveChance)**3)*p2serveChance+\
           3*(p1serveChance**3)*((1-p2serveChance)**4)*(1-p1serveChance)

    # Chance to win 7-2
    p7_2 = 6*(p1serveChance**5)*((1-p2serveChance)**2)*(p2serveChance**2)+\
           16*(p1serveChance**4)*((1-p2serveChance)**3)*(1-p1serveChance)*(p2serveChance)+\
           6*(p1serveChance**3)*((1-p2serveChance)**4)*((1-p1serveChance)**2)

    # Chance to win 7-3
    p7_3 = 4*(p1serveChance**5)*((1-p1serveChance) **0)*((1-p2serveChance)**2)*(p2serveChance**3)+\
           30*(p1serveChance**4)*((1-p1serveChance) **1)*((1-p2serveChance)**3)*(p2serveChance**2)+\
           40*(p1serveChance**3)*((1-p1serveChance) **2)*((1-p2serveChance)**4)*(p2serveChance**1)+\
           10*(p1serveChance**2)*((1-p1serveChance) **3)*((1-p2serveChance)**5)*(p2serveChance**0)

    # Chance to win 7-4
    p7_4 = 5*(p1serveChance**5)*((1-p1serveChance) **0)*((1-p2serveChance)**2)*(p2serveChance**4)+\
           50*(p1serveChance**4)*((1-p1serveChance) **1)*((1-p2serveChance)**3)*(p2serveChance**3)+\
           100*(p1serveChance**3)*((1-p1serveChance) **2)*((1-p2serveChance)**4)*(p2serveChance**2)+\
           50*(p1serveChance**2)*((1-p1serveChance) **3)*((1-p2serveChance)**5)*(p2serveChance**1)+\
           5*(p1serveChance**1)*((1-p1serveChance) **4)*((1-p2serveChance)**6)*(p2serveChance**0)

    # Chance to win 7-5
    p7_5 = 6*(p1serveChance**6)*((1-p1serveChance) **0)*((1-p2serveChance)**1)*(p2serveChance**5)+\
           75*(p1serveChance**5)*((1-p1serveChance) **1)*((1-p2serveChance)**2)*(p2serveChance**4)+\
           200*(p1serveChance**4)*((1-p1serveChance) **2)*((1-p2serveChance)**3)*(p2serveChance**3)+\
           150*(p1serveChance**3)*((1-p1serveChance) **3)*((1-p2serveChance)**4)*(p2serveChance**2)+\
           30*(p1serveChance**2)*((1-p1serveChance) **4)*((1-p2serveChance)**5)*(p2serveChance**1)+\
           1*(p1serveChance**1)*((1-p1serveChance) **5)*((1-p2serveChance)**6)*(p2serveChance**0)

    # Chance to win after 6-6
    p_end = (1*(p1serveChance**6)*((1-p1serveChance) **0)*((1-p2serveChance)**0)*(p2serveChance**6)+
             36*(p1serveChance**5)*((1-p1serveChance) **1)*((1-p2serveChance)**1)*(p2serveChance**5)+
             225*(p1serveChance**4)*((1-p1serveChance) **2)*((1-p2serveChance)**2)*(p2serveChance**4)+
             400*(p1serveChance**3)*((1-p1serveChance) **3)*((1-p2serveChance)**3)*(p2serveChance**3)+
             225*(p1serveChance**2)*((1-p1serveChance) **4)*((1-p2serveChance)**4)*(p2serveChance**2)+
             36*(p1serveChance**1)*((1-p1serveChance) **5)*((1-p2serveChance)**5)*(p2serveChance**1)+
             1*(p1serveChance**0)*((1-p1serveChance) **6)*((1-p2serveChance)**6)*(p2serveChance**0))*\
            (p1serveChance*(1-p2serveChance))*(1/(1-(p1serveChance*p2serveChance+(1-p1serveChance)*(1-p2serveChance))))

    return p7_0 + p7_1 + p7_2 + p7_3 + p7_4 + p7_5 + p_end


def set_chance(p1Tie, p1Game,p2Game):
    """
    Returns the probability a player winning a set based upon the probabilities of both players winning service games 
    and a tie break. Assumes games are independent and identically distributed.

    Args:
        p1Tie (float [0,1]) : Probability of player 1 winning a tie break
        p1Game (float [0,1]) : Probability of player 1 winning a game on their serve
        p2Game (float [0,1]) : Probability of player 1 winning a game on their serve
    Returns:
        p1tieChance (float [0,1]) : Probability of player 1 winning a set break
    """

    # Chance to win 6-0
    p6_0 = (p1Game**3)*((1-p2Game)**3)

    # Chance to win 6-1
    p6_1 = 3*(p1Game**4)*((1-p2Game)**2)*p2Game+\
           3*(p1Game**3)*((1-p2Game)**3)*(1-p1Game)

    # Chance to win 6-2
    p6_2 = 3*(p1Game**4)*((1-p2Game)**2)*(p2Game**2)+\
           12*(p1Game**3)*((1-p2Game)**3)*(1-p1Game)*(p2Game)+\
           6*(p1Game**2)*((1-p2Game)**4)*((1-p1Game)**2)

    # Chance to win 6-3
    p6_3 = 4*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**1)*(p2Game**3)+\
           24*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**2)*(p2Game**2)+\
           24*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**3)*(p2Game**1)+\
           4*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**4)*(p2Game**0)

    # Chance to win 6-4
    p6_4 = 1*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**1)*(p2Game**4)+\
           20*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**2)*(p2Game**3)+\
           60*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**3)*(p2Game**2)+\
           40*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**4)*(p2Game**1)+\
           5*(p1Game**1)*((1-p1Game)**4)*((1-p2Game)**5)*(p2Game**0)

    # Chance to win 7-5
    p7_5 = (1*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**0)*(p2Game**5)+25*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**1)*(p2Game**4)+
            100*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**2)*(p2Game**3)+100*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**3)*(p2Game**2)+
            25*(p1Game**1)*((1-p1Game)**4)*((1-p2Game)**4)*(p2Game**1)+1*(p1Game**0)*((1-p1Game)**5)*((1-p2Game)**5)*(p2Game**0))*(p1Game)*(1-p2Game)

    # Chance to win 7-6
    p7_6 = (1*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**0)*(p2Game**5)+25*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**1)*(p2Game**4)+
            100*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**2)*(p2Game**3)+100*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**3)*(p2Game**2)+
            25*(p1Game**1)*((1-p1Game)**4)*((1-p2Game)**4)*(p2Game**1)+
            1*(p1Game**0)*((1-p1Game)**5)*((1-p2Game)**5)*(p2Game**0))*(p1Game*p2Game+(1-p1Game)*(1-p2Game))*p1Tie

    return p6_0 + p6_1 + p6_2 + p6_3 + p6_4 + p7_5 + p7_6

def match_chance_sets(p1set,no_sets=3): # closed form match chance from game probabilities
    """
    Returns the probability a player winning a match based upon the probabilities of a player winning a set.
    Assumes sets are independent and identically distributed.

    Args:
        p1set (float [0,1]) : Probability of player 1 winning a set
        no_sets (integer) : 5 or 3 - the number of sets the match is out of
    Returns:
        p1Match (float [0,1]) : Probability of player 1 winning the match
    """
    if no_sets==3:
        return (p1set**2)+2*(p1set**2)*(1-p1set)
    else:    # otherwise assume 5 set match
        return (p1set**3)+3*(p1set**3)*(1-p1set)+6*(p1set**3)*((1-p1set)**2)


def match_chance_games(p1Game,p2Game,no_sets=3):
    """
    Returns the probability a player winning a match based upon the probabilities either player winning a game on 
    their serve. Assumes games are independent and identically distributed.

    Args:
        p1Game (float [0,1]) : Probability of player 1 winning a game on their serve
        p2Game (float [0,1]) : Probability of player 1 winning a game on their serve
        no_sets (integer) : 5 or 3 - the number of sets the match is out of
    Returns:
        p1Match (float [0,1]) : Probability of player 1 winning the match
    """
    p1Tie = (p1Game + 1- p2Game)/2. # average and use as tiebreak chance
    p1set = set_chance(p1Tie,p1Game,p2Game)
    if no_sets==3:
        return (p1set**2)+2*(p1set**2)*(1-p1set)
    else: # otherwise assume 5 set match
        return (p1set**3)+3*(p1set**3)*(1-p1set)+6*(p1set**3)*((1-p1set)**2)


def match_chance(p1serve,p2serve,no_sets=3):
    """
    Returns the probability a player winning a match based upon the probabilities either player winning a point on 
    their serve. Assumes points are independent and identically distributed.

    Args:
        p1serveChance (float [0,1]) : Probability of player 1 winning a point on their serve
        p2serveChance (float [0,1]) : Probability of player 2 winning a point on their serve
        no_sets (integer) : 5 or 3 - the number of sets the match is out of
    Returns:
        p1Match (float [0,1]) : Probability of player 1 winning the match
    """
    p1Game = game_chance(p1serve)
    p2Game = game_chance(p2serve)
    p1Tie = tie_chance(p1serve,p2serve)
    p1set = set_chance(p1Tie,p1Game,p2Game)
    if no_sets==3:
        return (p1set**2)+2*(p1set**2)*(1-p1set)
    else: # otherwise assume 5 set match
        return (p1set**3)+3*(p1set**3)*(1-p1set)+6*(p1set**3)*((1-p1set)**2)


'----------------------------------------------------------------------------------------------------------------------'

'SIMULATED MARKOV CHAIN MODEL IMPLEMENTATION (Slower)'

'----------------------------------------------------------------------------------------------------------------------'


def sim_game(p1serve):
    """
    Simulates a tennis game.
    
    Args:
        p1serve (float [0,1]) : Probability of the serving player winning a point
    Returns
        outcome (bool) : Whether the server won the game
    """
    p1=0
    p2=0
    while(1):
        if (np.random.uniform(0, 1) < p1serve):
            p1+=1
        else:
            p2+=1
        if(p1 > p2+1 and p1 > 3):
            return 1
        elif(p2 > p1+1 and p2 > 3):
            return 0


def sim_tie(p1serve,p2serve):
    """
    Simulates a tie break.

    Args:
        p1serve (float [0,1]) : Probability of player 1 winning a point on their serve
        p2serve (float [0,1]) : Probability of player 2 winning a point on their serve
    Returns
        outcome (bool) : Whether the player 1 won or lost the tie break
    """
    p1=0
    p2=0
    serve_switch = 1
    sc = p1serve    # Player 1 start serving
    while(1):
        if (np.random.uniform(0, 1) < sc):
            if(sc == p1serve):  # if player 1's serve
                p1 += 1
            else:
                p2 += 1
        else:
            if(sc == p1serve):  # if player 1's serve
                p2 += 1
            else:
                p1 += 1
        if(p1 > p2+1 and p1 > 6):
            return 1
        elif(p2 > p1+1 and p2 > 6):
            return 0
        serve_switch +=1
        if(serve_switch == 2):
            serve_switch = 0
            if(sc == p1serve):
                sc = p2serve
            else:
                sc = p1serve


def sim_set(p1serve,p2serve):
    """
    Simulates a set.

    Args:
        p1serve (float [0,1]) : Probability of player 1 winning a point on their serve
        p2serve (float [0,1]) : Probability of player 2 winning a point on their serve
    Returns
        outcome (bool) : Whether the player 1 won or lost the set
    """
    p1=0
    p2=0
    sc = p1serve    # Assume player 1 starts serving
    while(1):
        if(sc == p1serve):
            w = sim_game(sc)
            p1 += w
            p2 += 1-w
            sc = p2serve
        else:
            w = sim_game(sc)
            p2 += w
            p1 += 1-w
            sc = p1serve
        if(p1 > p2+1 and p1 > 5):
            return 1
        elif(p2 > p1+1 and p2 > 5):
            return 0
        elif(p2 == 6 and p1 ==6):
            return sim_tie(p1serve,p2serve)


def sim_m(p1serve,p2serve,no_sets=3):
    """
    Simulates a match.

    Args:
        p1serve (float [0,1]) : Probability of player 1 winning a point on their serve
        p2serve (float [0,1]) : Probability of player 2 winning a point on their serve
        no_sets (integer) : 5 or 3 - the number of sets the match is out of
    Returns
        outcome (bool) : Whether the player 1 won or lost the match
    """
    p1=0
    p2=0
    while(1):
        w =sim_set(p1serve,p2serve)
        p1+=w
        p2+=1-w
        if(p1 > p2 and p1 > no_sets-2):
            return 1
        elif(p2 > p1 and p2 > no_sets-2):
            return 0


def sim_match(p1serve,p2serve, no_loops=2000, no_sets=3):
    """
    Simulates a large number of matches and averages the outcome to give the probability of a player winning a match.

    Args:
        p1serve (float [0,1]) : Probability of player 1 winning a point on their serve
        p2serve (float [0,1]) : Probability of player 2 winning a point on their serve
        no_sets (integer) : 5 or 3 - the number of sets the match is out of
        no_loops (integer) : The number of trials to make
    Returns
        MatchProb (float [0,1]) : Probability of player 1 winning the match
    """
    count =0
    for i in range(no_loops):
        count += sim_m(p1serve,p2serve,no_sets)
    return count/no_loops


'----------------------------------------------------------------------------------------------------------------------'

'Test to compare simulated against closed form'

'----------------------------------------------------------------------------------------------------------------------'


# def test():
#     """Runs a comparison between the simulated markov chain and closed form"""
#     plt.close('All')
#     fig_1 = plt.figure(figsize=(12, 6))  # create a figure and axes
#     ax_1 = fig_1.add_subplot(111)
#
#     p2s = 0.5
#     p1s = np.arange(0,1,1/200)
#     m = match_chance(p1s,p2s)
#     m2 = []
#     for p1 in p1s:
#         m2.append(sim_match(p1,p2s,300))
#     ax_1.plot(p1s,m,label="closed form")
#     ax_1.plot(p1s,m2,label="simulated")
#     ax_1.set_title("Match Chance vs Serve Chance (for fixed opponent serve chance of 0.5")
#     ax_1.set_ylim(-0.1,1.1)
#     ax_1.legend(loc='best')
#     plt.show()


#
#
# def test2():
#     """Extra troubleshooting from when there was an error in the closed form solution (fixed now)"""
#     plt.close('All')
#     fig_1 = plt.figure(figsize=(12, 6))  # create a figure and axes
#     ax_1 = fig_1.add_subplot(111)
#
#     p2s = 0.5
#     p1s = np.arange(0,1,1/200)
#     mg = game_chance(p1s)
#     mt = tie_chance(p1s, p2s)
#     ms =[]
#
#     m2g = []
#     m2t = []
#     m2s =[]
#
#     for p1 in p1s:
#         no_loops=500
#         countg = 0
#         countt = 0
#         counts =0
#         ms.append(set_chance(tie_chance(p1,p2s),game_chance(p1),game_chance(p2s)))
#         for i in range(no_loops):
#             countg += sim_game(p1)
#             countt += sim_tie(p1,p2s)
#             counts += sim_set(p1,p2s)
#         m2g.append(countg/no_loops)
#         m2t.append(countt/ no_loops)
#         m2s.append(counts / no_loops)
#     ax_1.plot(p1s,mg,label="closed form game")
#     ax_1.plot(p1s, mt, label="closed form tie")
#     ax_1.plot(p1s, ms, label="closed form set")
#     ax_1.plot(p1s, m2g, label="simulated game")
#     ax_1.plot(p1s, m2t, label="simulated tie")
#     ax_1.plot(p1s, m2s, label="simulated set")
#     ax_1.set_title("Match Chance vs Serve Chance (for fixed oppenent serve chance of 0.5")
#     ax_1.set_ylim(-0.1,1.1)
#     ax_1.legend(loc='best')
#     plt.show()
