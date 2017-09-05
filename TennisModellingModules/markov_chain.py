"""
Markov chain.

Functions which implement Markov Chain model for the outcome of a Tennis match based on player service
chances. Includes test and comparison against simulated implementation in order to confirm accuracy

"""

import numpy as np
# import matplotlib.pyplot as plt


'----------------------------------------------------------------------------------------------------------------------'

'CLOSED FORM MARKOV CHAIN MODEL IMPLEMENTATION'

'----------------------------------------------------------------------------------------------------------------------'

def game_chance(serveChance): # closed form game chance
    return (serveChance**4)+4*(serveChance**4)*((1-serveChance)**1)+10*(serveChance**4)*((1-serveChance)**2)+\
           20*(serveChance**3)*((1-serveChance)**3)*(serveChance**2)*(1/(1-(2*serveChance*(1-serveChance))))


def tie_chance(p1serveChance, p2serveChance):    # closed form tie break chance
    return (p1serveChance**3)*((1-p2serveChance)**4)+4*(p1serveChance**4)*((1-p2serveChance)**3)*p2serveChance+\
        3*(p1serveChance**3)*((1-p2serveChance)**4)*(1-p1serveChance)+6*(p1serveChance**5)*((1-p2serveChance)**2)*(p2serveChance**2)+\
        16*(p1serveChance**4)*((1-p2serveChance)**3)*(1-p1serveChance) *(p2serveChance)+\
        6*(p1serveChance**3)*((1-p2serveChance)**4)*((1-p1serveChance)**2)+\
        4*(p1serveChance**5)*((1-p1serveChance) **0)*((1-p2serveChance)**2)*(p2serveChance**3)+\
        30*(p1serveChance**4)*((1-p1serveChance) **1)*((1-p2serveChance)**3)*(p2serveChance**2)+\
        40*(p1serveChance**3)*((1-p1serveChance) **2)*((1-p2serveChance)**4)*(p2serveChance**1)+\
        10*(p1serveChance**2)*((1-p1serveChance) **3)*((1-p2serveChance)**5)*(p2serveChance**0)+\
        5*(p1serveChance**5)*((1-p1serveChance) **0)*((1-p2serveChance)**2)*(p2serveChance**4)+\
        50*(p1serveChance**4)*((1-p1serveChance) **1)*((1-p2serveChance)**3)*(p2serveChance**3)+\
        100*(p1serveChance**3)*((1-p1serveChance) **2)*((1-p2serveChance)**4)*(p2serveChance**2)+\
        50*(p1serveChance**2)*((1-p1serveChance) **3)*((1-p2serveChance)**5)*(p2serveChance**1)+\
        5*(p1serveChance**1)*((1-p1serveChance) **4)*((1-p2serveChance)**6)*(p2serveChance**0)+\
        6*(p1serveChance**6)*((1-p1serveChance) **0)*((1-p2serveChance)**1)*(p2serveChance**5)+\
        75*(p1serveChance**5)*((1-p1serveChance) **1)*((1-p2serveChance)**2)*(p2serveChance**4)+\
        200*(p1serveChance**4)*((1-p1serveChance) **2)*((1-p2serveChance)**3)*(p2serveChance**3)+\
        150*(p1serveChance**3)*((1-p1serveChance) **3)*((1-p2serveChance)**4)*(p2serveChance**2)+\
        30*(p1serveChance**2)*((1-p1serveChance) **4)*((1-p2serveChance)**5)*(p2serveChance**1)+\
        1*(p1serveChance**1)*((1-p1serveChance) **5)*((1-p2serveChance)**6)*(p2serveChance**0)+\
        (1*(p1serveChance**6)*((1-p1serveChance) **0)*((1-p2serveChance)**0)*(p2serveChance**6)+
         36*(p1serveChance**5)*((1-p1serveChance) **1)*((1-p2serveChance)**1)*(p2serveChance**5)+
         225*(p1serveChance**4)*((1-p1serveChance) **2)*((1-p2serveChance)**2)*(p2serveChance**4)+
         400*(p1serveChance**3)*((1-p1serveChance) **3)*((1-p2serveChance)**3)*(p2serveChance**3)+
         225*(p1serveChance**2)*((1-p1serveChance) **4)*((1-p2serveChance)**4)*(p2serveChance**2)+
         36*(p1serveChance**1)*((1-p1serveChance) **5)*((1-p2serveChance)**5)*(p2serveChance**1)+
         1*(p1serveChance**0)*((1-p1serveChance) **6)*((1-p2serveChance)**6)*(p2serveChance**0))*\
        (p1serveChance*(1-p2serveChance))*(1/(1-(p1serveChance*p2serveChance+(1-p1serveChance)*(1-p2serveChance))))


def set_chance(p1Tie, p1Game,p2Game):    # closed form set chance
    return (p1Game**3)*((1-p2Game)**3)+3*(p1Game**4)*((1-p2Game)**2)*p2Game+3*(p1Game**3)*((1-p2Game)**3)*(1-p1Game)+\
        3*(p1Game**4)*((1-p2Game)**2)*(p2Game**2)+12*(p1Game**3)*((1-p2Game)**3)*(1-p1Game)*(p2Game)+\
        6*(p1Game**2)*((1-p2Game)**4)*((1-p1Game)**2)+4*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**1)*(p2Game**3)+\
        24*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**2)*(p2Game**2)+24*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**3)*(p2Game**1)+\
        4*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**4)*(p2Game**0)+1*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**1)*(p2Game**4)+\
        20*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**2)*(p2Game**3)+60*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**3)*(p2Game**2)+\
        40*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**4)*(p2Game**1)+5*(p1Game**1)*((1-p1Game)**4)*((1-p2Game)**5)*(p2Game**0)+\
        (1*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**0)*(p2Game**5)+25*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**1)*(p2Game**4)+
         100*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**2)*(p2Game**3)+100*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**3)*(p2Game**2)+
         25*(p1Game**1)*((1-p1Game)**4)*((1-p2Game)**4)*(p2Game**1)+1*(p1Game**0)*((1-p1Game)**5)*((1-p2Game)**5)*(p2Game**0))*(p1Game)*(1-p2Game)+\
        (1*(p1Game**5)*((1-p1Game)**0)*((1-p2Game)**0)*(p2Game**5)+25*(p1Game**4)*((1-p1Game)**1)*((1-p2Game)**1)*(p2Game**4)+
         100*(p1Game**3)*((1-p1Game)**2)*((1-p2Game)**2)*(p2Game**3)+100*(p1Game**2)*((1-p1Game)**3)*((1-p2Game)**3)*(p2Game**2)+
         25*(p1Game**1)*((1-p1Game)**4)*((1-p2Game)**4)*(p2Game**1)+
         1*(p1Game**0)*((1-p1Game)**5)*((1-p2Game)**5)*(p2Game**0))*(p1Game*p2Game+(1-p1Game)*(1-p2Game))*p1Tie


def match_chance_sets(p1set,no_sets=3): # closed form match chance from game probabilities
    if no_sets==3:
        return (p1set**2)+2*(p1set**2)*(1-p1set)
    else: # otherwise assume 5 set match
        return (p1set**3)+3*(p1set**3)*(1-p1set)+6*(p1set**3)*((1-p1set)**2)


def match_chance_games(p1Game,p2Game,no_sets=3): # closed form match chance from game probabilities
    p1Tie = (p1Game + 1- p2Game)/2.
    p1set = set_chance(p1Tie,p1Game,p2Game)
    if no_sets==3:
        return (p1set**2)+2*(p1set**2)*(1-p1set)
    else: # otherwise assume 5 set match
        return (p1set**3)+3*(p1set**3)*(1-p1set)+6*(p1set**3)*((1-p1set)**2)


def match_chance(p1serve,p2serve,no_sets=3): # closed form match chance
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


def sim_game(sc):   # outcome of a game
    p1=0
    p2=0
    while(1):
        if (np.random.uniform(0, 1) < sc):
            p1+=1
        else:
            p2+=1
        if(p1 > p2+1 and p1 > 3):
            return 1
        elif(p2 > p1+1 and p2 > 3):
            return 0


def sim_tie(p1serve,p2serve):  # outcome of a tie break, assumes player 1 serves first
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


def sim_set(p1serve,p2serve):   # outcome of a set
    p1=0
    p2=0
    sc = p1serve    # Player 1 start serving
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


def sim_m(p1serve,p2serve,no_sets=3):   # outcome of a game
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
    count =0
    for i in range(no_loops):
        count += sim_m(p1serve,p2serve,no_sets)
    return count/no_loops


'----------------------------------------------------------------------------------------------------------------------'

'CROSS CHECK'

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
#         m2.append(sim_match(p1,p2s,5000))
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
