import numpy as np
import construct_MP3 as cmp
import design_controller4 as K_des
import matplotlib as plt
from MC_construct2 import call_MC
# from figure_plot import probability_plot
import time
import json
import sys
sys.setrecursionlimit(10000)

def initialize(vmax, MAX_V):
    Ncar = int(MAX_V*(MAX_V+1)/2 + 10)
    Vlow=  0
    Vhigh = vmax
    x_vmax_stop = MAX_V*(MAX_V+1)/2 + 1
    xcross_start = 2
    Nped = Ncar - xcross_start + 1
    if x_vmax_stop >= xcross_start:
        min_xped = int(x_vmax_stop + 1 - (xcross_start - 1))
    else:
        min_xped = 3
    assert(min_xped > 0)
    assert(min_xped<= Nped)
    if min_xped < Nped:
        xped = np.random.randint(min_xped, Nped)
    else:
        xped = int(min_xped)
    xped = int(min_xped)
    xcar_stop = xped + xcross_start - 2
    assert(xcar_stop > 0)
    state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1)+v
    bad_states = set()
    good_state = set()
    def get_formula_states(xcar_stop):
        bst = set()
        for vi in range(0,Vhigh+1):
            state = state_f(xcar_stop, vi)
            bst |= {"S"+str(state)}
        gst = {"S" + str(state_f(xcar_stop,0))}
        bad = "" # Expression for bad states
        good = "" # Expression for good states
        for st in list(gst):
            if good == "":
                good = good + "\"" + st+"\""
            else:
                good = good + "|\""+st+"\""
        for st in list(bst):
            if bad == "":
                bad = bad + "\"" + st+"\""
            else:
                bad = bad + "|\""+st+"\""
        return good, bad, gst, bst
    good, bad, gst, bst = get_formula_states(xcar_stop)
    good_state |= gst
    bad_states |= bst
    formula = "P=?[!("+str(bad)+") U "+str(good)+"]"
    
    phi1 = "!("+good+")"
    phi2 = "("+good+") | !("+bad
    for xcar_ii in range(xcar_stop+1, Ncar+1):
        good, bad, gst, bst = get_formula_states(xcar_ii) # We only want the bad states; ignore the good states output here
        bad_states |= bst
        phi2 = phi2 + "|" + bad
    phi2 = phi2 + ")"
    formula = "P=?[G("+str(phi1)+") && G("+str(phi2)+")]"
    return Ncar, Vlow, Vhigh, xcross_start, xped, bad_states, good_state, formula

prec = [0.95, 0.9, 0.82, 0.7, 0.4]
recall = [0.2, 0.4, 0.6, 0.8, 0.9]

Np = len(prec)
ex= 1
if ex == 1:
    VMAX = []
    INIT_V = dict()
    P = dict()
    MAX_V = 10
    for ip in range(Np):
        prec_i = prec[ip]
        rec_i = recall[ip]
        print("Precision: ")
        print(prec_i)
        print("Recall: ")
        print(rec_i)
        INIT_V[ip] = dict()
        P[ip] = dict()
        for vmax in range(4,6):
            INIT_V[ip][vmax] = []
            P[ip][vmax] = []
            print("===========================================================")
            print("Max Velocity: ", vmax)
            # Initial conditions set for all velocities
            Ncar, Vlow, Vhigh, xcross_start, xped, bad_states, good_state, formula = initialize(vmax, MAX_V)
            print("Specification: ")
            print(formula)
            for vcar in range(1, vmax+1):  # Initial speed at starting point
                state_f = lambda x,v: (Vhigh-Vlow+1)*(x-1) + v
                start_state = "S"+str(state_f(1,vcar))
                print(start_state)
                S, state_to_S, K_backup = cmp.system_states_example_ped(Ncar, Vlow, Vhigh)
                C = cmp.confusion_matrix_ped2(prec_i, rec_i)
                K = K_des.construct_controllers(Ncar, Vlow, Vhigh, xped, vcar, xcross_start)
                true_env = str(1) #Sidewalk 3
                true_env_type = "empty"
                O = {"ped", "obj", "empty"}
                state_info = dict()
                state_info["start"] = start_state
                state_info["bad"] = bad_states
                state_info["good"] = good_state
                # Formula for when true_env = "obj", the requirement is to never come to a stop G!(st)
                # Formula for when true_env = "empty", the requirement is to never go slower than the initial speed: G(v >= v_0)
                for st in list(good_state):
                    formula2 = 'P=?[G!(\"'+st+'\")]'
                # for st in list(good_state):
                #    formula2 = 'P=?[G!(\"'+st+'\")]'
                M = call_MC(S, O, state_to_S, K, K_backup, C, true_env, true_env_type, state_info)
                # result = M.prob_TL(formula)
                result2 = M.prob_TL(formula2)
                print('Probability of eventually reaching good state for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, vmax, result2[start_state]))
                # Store results:
                VMAX.append(vmax)
                INIT_V[ip][vmax].append(vcar)
                # p = result[start_state]
                # print('Probability of satisfaction for initial speed, {}, and max speed, {} is p = {}:'.format(vcar, vmax, p))
                P[ip][vmax].append(result2[start_state])



# Write to json file:
timestr = time.strftime("%Y%m%d-%H%M%S")
fname_v = "type_"+str(ex)+"_"+"init_v_" + timestr+"_.json"
fname_p = "type_"+str(ex)+"_"+"prob_" + timestr+"_.json"
fname_v = "test_prec_rec_empty_vmax_4_5_initv.json"
fname_p = "test_prec_rec_empty_vmax_4_5_prob.json"
with open(fname_v, 'w') as f:
    json.dump(INIT_V, f)
with open(fname_p, 'w') as f:
    json.dump(P, f)
