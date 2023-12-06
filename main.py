import numpy as np 
import functions as f
import matplotlib.pyplot as plt

""" setting of N and M """
N = 50 
M = 3

""" generate random binary paterns """

memorized_patterns = np.random.choice([-1,1], size = (M,N))

""" calculation of the weight matrix W using the Hebbian rule """
W = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            W[i, j] = (1 / M) * np.sum(memorized_patterns[µ, i] * memorized_patterns[µ, j] for µ in range(M))

print (W)

"""verification of the conditions for W"""
print ("the elements on the diagonal are 0 :", np.all(np.diag(W)==0))

transposee = np.array(np.transpose(W))
if np.array_equal(W,transposee) : 
    print ("W is symmetric")
else : 
    print ("W is not symmetric")

if np.all((W >= -1) & (W <= 1)):  
    print ("all the weights are in [−1, 1]")  
else : 
    print ("the weights are not in the range [-1,1]")

""" Generate a new pattern by changing 3 random values of one of the memorized patterns"""
initial_pattern_index = np.random.choice(range(M))
new_pattern = np.array([x for x in memorized_patterns[initial_pattern_index]])
new_values = np.random.choice(range(N), size=3)
new_pattern[new_values] *= (-1) 
""" verification """
print (np.all(memorized_patterns[initial_pattern_index]==new_pattern))

""" function definition for the update rule """
def update_rule(W, pattern):
    dot_product = np.dot(W, pattern)
    updated_pattern = np.ones_like(dot_product) 
    updated_pattern[dot_product < 0] = -1  
    return updated_pattern
    
""" Def of a loop which applies the update rule until convergence (or up
to a maximum of T = 20 iterations). """
T = 20 
convergence = False

for i in range (T): 
    updated_pattern = update_rule(W, new_pattern)
    if np.array_equal(updated_pattern, memorized_patterns[initial_pattern_index]):
        convergence = True
        break
    else : 
        i= i+1
        new_pattern = updated_pattern

""" Verify that the iterative process converged to the original memorized pattern """
print ("convergence :", convergence, "i:", i)


"-----------------------------TESTS V2-----------------------------------------------------------------"
print("TESTS V2:")

"Generate 80 random patterns of size 1000"
generated_patterns = f.generate_patterns (80,1000)
# print ("generated patterns :", generated_patterns)
print("size of generated patterns:", generated_patterns.shape)

"Choose a base pattern and perturb 200 of its elements"
perturbed_pattern = f.perturb_pattern(generated_patterns[17],200)
# print ("original pattern :", generated_patterns[])
# print ("perturbed pattern:", perturbed pattern)
print ("number of perturbations:", f.compare_patterns(generated_patterns[17], perturbed_pattern))

"Run the synchronous update rule until convergence, or up to 20 iterations"
weights = f.hebbian_weights(generated_patterns)
state_history1 , convergence1, energy1 = f.dynamics(perturbed_pattern, weights, 20) 
print ("synchronous update :")
#print ("state history:", state_history)
print("convergence:", convergence1)
print("pattern match:", f.pattern_match(generated_patterns,state_history1[-1]))

"Run the asynchronous update rule for a maximum of 20000 iterations, setting 3000 iterations without a change in the state as the convergence criterion"
state_history2, convergence2, energy2 = f.dynamics_async(perturbed_pattern,weights,30000,10000)
print("asynchronous update:")
#print("state history:", state_history)
print("convergence:", convergence2)
print("pattern match:", f.pattern_match(generated_patterns, state_history2[-1]))

"Test Storkey: check that it converges to the most similar memorized pattern"
print ("Test Storkey :")
weights = f.storkey_weights(generated_patterns)
state_history3, convergence3, energy3 = f.dynamics(perturbed_pattern, weights, 20)
print("convergence:", convergence3)
print("patern match:", f.pattern_match(generated_patterns,state_history3[-1]))


"-----------------------------TESTS ENERGY V3-----------------------------------"
print ("tests V3")

# # "Create 50 random patterns of (network) size 2500"
patterns = f.generate_patterns(50, 2500)
print("size of generated patterns:", patterns.shape)

# # """Perturb one of the memorized patterns by swapping the values of 1000 
# # of its elements (from -1 to 1 or vice versa)."""
perturbed_pattern = f.perturb_pattern(patterns[17], 1000)
print ("number of perturbations:", f.compare_patterns(patterns[17], perturbed_pattern))

# # """Store the random patterns in a Hopfield network and make the dynamical system evolve for
# # a maximum of 20 iterations or until convergence, first with the Hebbian weights and then with
# # the Storkey weights, with a synchronous update rule. Use the perturbed pattern as the initial
# # state. Store the state of the network at each time step."""
hebbian_weights = f.hebbian_weights(patterns)
storkey_weights = f.storkey_weights(patterns)

# # with the Hebbian weights: 
state_history_SH, convergence, energy_history_SH = f.dynamics(perturbed_pattern, hebbian_weights, 20)
print ("synchronous update with Hebbian weights:")
print("convergence:", convergence) 
print ("energy history: ", energy_history_SH)

# # with the Storkey weights:
state_history_SS, convergence, energy_history_SS = f.dynamics(perturbed_pattern, storkey_weights, 20)
print ("synchronous update with Storkey weights:")
print ("convergence:", convergence)
print ("energy history: ", energy_history_SS)


# # Run the dynamical system for a maximum of 30000 iterations or until convergence 
# # (we consider that convergence is reached when the state does not change for 10000 consecutive
# # iterations), first with the Hebbian weights and then with the Storkey weights, this time with
# # the asynchronous update rule. Store the state of the network at each time step."""

# with the Hebbian weights: 
state_history_AH, convergence, energy_history_AH = f.dynamics_async(perturbed_pattern, hebbian_weights, 30000, 10000)
print ("asynchronous update with Hebbian weights:")
print("convergence:", convergence) 
print ("energy history: ", energy_history_AH)

# with the Storkey weights:
state_history_AS, convergence, energy_history_AS = f.dynamics_async(perturbed_pattern, storkey_weights, 30000, 10000)
print ("asynchronous update with Storkey weights:")
print ("convergence:", convergence)
print ("energy history: ", energy_history_AS)


"""
Evaluate the energy function in each of the states. 
Verify that the function is always non increasing, both for the Hebbian weights and for the Storkey weights, 
and for both update rules, by generating a time-energy plot. 
"""
E = energy_history_SH
T = np.arange(0.0, len(energy_history_SH)*1000, 1000) 
plt.plot(T,E)
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.show() 

E = energy_history_SS
T = np.arange(0.0, len(energy_history_SS)*1000, 1000) 
plt.plot(T,E)
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.show() 

E = energy_history_AH
T = np.arange(0.0, len(energy_history_AH)*1000, 1000) 
plt.plot(T,E)
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.show() 

E = energy_history_AS
T = np.arange(0.0, len(energy_history_AS)*1000, 1000) 
plt.plot(T,E)
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.show() 


"""Flatten this matrix and store it in your pattern matrix, replacing one of the random patterns."""
patterns = f.generate_patterns(5, 2500)
checkerboard = f.generate_checkerboard(size_board=50, size_checker=5)
checkerboard_pattern = checkerboard.flatten()
index = (np.random.choice(patterns.shape[0])) 
patterns[index]=checkerboard_pattern

"""Generate the Hebbian weights based on the patterns that also include the checkerboard pattern."""
HW_checkerboard = f.hebbian_weights(patterns)

"""Perturb the checkerboard pattern by modifying 1000 of its elements."""
perturbed_pattern_C = f.perturb_pattern(pattern=patterns[index],num_perturb=1000)


"""
Run the dynamical system with the Hebbian weights, both with the synchronous and the
asynchronous update rule. Use the same maximum number of iterations and convergence
criterion as before. For the asynchronous update rule, remember not to store all the states,
but only one every 1000.
"""

# with the synchronous update: 
state_history_SHC, convergence, energy_history_SHC = f.dynamics(state=perturbed_pattern_C, weights=HW_checkerboard, max_iter=20)
#Reshape the stored states into 50 × 50 matrices.
reshaped_SHC = [state.reshape(50, 50) for state in state_history_SHC]

# with the asynchronous update:
state_history_AHC, convergence_AHC, energy_history_AHC = f.dynamics_async(state=perturbed_pattern_C.copy(), weights=HW_checkerboard, max_iter=30000, convergence_num_iter=10000)
selected_states = state_history_AHC[::1000]
#  access 1 state every 1000 states 
# Reshape the stored states into 50 × 50 matrices. 
reshaped_AHC = [(np.array(state)).reshape(50, 50) for state in selected_states]

# Use the function save video() to save a video of each of the experiments (synchronous and asynchronous execution).
# with the synchronous update: 
f.save_video(state_list=reshaped_SHC,out_path="output1.mp4")

# with the asynchronous update:
f.save_video(state_list=reshaped_AHC,out_path="output2.mp4")




