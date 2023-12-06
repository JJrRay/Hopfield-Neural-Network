import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.animation as animation

def generate_patterns (num_patterns, pattern_size):
   """
   Generate the patterns to memorize of binary values (-1 or 1).

   Parameters
   ----------
   num_patterns (int): Number of patterns to generate
   pattern_size (int): Size of each pattern

   Returns
   -------
   generated_patterns (numpy.ndarray): 2D array of the generated patterns
   each row represents a pattern, and columns represent individual elements (-1 or 1)
   
   Example
   ----------
   >>> (generate_patterns(3,4)).shape
   (3, 4)

   This example generates 3 patterns, each with a size of 4 elements. 
   """
   # Handle the case when num_patterns and/or pattern_size are 0 or negative
   if num_patterns <= 0 or pattern_size <= 0:    
      raise ValueError("num_patterns and pattern_size must be positive integers.")
   
   # Generate patterns 
   generated_patterns = np.array(np.random.choice([-1,1], size = (num_patterns, pattern_size))) 
   return generated_patterns


def perturb_pattern(pattern, num_perturb):
   """
   Perturb a given pattern by changing the signs of randomly selected elements.

   Parameters
   -----------
   pattern (numpy array): The orginal pattern to perturb
   num_pertub (int): The number of element to perturb

   Returns
   -------
   perturbed_pattern (numpy array): the perturbed pattern with randomly changed signs
    
   """
   # Error check : 

   # Ensure that the number of perturbations is a positive integer
   if not isinstance(num_perturb, int) or num_perturb <= 0:
      raise ValueError("Number of perturbations must be a positive integer.")
   
   # Ensure that the nb of perturbations is not greater than the length of the pattern
   if num_perturb > len(pattern):
      raise ValueError("Number of perturbations cannot be greater than the length of the pattern.")
   
   # Perturb the pattern : 
   perturbed_pattern = pattern.copy()
   indices_perturbations = np.random.choice(len(pattern), num_perturb, replace=False)
   perturbed_pattern[indices_perturbations] *= -1
   return perturbed_pattern


def pattern_match(memorized_patterns, pattern):
   """
   Match a pattern with the corresponding memorized one.

   Parameters
   -----------
   memorized_patterns (list of numpy array): list containing the memorized patterns
   pattern (numpy array): the pattern to be matched with the memorized patterns
    
   Returns
   -----------
   int or None : If a match is found, it returns the index of the matching pattern. Otherwise, it returns None.
   """
   # Error check
   # Ensure that the pattern is a numpy array :
   if not isinstance(pattern, np.ndarray):
      raise TypeError("Pattern must be a numpy array.")
   
   # Ensure that the pattern is not empty :
   if len(pattern) == 0:
      raise ValueError("Pattern can't be empty.")
   
   # Ensure that the list of memorized patterns is not empty : 
   if len(memorized_patterns)==0:
      raise ValueError("List of memorized_patterns can't be empty")
   
   # Ensure that the lenght of the memorized patterns match the one of the pattern : 
   if not all(len(memorized_pattern) == len(pattern) for memorized_pattern in memorized_patterns):
        raise ValueError("Length of the memorized patterns must match the length of the pattern.")

   # Pattern match :
   for i, memorized_patterns in enumerate(memorized_patterns):
      if np.allclose(memorized_patterns,pattern):
         return i
   return None
  

def hebbian_weights(patterns):
   """
   Apply the hebbian learning rule on some given patterns to create the weight matrix.

   Parameters
   -----------
   patterns (numpy array): a 2D array where each row represents a binary pattern

   Returns
   -------
   W (numpy array): the Hebbian weight matrix (computed with the Hebbian rule)

   The function takes a matrix of binary patterns as input and computes the weights matrix with the Hebbian rule.
   For each pair of neurons (i, j), the weight W[i, j] is calculated as the average product of their activities
   over all patterns. The diagonal weights (i == j) are set to zero.

   Example 
   --------
   >>> hebbian_weights( np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]]))
   array([[ 0.        ,  0.33333333, -0.33333333, -0.33333333],
          [ 0.33333333,  0.        , -1.        ,  0.33333333],
          [-0.33333333, -1.        ,  0.        , -0.33333333],
          [-0.33333333,  0.33333333, -0.33333333,  0.        ]])
   """
   # Error checks
   if (len(patterns) == 0):
      raise ValueError("Patterns can't be empty.")
    
   if not isinstance(patterns, np.ndarray) or patterns.ndim != 2:
      raise TypeError("Patterns should be a 2D numpy array.")

   if not np.all(np.logical_or(patterns == 1, patterns == -1)):
      raise ValueError("Patterns must be binary.")
   
  # Check if patterns are unique
   unique_patterns, indices, counts = np.unique(patterns, axis=0, return_counts=True, return_index=True)
   if len(unique_patterns) != patterns.shape[0]:
      # There are duplicate patterns
      raise ValueError("Generated patterns are not unique.")
   
   # Compute the Hebbian weights 
   M, N = np.shape(patterns)
   W = (1 / M) * np.dot(patterns.T, patterns)
   np.fill_diagonal(W, 0)

   # Check that the elements on the diagonal of the resulting Hebbian weights matrix are 0
   if np.any(np.diagonal(W) != 0):
      raise ValueError("Diagonal of the weights matrix are not zero.")
   
   # Check for symmetry
   transposee = np.transpose(W.copy())
   if not np.allclose(W, transposee):
      raise ValueError("Hebbian weights matrix is not symmetric.")
   
   return W


def update(state, weights):
  """
  Apply the update rule to a state pattern.

  Parameters
  -----------
  state (numpy array): The current state (= the current binary pattern)
  weights (numpy array): The Hebbian weights matrix
  
  Returns
  -----------
  state (numpy array): The updated binary pattern

  The function computes the dot product of the current state and the Hebbian weights.
  The elements with the resulting dot product less than 0 become -1 and the others 1.
  """
  # Error checks : 

  # Ensure that the dimensions of the state and weights matrix are compatible for the dot product:
  if not all(len(pattern) == len(state) for pattern in weights):
      raise ValueError("Incompatible dimensions between state and weights.")
  
  # Ensure that the state is binary:
  if not np.all(np.logical_or(state == 1, state == -1)):
    raise ValueError("State must contain only binary values.")

  # Update rule : 
  dot_product = np.dot(weights, state)
  updated_pattern = np.ones_like(dot_product) 
  updated_pattern[dot_product < 0] = -1  
  return updated_pattern


def update_async(state,weights): 
   """
   Apply the asynchronous update rule to a state pattern.

   Parameters
   -----------
   state (numpy array): The current state (= the current binary pattern)
   weights (numpy array): The Hebbian weights matrix

   Returns
   -----------
   state (numpy array): The asynchronously updated binary pattern

   The function randomly select an index 'i' from the state, computes the dot product of the state and weights, and updates the selected element.
   If the dot product is greater or equal to 0, the selected element becomes 1. Otherwise, it becomes 0.
   """
   # Error checks: 
   # Ensure that the dimensions of the state and weights matrix are compatible for the dot product:
   if not all(len(pattern) == len(state) for pattern in weights):
      raise ValueError("Incompatible dimensions between state and weights.")
   
   # Ensure that the state is binary:
   if not np.all(np.logical_or(state == 1, state == -1)):
      raise ValueError("State must contain only binary values.")

   # Asynchronous update rule :
   index = np.random.randint(0, len(state))
   dot_product = np.dot(weights[index], state)

   if dot_product >= 0 : 
    updated_value = 1
   else : 
    updated_value = -1
   
   state[index] = updated_value
   return state


def dynamics(state, weights, max_iter):
   """
   Run the dynamical system from an initial state until convergence or until a maximum number of steps is reached. 
   Convergence is achieved when two consecutive updates return the same state.
    
   Parameters
   ----------
   state : numpy array
   the initial state of the perturbed pattern 

   weights : numpy array
   the weights matrix 

   max_iter : int
   the maximum number of iterations to run the dynamical system

   Returns
   -------
   A tuple containing the following elements:

      state_history : list of numpy array
                     the history of states computed during the dynamical system evolution

        convergence : bool
                     indicating whether the system has converged (True) or not (False)

        energy_history : list of float
                        the energy history corresponding to each state in the state_history
   """
   # Error checks
   if not all(len(pattern) == len(state) for pattern in weights):
      raise ValueError("Incompatible dimensions between state and weights.")
    
   if not np.all(np.logical_or(state == 1, state == -1)):
      raise ValueError("State must contain only binary values.")

   if not isinstance(max_iter, int) or max_iter <= 0: # Ensure that max_iter is a positive integer
      raise ValueError("max_iter must be a positive integer.")
   
   # Implementation of the synchronous dynamical evolution 
   state_history = [state.copy()]  # Initialize the state history with the initial given state
   energy_history = [energy(state,weights)]
   
   for _ in range(max_iter):
      updated_state = update(state, weights.copy())  # Make a copy of weights to avoid modifying the original
      state_history.append(updated_state)
      energy_history.append(energy(updated_state,weights))

      # if two consecutive updates return the same state convergence is achieved 
      if np.array_equal(state_history[-1], state_history[-2]):
         return state_history, True, energy_history

      state = updated_state  # Update of the state for the next iteration

   return state_history, False, energy_history


def dynamics_async(state, weights, max_iter, convergence_num_iter):
   """
   Run the dynamical system asynchronously from an initial state until a maximum number of steps is reached.
   For asynchronous updates, the convergence criterion is determined by the number of consecutive steps with no change. 
   If the solution does not change for "convergence_num_iter" steps in a row, then convergence is reached.

   Parameters
   ----------
   state : numpy array
      the initial state of the perturbed pattern 

   weights : numpy array
      the weights matrix

   max_iter : int
      the maximum number of iterations to run the dynamical system

   convergence_num_iter : int
      the number of consecutive iterations with no change needed to say the algorithm has reached convergence

   Returns
   -------
   A tuple containing the following elements:

      state_history : list of numpy ndarray
         the history of states computed during the dynamical system evolution

      convergence : bool
         indicating whether the system has converged (True) or not (False)

      energy_history : list of float
         the energy history corresponding to each state in the state_history
   """
   # Error checks
   if not all(len(pattern) == len(state) for pattern in weights):
      raise ValueError("Incompatible dimensions between state and weights.")
    
   if not np.all(np.logical_or(state == 1, state == -1)):
      raise ValueError("State must contain only binary values.")

   if not isinstance(max_iter, int) or max_iter <= 0: 
      raise ValueError("max_iter must be a positive integer.")
   
   # Check that convergence_num_iter is a positive integer:
   if not isinstance(convergence_num_iter, int) or convergence_num_iter <= 0:
      raise ValueError("convergence_num_iter must be a positive integer.")
   
   # Check that convergence_num_iter is less than or equal to max_iter:
   if convergence_num_iter > max_iter:
      raise ValueError("convergence_num_iter must be less than or equal to max_iter.")

   
   # Implementation of the asynchronous dynamical evolution 
   state_history = state.copy()
   energy_history = energy(state, weights)
   unchanged_steps = 0 
   
   
   for _ in range(max_iter): 
      
      state = update_async(state, weights)
      state_history.append(state.copy())
      if np.array_equal(state_history[-1], state_history[-2]): 
          unchanged_steps += 1
          if unchanged_steps >= convergence_num_iter: 
              return state_history, True, energy_history  
      else: 
          unchanged_steps = 0 
          

   return state_history, False, energy_history


def compare_patterns(pattern1, pattern2):
    # check if the patterns have the same size before comparison
    if len(pattern1) != len(pattern2):
        print ("the patterns are not of the same size")

#Compare the values of the patterns and return the number of perturbations
    perturbations = 0
    for i in range(len(pattern1)):
        if pattern1[i] != pattern2[i]:
            perturbations += 1

    return perturbations


def storkey_weights(patterns):
   size_pattern = len(patterns[0])
   SW = np.zeros((size_pattern, size_pattern))

   for pattern in patterns:
      H = pattern.reshape((1, -1)) * SW
      sum = np.sum(H, axis=1, keepdims=True)
      diagonal = np.repeat(np.diagonal(H).reshape((-1, 1)), size_pattern, 1)
      np.fill_diagonal(diagonal, 0)
      hi_j = (sum - (H + diagonal))
      Hfinal = (pattern * hi_j).T
      SW += (np.outer(pattern, pattern) - Hfinal - Hfinal.T) / size_pattern

   return SW

"""
def storkey_weights(patterns):
   nb_patterns, pattern_size = len(patterns), len(patterns[0])

   # Initialize the weight matrix
   W = np.zeros((pattern_size, pattern_size))

   for u in range(nb_patterns):
      # Initialize the previous_weights for the first iteration
      if u == 0 : 
        previous_weights = np.zeros((pattern_size, pattern_size)) 
      else : 
        previous_weights = np.copy(W)

      # Calculate the contribution to the weights matrix for each pattern
      for i in range(pattern_size):
         for j in range(pattern_size):
            if i != j:
               # Calculate h_ij for the current i,j and the u-th pattern
               for k in range(pattern_size): 
                  if (k != i & k != j): 
                     h_ij = np.sum(previous_weights[i, k] * patterns[u][k])

               # Update the weights matrix
               W[i, j] += (1/pattern_size) * (patterns[u][i] * patterns[u][j] - patterns[u][i]*h_ij - patterns[u][j]*h_ij)

   return W
"""


def energy(state, weights): 
   """
   Calculate the energy of a given state in a Hopfield network.

   Parameters:
   -----------
   state : numpy.ndarray
         Binary vector representing the current state of the network.
   weights : numpy.ndarray 
            Weight matrix representing the synaptic connections in the Hopfield network.

   Returns:
   --------
   float: Energy value associated with the given state and network weights.

   Example:
   --------
   >>> energy(np.array([1, -1, 1, -1]), np.array([[0, 1, -1, 0], [1, 0, 0, -1], [-1, 0, 0, 1], [0, -1, 1, 0]]))
   4.0

   """
   # Error checks
   # Check that state is a np array with a 1D shape: 
   if not isinstance(state, np.ndarray) or state.ndim != 1:
      raise ValueError("state must be a 1D numpy array.")

   # Check that weights is a np array with a 2D square shape:
   if not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
      raise ValueError("weights must be a square 2D numpy array.")
   
   # Energy function implementation 
   N,M = np.shape(weights)
   energy_value = 0
   for i in range(N):
      for j in range(M): 
         energy_value += weights[i,j]*state[i]*state[j]

   return energy_value*(-0.5)


def generate_checkerboard(size_board, size_checker):
   """
   Generate a checkerboard matrix with alternating patterns of 1 and -1.

   Parameters:
   -------------
   size_board : int
      Size of the square checkerboard matrix.
   size_checker : int
      Size of individual checkers in the checkerboard.

   Returns:
   -------------
   checkerboard : numpy.ndarray
      Checkerboard matrix with alternating 1s and -1s.

   Example:
   ------------ 
   >>> checkerboard = generate_checkerboard(50, 5)
   >>> checkerboard
   array([[ 1.,  1.,  1., ..., -1., -1., -1.],
          [ 1.,  1.,  1., ..., -1., -1., -1.],
          [ 1.,  1.,  1., ..., -1., -1., -1.],
          ...,
          [-1., -1., -1., ...,  1.,  1.,  1.],
          [-1., -1., -1., ...,  1.,  1.,  1.],
          [-1., -1., -1., ...,  1.,  1.,  1.]])

   This example generates a 50x50 checkerboard with 5x5 alternating checkers of 1 and -1.

   """
   # Error checks
   # Verify that size_board is an integer greater than zero:
   if not isinstance(size_board, int) or size_board <= 0:
      raise ValueError("size_board must be an integer greater than zero.")
   
   # Verify that size_checker is an integer greater than zero:
   if not isinstance(size_checker, int) or size_checker <= 0:
      raise ValueError("size_checker must be an integer greater than zero.")

   # Create a 50x50 matrix with alternating 5x5 checkers of 1 and -1
   checkerboard = np.zeros((size_board, size_board))

   for i in range(0, size_board, size_checker):
      for j in range(0, size_board, size_checker):
         checkerboard[i:i+size_checker, j:j+size_checker] = 1 if (((i + j)//size_checker) %2 == 0) else -1

   return checkerboard   


def save_video(state_list, out_path):
   Quality=5 #Increase for better video quality but higher running time ( must be higher or = 1)
   # Create a figure:
   fig, axes = plt.subplots()
   frames = []
   for state in state_list:
        frame = axes.imshow(state, animated=True, cmap='gray', vmin=-1, vmax=1)
        frames.append([frame])
   anim = ArtistAnimation(fig, frames, interval=300, blit=True) #increase interval to slow down frame rate
   # Save the animation as a video file using the writer 'ffmpeg'
   anim.save(out_path, writer='ffmpeg', fps=3*Quality, metadata=dict(artist='Me'), bitrate=180*Quality)
   plt.show()
   return out_path




