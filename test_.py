import numpy as np 
import doctest
import functions as f 
import pytest
from pathlib import Path

"""-------------------Doctests-----------------------"""
def test_functions():
    """ integrating the doctests in the pytest framework """
    assert doctest.testmod(f, raise_on_error=True)

"""-----------------Tests for the exceptions-------------------"""
# tests raise error for generate_patterns
def test_generate_patterns_error():
    with pytest.raises(ValueError, match="num_patterns and pattern_size must be positive integers."):
        f.generate_patterns(0, 4)

    with pytest.raises(ValueError, match="num_patterns and pattern_size must be positive integers."):
        f.generate_patterns(3, 0)

    with pytest.raises(ValueError, match="num_patterns and pattern_size must be positive integers."):
        f.generate_patterns(-2, -3)

# tests raise error for perturb pattern 
def test_perturb_pattern_error_num_perturb():
    with pytest.raises(ValueError, match="Number of perturbations must be a positive integer."):
        f.perturb_pattern(np.array([1, -1, 1]), -2)

    with pytest.raises(ValueError, match="Number of perturbations must be a positive integer."):
        f.perturb_pattern(np.array([1, -1, 1]), 0)

    with pytest.raises(ValueError, match="Number of perturbations must be a positive integer."):
        f.perturb_pattern(np.array([1, -1, 1]), 1.5)

def test_perturb_pattern_error_num_perturb_greater_than_pattern_length():
    with pytest.raises(ValueError, match="Number of perturbations cannot be greater than the length of the pattern."):
        f.perturb_pattern(np.array([1, -1, 1]), 5)

    with pytest.raises(ValueError, match="Number of perturbations cannot be greater than the length of the pattern."):
        f.perturb_pattern(np.array([]), 1)

# test raise error for pattern_match 
def test_pattern_match_error_non_numpy_array():
    with pytest.raises(TypeError, match="Pattern must be a numpy array."):
        f.pattern_match([], [1, -1, 1])

def test_pattern_match_error_empty_pattern():
    with pytest.raises(ValueError, match="Pattern can't be empty."):
        f.pattern_match([], np.array([]))

def test_pattern_match_error_empty_memorized_patterns():
    with pytest.raises(ValueError, match="List of memorized_patterns can't be empty"):
        f.pattern_match([], np.array([1, -1, 1]))

def test_pattern_match_error_length_mismatch():
    with pytest.raises(ValueError, match="Length of the memorized patterns must match the length of the pattern."):
        f.pattern_match([np.array([1, -1, 1]), np.array([1, 1, -1, 1])], np.array([1, -1, 1, 1]))

# test raise error for hebbian_weights 
def test_hebbian_weights_error_for_empty_patterns():
    with pytest.raises(ValueError, match="Patterns can't be empty."):
        f.hebbian_weights(np.array([]))

def test_hebbian_weights_error_for_non_numpy_array():
    with pytest.raises(TypeError, match="Patterns should be a 2D numpy array."):
        f.hebbian_weights([1, -1, 1])

def test_hebbian_weights_error_for_non_binary_patterns():
    with pytest.raises(ValueError, match="Patterns must be binary."):
        f.hebbian_weights(np.array([[1, 2, -1], [1, -1, 1]]))

def test_hebbian_weights_error_for_duplicate_patterns():
    with pytest.raises(ValueError, match="Generated patterns are not unique."):
        f.hebbian_weights(np.array([[1, -1, 1], [1, -1, 1]]))

# tests raise error for update function 
def test_update_error_for_incompatible_dimensions():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1, 0], [1, 0, -1, 1]])
    with pytest.raises(ValueError, match="Incompatible dimensions between state and weights."):
        f.update(state, weights)

def test_update_error_for_non_binary_state():
    state = np.array([1, 2, -1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    with pytest.raises(ValueError, match="State must contain only binary values."):
        f.update(state, weights)

# tests raise error for async update function 
def test_async_update_error_for_incompatible_dimensions():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="Incompatible dimensions between state and weights."):
        f.update_async(state, weights)

def test_async_update_error_for_non_binary_state():
    state = np.array([1, 1, -5])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    with pytest.raises(ValueError, match="State must contain only binary values."):
        f.update(state, weights)

# tests raise error for dynamics function 
def test_dynamics_error_for_incompatible_dimensions():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1, 0, 0], [1, 0, -1, 0, 0]])
    max_iter = 10
    with pytest.raises(ValueError, match="Incompatible dimensions between state and weights."):
        f.dynamics(state, weights, max_iter)

def test_dynamics_error_for_non_binary_state():
    state = np.array([1, 2, -1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    max_iter = 10
    with pytest.raises(ValueError, match="State must contain only binary values."):
        f.dynamics(state, weights, max_iter)

def test_dynamics_error_for_non_positive_integer_max_iter():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    # case 1 : max_iter is negative
    max_iter_1 = -5
    with pytest.raises(ValueError, match="max_iter must be a positive integer."):
        f.dynamics(state, weights, max_iter_1)

    # case 2 : max iter is not >0 
    max_iter_2 = 0 
    with pytest.raises(ValueError, match="max_iter must be a positive integer."):
        f.dynamics(state, weights, max_iter_2)
    
    # case 3 : max_iter is not an integer
    max_iter_3 = 1.5
    with pytest.raises(ValueError, match="max_iter must be a positive integer."):
        f.dynamics(state, weights, max_iter_3)

# tests raise error for dynamics_async 
def test_dynamics_async_error_for_incompatible_dimensions():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1, 0, 0], [1, 0, -1, 0, 0]])
    max_iter = 10
    convergence_num_iter = 2
    with pytest.raises(ValueError, match="Incompatible dimensions between state and weights."):
        f.dynamics_async(state, weights, max_iter, convergence_num_iter)

def test_dynamics_async_error_for_non_binary_state():
    state = np.array([1, 2, -1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    max_iter = 10
    convergence_num_iter = 2
    with pytest.raises(ValueError, match="State must contain only binary values."):
        f.dynamics_async(state, weights, max_iter, convergence_num_iter)

def test_dynamics_error_for_non_positive_integer_max_iter():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    convergence_num_iter = 2
    # case 1 : max_iter is negative
    max_iter_1 = -10
    with pytest.raises(ValueError, match="max_iter must be a positive integer."):
        f.dynamics_async(state, weights, max_iter_1, convergence_num_iter)

    # case 2 : max iter is not >0 
    max_iter_2 = 0 
    with pytest.raises(ValueError, match="max_iter must be a positive integer."):
        f.dynamics_async(state, weights, max_iter_2, convergence_num_iter)

    # case 3 : max_iter is not an integer
    max_iter_3 = 1.5
    with pytest.raises(ValueError, match="max_iter must be a positive integer."):
        f.dynamics_async(state, weights, max_iter_3, convergence_num_iter)

def test_dynamics_async_error_for_non_positive_integer_convergence_num_iter():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    max_iter = 10

    # case 1 : negative 
    convergence_num_iter_1 = -2
    with pytest.raises(ValueError, match="convergence_num_iter must be a positive integer."):
        f.dynamics_async(state, weights, max_iter, convergence_num_iter_1)

    # case 2 : zero 
    convergence_num_iter_2 = 0
    with pytest.raises(ValueError, match="convergence_num_iter must be a positive integer."):
        f.dynamics_async(state, weights, max_iter, convergence_num_iter_2)

    # case 3: not an integer 
    convergence_num_iter_3 = 1.5
    with pytest.raises(ValueError, match="convergence_num_iter must be a positive integer."):
        f.dynamics_async(state, weights, max_iter, convergence_num_iter_3)

def test_dynamics_async_error_for_convergence_num_iter_greater_than_max_iter():
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, -1]])
    max_iter = 5
    convergence_num_iter = 7
    with pytest.raises(ValueError, match="convergence_num_iter must be less than or equal to max_iter."):
        f.dynamics_async(state, weights, max_iter, convergence_num_iter)

# tests raise error for storkey_weights 
def test_storkey_weights_error_for_non_2D_numpy_array():
    # case 1 : not a numpy array 
    patterns_1 = [[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]
    with pytest.raises(ValueError, match="patterns must be a 2D numpy array."):
        f.storkey_weights(patterns_1)

    # case 2 : not a 2D array 
    patterns_2 = [1, 1, -1, -1]
    with pytest.raises(ValueError, match="patterns must be a 2D numpy array."):
        f.storkey_weights(patterns_2)

def test_storkey_weights_for_non_binary_patterns():
    # example 1
    patterns_1 = np.array([[1, 1, -1, -2], [1, 1, -1, 1], [-1, 1, -1, 1]])
    with pytest.raises(ValueError, match="Patterns must contain only binary values."):
        f.storkey_weights(patterns_1)
    
    # example 2
    patterns_2 = np.array([[1, 1, -1, 1], [1, 0, -1, 1], [-1, 1, -1, 1]])
    with pytest.raises(ValueError, match="Patterns must contain only binary values."):
        f.storkey_weights(patterns_2)
    
    # example 3
    patterns_3 = np.array([[1, 1, -1, 1], [1, -1, -1, 1], [-1, 1, -1, 7]])
    with pytest.raises(ValueError, match="Patterns must contain only binary values."):
        f.storkey_weights(patterns_3)

# tests raise error for the energy function 
def test_energy_error_for_non_1D_numpy_array_state():
    # Case 1 : not a numpy array 
    state = [1, -1, 1, -1]
    weights = np.array([[0, 1, -1, 0], [1, 0, 0, -1], [-1, 0, 0, 1], [0, -1, 1, 0]])
    with pytest.raises(ValueError, match="state must be a 1D numpy array."):
        f.energy(state, weights)

    # Case 2 : not a 1D array 
    state = [[1, -1, 1, -1], [1, 1, 1, 1]]
    weights = np.array([[0, 1, -1, 0], [1, 0, 0, -1], [-1, 0, 0, 1], [0, -1, 1, 0]])
    with pytest.raises(ValueError, match="state must be a 1D numpy array."):
        f.energy(state, weights)

def test_energy_error_for_non_2D_numpy_array_weights():
    state = np.array([1, -1, 1, -1])
    # case 1 : not a numpy array 
    weights_1 = [[0, 1, -1, 0], [1, 0, 0, -1], [-1, 0, 0, 1], [0, -1, 1, 0]]
    with pytest.raises(ValueError, match="weights must be a square 2D numpy array."):
        f.energy(state, weights_1)
    
    # case 2 : not a 2D array 
    weights_2 = [0, 1, -1, 0]
    with pytest.raises(ValueError, match="weights must be a square 2D numpy array."):
        f.energy(state, weights_2)

def test_energy_error_for_non_square_weights():
    state = np.array([1, -1, 1, -1])
    weights = np.array([[0, 1, -1, 0], [1, 0, 0, -1], [-1, 0, 0, 1]])
    with pytest.raises(ValueError, match="weights must be a square 2D numpy array."):
        f.energy(state, weights)

# tests raise error for generate checkerboard 
def test_generate_checkerboard_error_for_size_board():
    # case 1 : size_board is not an integer : 
    size_board_1 = 50.5
    size_checker = 5
    with pytest.raises(ValueError, match="size_board must be an integer greater than zero."):
        f.generate_checkerboard(size_board_1, size_checker)

    # case 2 : size board is zero (should be >0)
    size_board_2 = 0
    size_checker = 5
    with pytest.raises(ValueError, match="size_board must be an integer greater than zero."):
        f.generate_checkerboard(size_board_2, size_checker)
    
    # case 3 : size board is negative
    size_board_3 = -50
    size_checker = 5
    with pytest.raises(ValueError, match="size_board must be an integer greater than zero."):
        f.generate_checkerboard(size_board_3, size_checker)

def test_generate_checkerboard_error_for_size_checker():
    # case 1 : size_checker is not an integer
    size_board = 50
    size_checker_1 = 5.5
    with pytest.raises(ValueError, match="size_checker must be an integer greater than zero."):
        f.generate_checkerboard(size_board, size_checker_1)
    
    # case 2 : size_checker is zero (should be >0)
    size_board = 50
    size_checker_2 = 0
    with pytest.raises(ValueError, match="size_checker must be an integer greater than zero."):
        f.generate_checkerboard(size_board, size_checker_2)
    
    # case 3 : size_checker is negative 
    size_board = 50
    size_checker_3 = -5
    with pytest.raises(ValueError, match="size_checker must be an integer greater than zero."):
        f.generate_checkerboard(size_board, size_checker_3)

# tests raise error for save_video 
def test_save_video_error_for_state_list():
    state_list = np.array([[1, -1], [-1, 1]])
    out_path = 'output_video.mp4'
    with pytest.raises(TypeError, match="state_list must be a list."):
        f.save_video(state_list, out_path)

def test_save_video_error_for_out_path():
    state_list = [np.array([[1, -1], [-1, 1]])]
    out_path = 7
    with pytest.raises(TypeError, match="out_path must be a string."):
        f.save_video(state_list, out_path)


"-------------Tests for the functions------------"
# Tests for the Hebbian weights matrix 
def test_HWM_shape():
    """
    Test that the weights matrix with the Hebbian weights has the correct size,
    with 2 cases.
    """
    # Case 1 :  
    generated_patterns_1 = f.generate_patterns(num_patterns=10,pattern_size=50)
    assert f.hebbian_weights(generated_patterns_1).shape[1] == generated_patterns_1.shape[1]

    # Case 2 :
    generated_patterns_2 = f.generate_patterns(num_patterns=10,pattern_size=10)
    assert f.hebbian_weights(generated_patterns_2).shape[1] == generated_patterns_2.shape[1]

def test_HWM_diag():
    """
    Test that the elements on the diagonal of the Hebbian weights matrix are 0, 
    with 2 examples of different sizes. 
    """
    # example 1 : 
    patterns_1 = f.generate_patterns(num_patterns=3, pattern_size=50)
    assert np.all(np.diag(f.hebbian_weights(patterns_1))==0)

    # example 2 : 
    patterns_2 = f.generate_patterns(num_patterns=80, pattern_size=1000)
    assert np.all(np.diag(f.hebbian_weights(patterns_2))==0)

def test_HWM_sym(): 
    """
    Test that the Hebbian weight matrix is symmetric, 
    with 2 examples of different sizes.
    """
    # example 1 : 
    patterns_1 = f.generate_patterns(num_patterns=3, pattern_size=50)
    transposee_1 = np.array(np.transpose(f.hebbian_weights(patterns_1)))
    assert np.array_equal(f.hebbian_weights(patterns_1), transposee_1)

    # example 2 : 
    patterns_2 = f.generate_patterns(num_patterns=80, pattern_size=1000)
    transposee_2 = np.array(np.transpose(f.hebbian_weights(patterns_2)))
    assert np.array_equal(f.hebbian_weights(patterns_2), transposee_2)

def test_HWM_values(): 
    """
    Test that the values of the hebbian weights matrix are in the correct range, 
    with 2 examples of different sizes. 
    """ 
    # example 1 : 
    patterns_1 = f.generate_patterns(num_patterns=3, pattern_size=50)
    assert np.all((f.hebbian_weights(patterns_1)>=-1) & (f.hebbian_weights(patterns_1)<=1))

    # example 2 : 
    patterns_2 = f.generate_patterns(num_patterns=80, pattern_size=1000)
    assert np.all((f.hebbian_weights(patterns_2)>=-1) & (f.hebbian_weights(patterns_2)<=1))

def test_weights_hebbian() : 
    """
    Test if the function hebbian_weights return the correct 
    weights matrix associated with the given patterns.
    """
    patterns = np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]])
    HW_matrix = np.array([[0., 0.33333333, -0.33333333, -0.33333333],[0.33333333, 0., -1., 0.33333333],[-0.33333333, -1., 0., -0.33333333],[-0.33333333, 0.33333333, -0.33333333, 0.]])
 
    assert (np.allclose(f.hebbian_weights(patterns),HW_matrix))

# Test for the Storkey weight matrix :
def test_weights_storkey():
    """
    Test if the function storkey_weights return the correct 
    weights matrix associated with the given patterns.
    """
    patterns = np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]])
    SW_matrix = np.array([[1.125, 0.25, -0.25, -0.5],[0.25, 0.625, -1., 0.25],[-0.25, -1., 0.625, -0.25],[-0.5, 0.25, -0.25, 1.125]])
    
    # Calculate the result:
    result = np.array(f.storkey_weights(patterns))
    print (result)

    # Check if all elements in result are close to the corresponding elements in SW_matrix 
    assert np.allclose(result, SW_matrix) 

# Test for the energy function : 
def is_not_increasing(list):
    """ 
    Check if a list values is increasing. 
    Returns True if it's not increasing and False if it is increasing.
    This function is used in the evolution tests to determine if the energy function is increasing. 
    """ 
    for i in range(len(list) - 1):
        if list[i] < list[i + 1]: # it's increasing 
            return False
    return True # it's not increasing 

# Test for perturb_pattern : 
def test_perturb_pattern():
    """
    Test the perturb_pattern function by checking if the number of perturbations made by the function  
    corresponds to the expected/given number of perturbations. 
    """
    nb_perturb = 3 
    perturbations = 0 
    original_pattern = np.array([1, 1, -1, 1])
    perturbed_pattern = f.perturb_pattern(pattern=original_pattern, num_perturb=nb_perturb)

    for i in range(len(original_pattern)):
        if perturbed_pattern[i] != original_pattern[i] :
            perturbations += 1
    
    assert (nb_perturb==perturbations)

# Test for generate_patterns :
def test_generate_patterns():
    """
    Test the generate_patterns function by checking if the shape of the generated patterns is correct 
    and if all the elements in the generated patterns are -1 or 1. 
    """
   
   # Test with a small given number of patterns and pattern size
    nb_patterns = 10
    pattern_size = 20
    patterns = f.generate_patterns(nb_patterns, pattern_size)
    
   # Check if the shape of the generated patterns is correct
    assert patterns.shape == (nb_patterns, pattern_size)

   # Check if all elements in the generated patterns are -1 or 1
    assert np.all(np.logical_or(patterns == -1, patterns == 1))

# Test for pattern_match : 
def test_pattern_match():
    """
    Test the pattern_match function with 2 different cases : 
    1) using a matching pattern => it should return the corresponding index of the correct matching pattern
    2) using a non_matching pattern => it should return None since the pattern can't match with the memorized 
    patterns (it's not in the list)
    """

    # Case 1 : Test with a matching pattern
    memorized_patterns = np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]])
    pattern_to_match = np.array([1, 1, -1, 1])
    result = f.pattern_match(memorized_patterns, pattern_to_match)
    assert result == 1 # Index of the matching pattern

    # Case 2: Test with a non-matching pattern
    non_matching_pattern = np.array([1, 1, 1, 1])
    result = f.pattern_match(memorized_patterns, non_matching_pattern)
    assert result == None  # no matching pattern => we expect None

# Test for the synchronous update rule : 
def test_update():
    """
    Test the update function with 5 different cases.
    """
    # 1st case: 
    state = np.array([1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, -1], [-1, -1, 0]])
    updated_pattern = f.update(state, weights) 
    expected_result = np.array([-1, 1, 1])  # calculated manually based on the update rule
    assert np.all(updated_pattern == expected_result)

    # 2nd case:
    state = np.array([-1, -1, 1])
    weights = np.array([[0, 1, -1], [1, 0, -1], [-1, -1, 0]])
    updated_pattern = f.update(state, weights)
    expected_result = np.array([-1, -1, 1])  # Calculated manually based on the update rule
    assert np.all(updated_pattern == expected_result)

    # 3rd case: 
    state = np.array([1, -1, 1])
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    updated_pattern = f.update(state, weights)
    expected_result = np.array([1, 1, 1])  # all of the dot products are positive => the result should be all 1
    assert np.all(updated_pattern == expected_result)

    # 4th case: 
    state = np.array([1, -1, 1])
    weights = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
    updated_pattern = f.update(state, weights)
    expected_result = np.array([-1, -1, -1])  # all of the dot products are negative => the result should be all -1
    assert np.all(updated_pattern == expected_result)

    # 5th case: 
    state = np.array([-1, -1, 1])
    weights = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    updated_pattern = f.update(state, weights)
    expected_result = np.array([1, 1, 1])  # all of the dot products are 0 => the result should be all 1
    assert np.all(updated_pattern == expected_result)

# Test for the asynchronous update rule : 
def test_update_async() : 
    """
    Test the update_async function by checking that there is exactly one difference 
    between the original state and the updated state (knowing that there will necesserly be 
    a difference with the chosen examples of weights and state).
    """
    state = np.array([1, 1, 1])
    weights = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]) # avec cet exemple il y aura forcément une différence
    differences = 0 
    updated_state = f.update_async(state.copy(), weights) # without the copy it would modify the original state
    
    for i in range(len(updated_state)):
        if state[i]!=updated_state[i]:
            differences += 1
    
    assert differences == 1

Test for generate_checkerboard : 
def test_generate_checkerboard():
    """
    Test the generate_checkerboard function
    """
    size_board = 50
    size_checker = 5
    checkerboard = f.generate_checkerboard(size_board, size_checker)

    # Check if the shape of the generated checkerboard is correct : 
    assert checkerboard.shape == (size_board, size_board)

    # Check if the values in the checkerboard are correct (alternating 5x5 checkers of 1 and -1): 
    for i in range(0, size_board, size_checker):
        for j in range(0, size_board, size_checker):
            if (((i + j) // size_checker) % 2 == 0):
                expected_value = 1 
            else: 
                expected_value = -1
            assert np.all(checkerboard[i:i+size_checker, j:j+size_checker] == expected_value)

Test for save_video : 
def test_save_video():
    patterns = f.generate_patterns(50, 2500)

    index = np.random.choice(patterns.shape[0])
    pattern_to_perturb = patterns[index].copy()
    perturbed_pattern = f.perturb_pattern(pattern=pattern_to_perturb, num_perturb=1000)
    hebbian_weights = f.hebbian_weights(patterns).copy()


    state_history_SHC, _, _ = f.dynamics(state=perturbed_pattern, weights=hebbian_weights, max_iter=20)

    reshaped_SHC = [state.reshape(50, 50) for state in state_history_SHC]
    video1 = f.save_video(state_list=reshaped_SHC, out_path="output1.mp4")
    assert Path(video1).exists()

    state_history_AHC, _, _ = f.dynamics_async(state=perturbed_pattern, weights=hebbian_weights, max_iter=30000, convergence_num_iter=10000)
    selected_states = state_history_AHC[::1000]
    reshaped_AHC = [(np.array(state)).reshape(50, 50) for state in selected_states]
    video2 = f.save_video(state_list=reshaped_AHC, out_path="output2.mp4")
    assert Path(video2).exists()


"-----------------Testing the evolution of a system-------------------"
# Evolution of a system with synchronous update and Hebbian weights 
def test_evolution_SH(): 
   """
   Test the evolution of the synchronous dynamical system with the Hebbian rule by checking that : 
    - it converges,  
    - it matches with the correct original pattern 
    - the energy function is non-increasing 
"""
   generated_patterns = f.generate_patterns (80,1000)
   pattern_to_perturb = generated_patterns[7].copy() 
   perturbed_pattern = f.perturb_pattern(pattern_to_perturb, num_perturb=200)
   weights = f.hebbian_weights(generated_patterns)
   
   state_history_SH, convergence_SH, energy_history_SH = f.dynamics(perturbed_pattern, weights, max_iter=20)
   
   print (state_history_SH) 
   
   assert (is_not_increasing(energy_history_SH)) # test if the energy function of the system is not increasing 
   assert convergence_SH # test if convergence is found (=True)
   assert (f.pattern_match(generated_patterns,state_history_SH[-1])==7) # test if the system can find the correct original pattern
   
# Evolution of a system with asynchronous update and Hebbian weights 
def test_evolution_AH(): 
    """
    Test the evolution of the asynchronous dynamical system with the Hebbian rule by checking that it converges, that it matches with the correct
    original pattern and that the energy function is non-increasing. 
"""
    generated_patterns = f.generate_patterns (80,1000)
    pattern_to_perturb = generated_patterns[17].copy() 
    perturbed_pattern = f.perturb_pattern(pattern_to_perturb, num_perturb=200)
    weights = f.hebbian_weights(generated_patterns)
   
    state_history_AH, convergence_AH, energy_history_AH = f.dynamics_async(state=perturbed_pattern, weights=weights, max_iter=20000, convergence_num_iter=3000)

    assert (is_not_increasing(energy_history_AH))
    assert convergence_AH
    assert (f.pattern_match(generated_patterns, state_history_AH[-1])==17)
    
# Evolution of a system with synchronous update and Storkey weights 
def test_evolution_SS(): 
    """
    Test the evolution of the synchronous dynamical system with the Storkey rule by checking that it converges, that it matches with the correct
    original pattern and that the energy function is non-increasing. 
"""
    generated_patterns = f.generate_patterns (80,1000)
    pattern_to_perturb = generated_patterns[7].copy() 
    perturbed_pattern = f.perturb_pattern(pattern_to_perturb, num_perturb=200)
    weights = f.storkey_weights(generated_patterns)
    state_history_SS, convergence_SS, energy_history_SS = f.dynamics(state=perturbed_pattern, weights=weights,max_iter=20)
    
    assert (is_not_increasing(energy_history_SS))
    assert convergence_SS
    assert (f.pattern_match(generated_patterns, state_history_SS[-1])==7)
    
# Evolution of a system with asynchronous update and Storkey weights
def test_evolution_AS():
    """
    Test the evolution of the asynchronous dynamical system with the Storkey rule by checking that it converges, that it matches with the correct
    original pattern and that the energy function is non-increasing. 
"""
    
    generated_patterns = f.generate_patterns (80,1000)
    pattern_to_perturb = generated_patterns[7].copy() 
    perturbed_pattern = f.perturb_pattern(pattern_to_perturb, num_perturb=200)
    weights = f.storkey_weights(generated_patterns)
   
    state_history_AS, convergence_AS, energy_history_AS = f.dynamics_async(state=perturbed_pattern, weights=weights, max_iter=20000, convergence_num_iter=3000)

    assert (is_not_increasing(energy_history_AS))
    assert convergence_AS
    assert (f.pattern_match(generated_patterns, state_history_AS[-1])==7)











