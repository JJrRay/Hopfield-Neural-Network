import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.animation as animation



def generate_checkerboard(size_board, size_checker):


   # Create a 50x50 matrix with alternating 5x5 checkers of 1 and -1
   checkerboard = np.zeros((size_board, size_board))

   for i in range(0, size_board, size_checker):
      for j in range(0, size_board, size_checker):
         checkerboard[i:i+size_checker, j:j+size_checker] = 1 if (((i + j)//size_checker) %2 == 0) else -1

   return checkerboard   

def generate_checkerboard2(size_board, size_checker):


   # Create a 50x50 matrix with alternating 5x5 checkers of 1 and -1
   checkerboard = np.zeros((size_board, size_board))

   for i in range(0, size_board, size_checker):
      for j in range(0, size_board, size_checker):
         checkerboard[i:i+size_checker, j:j+size_checker] = -1 if (((i + j)//size_checker) %2 == 0) else 1

   return checkerboard 


checker2= generate_checkerboard2(50,5)  
chercker3= -1*np.zeros((50,50))
checker = generate_checkerboard(50,5)

CheckerList= [checker,chercker3]




def save_video(state_list, out_path):
   Quality=5 #Increase for better video quality but higher running time ( must be higher or = 1)
   # Create a figure:
   fig, axes = plt.subplots()
   frames = []
   for state in state_list:
        frame = axes.imshow(state, animated=True, cmap='gray', vmin=-1, vmax=1)
        frames.append([frame])
   anim = ArtistAnimation(fig, frames, interval=300, blit=True)
   # Save the animation as a video file using the writer 'ffmpeg'
   anim.save(out_path, writer='ffmpeg', fps=3*Quality, metadata=dict(artist='Me'), bitrate=180*Quality)
   plt.show()
   return out_path
save_video(CheckerList, "output.mp4")