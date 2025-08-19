# Kohonen Self-Organizing Map for Handwritten Letter Recognition

This project implements a **Kohonen Self-Organizing Map (SOM)** in Python to cluster and recognize pre-processed handwritten English letters (A–Z).  
It was developed as part of a **machine learning assignment in university**.

---

## Project Overview
- The SOM is trained on vectors of **16 normalized features** representing handwritten letters.  
- Each input vector is mapped onto a **2D square grid** of neurons (`m x m`).  
- Weights between inputs and neurons are updated using the **Kohonen learning algorithm**.  
- The goal is to cluster similar letters into neighborhoods on the grid, so that each neuron (or cluster) corresponds to a specific letter.  

---

## Experiments
Several experiments were conducted to find the **optimal grid size** and **learning rate**:

### Grid Size
- Grid sizes from **20×20 to 110×110** were tested.  
- **Smaller grids (20×20)** converged slower and ended with higher error.  
- **Larger grids (110×110)** had the lowest final error but required more computation.  
- **Best trade-off**: **90×90**, which balanced error reduction, stability, and efficiency.  

### Learning Rate
- Learning rates from **0.1 to 0.9** were tested.  
- Most configurations showed similar convergence, but:  
  - **Higher rates** had higher error early on.  
  - **Learning rate = 0.7** provided the best balance at epoch 100.  

### Clustering
- Using **GridSize = 90×90** and **Learning Rate = 0.7**, the SOM produced meaningful clusters where each region corresponded to a distinct letter of the alphabet.  

---

## Instructions to run the Neural Network:
1) You need to have in the same folder as kohonen.py the train.txt file
2) Open terminal and navigate to the file referred in step 1
3) Execute the program with the command 'python kohonen.py'
Note: The program uses the numpy, collections and random libraries, in case there are errors about that, the libraries can be install with the command: 'pip install numpy'(the same goes for the other libraries)(run these commands in terminal before running the program)'
4) Results should be printed in files results.txt, clustering.txt
