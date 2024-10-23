import tkinter as tk
from collections import deque
from heapq import heappush, heappop
import itertools
import random
import time
import sys

class ArcConsistencySolver:
    # Create a list of domains (possible assignments) for each cell of the board
    def getDomains(self, board):
        domains = []
        for i in range(self.size):
            rowDoms = []
            for j in range(self.size):
                # Domain of unassigned cell is digits 1-9
                if board[i][j] == 0:
                    rowDoms.append([1,2,3,4,5,6,7,8,9])
                # Domain of pre-assigned cell is only the digit assigned to that cell 
                else:
                    rowDoms.append([board[i][j]])
            domains.append(rowDoms)
        return domains
    
    def __init__(self, board):
        self.board = board
        self.size = len(board)
        self.box_size = int(self.size ** 0.5)
        self.domains = self.getDomains(board)
        self.numBacktracks = 0

    def isValid(self, row, col, num):
        # Check if the number is already in the row
        if num in self.board[row]:
            return False

        # Check if the number is already in the column
        if num in [self.board[i][col] for i in range(self.size)]:
            return False

        # Check if the number is already in the box
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i, j in itertools.product(range(self.box_size), repeat=2):
            if self.board[start_row + i][start_col + j] == num:
                return False

        return True

    def getBoxCoord(self, row, col):
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        return [(start_row + i, start_col + j) for i, j in itertools.product(range(self.box_size), repeat=2)]

    def checkArc(self, point1, num1, point2, num2):
        if num1 != num2:
            return True
        elif point1[0] == point2[0] or point1[1] == point2[1] or point1 in self.getBoxCoord(point2[0], point2[1]):
            return False
        else:
            return True

    def enforceArcConsistency(self):
        queue = deque()

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    for k in range(self.size):
                        if k != j:
                            queue.append(((i, j), (i, k)))
                        if k != i:
                            queue.append(((i, j), (k, j)))
                    for r, c in self.getBoxCoord(i, j):
                        if (r, c) != (i, j):
                            queue.append(((i, j), (r, c)))

        while queue:
            (row1, col1), (row2, col2) = queue.popleft()

            domain_changed = False
            for num1 in self.domains[row1][col1]:
                remove = True
                for num2 in self.domains[row2][col2]:
                    if self.checkArc((row1,col1), num1, (row2,col2), num2):
                        remove = False
                if remove:
                    self.domains[row1][col1].remove(num1)
                    domain_changed = True

            if domain_changed:
                for i in range(self.size):
                    if i != row1:
                        queue.append(((i, col1), (row1, col1)))
                    if i != row2:
                        queue.append(((i, col2), (row2, col2)))
                for j in range(self.size):
                    if j != col1:
                        queue.append(((row1, j), (row1, col1)))
                    if j != col2:
                        queue.append(((row2, j), (row2, col2)))
                for r, c in self.getBoxCoord(row1, col1):
                    if (r, c) != (row1, col1):
                        queue.append(((r, c), (row1, col1)))
                for r, c in self.getBoxCoord(row2, col2):
                    if (r, c) != (row2, col2):
                        queue.append(((r, c), (row2, col2)))

    # Use Arc Consistency to solve the Sudoku board
    def solve(self, ui):
        self.enforceArcConsistency()
        return self.backtrack(ui)

    def backtrack(self, ui):
        empty_cell = self.findEmptyCell()
        if not empty_cell:
            print("Backtracks: " + str(self.numBacktracks))
            return True  # Puzzle solved

        row, col = empty_cell
        cellDoms = self.domains[row][col].copy()
        for num in cellDoms:
            if self.isValid(row, col, num):
                domCopy = self.copyDoms()
                self.board[row][col] = num
                self.domains[row][col] = [num]
                ui.updateUI(self)
                if self.solve(ui):
                    return True
                self.board[row][col] = 0  # Backtrack
                self.domains = domCopy
                self.numBacktracks += 1
                self.domains[row][col].remove(num)
                ui.updateUI(self)

        return False

    def copyDoms(self):
        domCopy = []
        for row in self.domains:
            rowCopy = []
            for x in row:
                rowCopy.append(x.copy())
            domCopy.append(rowCopy)
        return domCopy

    def findEmptyCell(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        return None

    def getBoard(self):
        return self.board
    
class BreakoutMethodSolver:
    def __init__(self, board):
        self.board = board
        self.orig_board = []
        for row in board:
            self.orig_board.append(row.copy())
        self.size = len(board)
        self.box_size = int(self.size ** 0.5)
        self.numBacktracks = 0
    
    def is_valid(self, row, col, num):
        # Check if the number is already in the row
        if num in self.board[row]:
            return False

        # Check if the number is already in the column
        if num in [self.board[i][col] for i in range(self.size)]:
            return False

        # Check if the number is already in the box
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i, j in itertools.product(range(self.box_size), repeat=2):
            if self.board[start_row + i][start_col + j] == num:
                return False

        return True

    def getBoxCoord(self, row, col):
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        return [(start_row + i, start_col + j) for i, j in itertools.product(range(self.box_size), repeat=2)]
    
    # Creates a dictionary of all constraints in the sudoku board and their associated weight
    def getConstraintList(self):
        constraints = dict()
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    if j != k:
                        constraints.update({((i,j),(i,k)): 1})
                        constraints.update({((j,i),(k,i)): 1})
        for a in range(0, self.size, 3):
            for b in range(0, self.size, 3):
                for r1,c1 in self.getBoxCoord(a,b):
                    for r2,c2 in self.getBoxCoord(a,b):
                        if (r1,c1) != (r2,c2):
                            constraints.update({((r1,c1),(r2,c2)): 1})
        return constraints
    
    # Returns a list of all cells that are invalid due to the assignment at the given cell
    def invalidCells(self, row, col):
        invCells = []
        for i in range(self.size):
            if i != col and self.board[row][col] == self.board[row][i]:
                invCells.append((row, i))
            if i != row and self.board[row][col] == self.board[i][col]:
                invCells.append((i, col))
        for r,c in self.getBoxCoord(row,col):
            if (r,c) != (row,col) and self.board[row][col] == self.board[r][c] and (r,c) not in invCells:
                invCells.append((r,c))
        return invCells
    
    # Calculates the score of the current assignment. Score calculated as sum of weights of all unsatisfied constraints
    def evaluate(self, constraints):
        total = 0
        for i in range(self.size):
            for j in range(self.size):
                invCells = self.invalidCells(i,j)
                for r,c in invCells:
                    if constraints.get(((i,j),(r,c))):
                        total = total + constraints.get(((i,j),(r,c)))
        return total

    # Uses the breakout method for hill-climbing search to solve the puzzle
    def solve(self, ui):
        # Starts with a random assignment to all cells not given by the board
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.board[i][j] = random.randint(1,9)
        ui.updateUI(self)
        constraints = self.getConstraintList()
        # Calculate the score for the starting assignment
        currVal = self.evaluate(constraints)
        # While there are unsatisfied constraints, continue making adjustments
        while currVal > 0:
            smallVal = sys.maxsize
            smallChange = None
            # Try adjusting one variable at a time, finding the change that results in the lowest score
            for i in range(self.size):
                for j in range(self.size):
                    if self.orig_board[i][j] == 0:
                        for z in range(1,10):
                            prev = self.board[i][j]
                            self.board[i][j] = z
                            nxt = self.evaluate(constraints)
                            if nxt < smallVal:
                                smallVal = nxt
                                smallChange = (i,j,z)
                            self.board[i][j] = prev
            # If a change with a lower score than the currect assignment's was found, make that change
            if smallVal < currVal and smallChange:
                self.board[smallChange[0]][smallChange[1]] = smallChange[2]
                currVal = self.evaluate(constraints)
            # Otherwise, increase the weight of all currently unsatisfied constraints by one
            else:
                total = 0
                for i in range(self.size):
                    for j in range(self.size):
                        invCells = self.invalidCells(i,j)
                        for r,c, in invCells:
                            weight = constraints.get(((i,j),(r,c))) + 1
                            total = total + weight
                            constraints.update({((i,j),(r,c)): weight})
                currVal = total
            ui.updateUI(self)
            
        return True

# Utilizes a Maintaining Path Consistency Algorithm to solve a partially filled Sudoku Board
class PathConsistencySolver:
    # Create a list of domains (possible assignments) for each cell of the board
    def getDomains(self, board):
        domains = []
        for i in range(self.size):
            rowDoms = []
            for j in range(self.size):
                # Domain of unassigned cell is digits 1-9
                if board[i][j] == 0:
                    rowDoms.append([1,2,3,4,5,6,7,8,9])
                # Domain of pre-assigned cell is only the digit assigned to that cell 
                else:
                    rowDoms.append([board[i][j]])
            domains.append(rowDoms)
        return domains

    def __init__(self, board):
        self.board = board
        self.size = len(board)
        self.box_size = int(self.size ** 0.5)
        self.domains = self.getDomains(board)
        self.numBacktracks = 0

    def isValid(self, row, col, num):
        # Check if the number is already in the row
        if num in self.board[row]:
            return False
         # Check if the number is already in the column
        if num in [self.board[i][col] for i in range(self.size)]:
            return False
        # Check if the number is already in the box
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i, j in itertools.product(range(self.box_size), repeat=2):
            if self.board[start_row + i][start_col + j] == num:
                return False

        return True

    # Checks the validity of an assignment to two cells in an arc
    def checkArc(self, point1, num1, point2, num2):
        if num1 != num2:
            return True
        elif point1[0] == point2[0] or point1[1] == point2[1] or point1 in self.getBoxCoord(point2[0], point2[1]):
            return False
        else:
            return True

    # Get the range of coordinates for the mini 3x3 grid containing the given cell
    def getBoxCoord(self, row, col):
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        return [(start_row + i, start_col + j) for i, j in itertools.product(range(self.box_size), repeat=2)]
    
    # Enforces Path Consistency on the domains of all cells
    def enforcePathConsistency(self):
        queue = deque()

        # Add all paths to the queue
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    for k in range(self.size):
                        if k != j:
                            for z in range(self.size):
                                if z != k and z != j:
                                    queue.append(((i, j), (i, k), (i, z)))
                                if z != i:
                                    queue.append(((i, j), (i, k), (z, k)))
                            for r, c in self.getBoxCoord(i, k):
                                if (r, c) != (i, k) and (r, c) != (i, j):
                                    queue.append(((i, j), (i, k), (r, c)))
                        if k != i:
                            for z in range(self.size):
                                if z != j:
                                    queue.append(((i, j), (k, j), (k, z)))
                                if z != k and z != i:
                                    queue.append(((i, j), (k, j), (z, j)))
                            for r, c in self.getBoxCoord(k, j):
                                if (r, c) != (k, j) and (r, c) != (i, j):
                                    queue.append(((i, j), (k, j), (r, c)))
                    for r, c in self.getBoxCoord(i, j):
                        if (r, c) != (i, j):
                            for z in range(self.size):
                                if z != c and (i, j) != (r, z):
                                    queue.append(((i, j), (r, c), (r, z)))
                                if z != r and (i, j) != (z, c):
                                    queue.append(((i, j), (r, c), (z, c)))
                            for r2, c2 in self.getBoxCoord(r, c):
                                if (r2, c2) != (i, j) and (r2, c2) != (r, c):
                                    queue.append(((i, j), (r, c), (r2, c2)))

        while queue:
            # Get a path {X1,X2,X3} from the queue
            (row1, col1), (row2, col2), (row3, col3) = queue.popleft()

            domainChanged = False
            for num1 in self.domains[row1][col1]:
                remove = True
                for num3 in self.domains[row3][col3]:
                    # For every valid assignment to {X1=num1, X3=num3}
                    if not self.checkArc((row1, col1), num1, (row3, col3), num3):
                        continue
                    # Ensure that there is at least one assignment X2=num2 that satisfies both {X1=num1, X2=num2} and {X2=num2, X3=num3}
                    for num2 in self.domains[row2][col2]:
                        if self.checkArc((row1, col1), num1, (row2, col2), num2) and self.checkArc((row2, col2), num2, (row3, col3), num3):
                            remove = False
                # If there is no X2=num2 that satisfies both {X1=num1, X2=num2} and {X2=num2, X3=num3}, then num1 is not a valid assignment for X1.
                # So, remove num1 from the domain of X1
                if remove:
                    self.domains[row1][col1].remove(num1)
                    domainChanged = True

            # If the domain of X1 was changed, add to the queue all paths ending on X1
            if domainChanged:
                for k in range(self.size):
                    if k != col1:
                        for z in range(self.size):
                            if z != k and z != col1:
                                queue.append(((row1, z), (row1, k), (row1, col1)))
                            if z != row1:
                                queue.append(((z, k), (row1, k), (row1, col1)))
                        for r, c in self.getBoxCoord(row1, k):
                            if (r, c) != (row1, k) and (r, c) != (row1, col1):
                                queue.append(((r, c), (row1, k), (row1, col1)))
                    if k != row1:
                        for z in range(self.size):
                            if z != k and z != row1:
                                queue.append(((z, col1), (k, col1), (row1, col1)))
                            if z != col1:
                                queue.append(((k, z), (k, col1), (row1, col1)))
                        for r, c in self.getBoxCoord(k, col1):
                            if (r, c) != (k, col1) and (r, c) != (row1, col1):
                                queue.append(((r, c), (k, col1), (row1, col1)))

    # Creates a copy of the current list of domains for backtracking              
    def copyDoms(self):
        domCopy = []
        for row in self.domains:
            rowCopy = []
            for x in row:
                rowCopy.append(x.copy())
            domCopy.append(rowCopy)
        return domCopy

    # Use Path Consistency to solve the Sudoku board
    def solve(self, ui):
        self.enforcePathConsistency()
        return self.backtrack(ui)

    def backtrack(self, ui):
        # Find an empty cell
        empty_cell = self.findEmptyCell()
        # If there are no empty cells, the puzzle is solved
        if not empty_cell:
            print("Backtracks: " + str(self.numBacktracks))
            return True  # Puzzle solved

        row, col = empty_cell

        # If a cell with an empty domain was found, this assignment will not work, so backtrack
        if(len(self.domains[row][col]) < 1):
            return False

        # For each number (num) in the domain of the selected cell:
        cellDoms = self.domains[row][col].copy()
        for num in cellDoms:
            # Assign the cell to be num
            self.board[row][col] = num
            domainCopy = self.copyDoms()
            self.domains[row][col] = [num]
            # Update the UI to display the current assignment
            ui.updateUI(self)
            # Enforce path consistency using the new assignment, and attempt to continue solving
            if self.solve(ui):
                return True
            # If this assignment did no work, backtrack and revert to previous state
            self.board[row][col] = 0  # Backtrack
            self.numBacktracks = self.numBacktracks + 1
            self.domains = domainCopy
            # Remove num from this cell's domain
            self.domains[row][col].remove(num)

        return False

    # Finds an unassigned cell with the smallest domain
    def findEmptyCell(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        return None

    def getBoard(self):
        return self.board


class BeamSudokuSolver:
    def __init__(self, initial_board):
        self.initial_board = initial_board
        self.board = initial_board
        self.size = len(initial_board)
        self.box_size = int(self.size ** 0.5)
        self.beam_width = 5
        
        
    def is_valid(self, board, num, row, col):
        # Check row, column, and block constraints
        block_row = (row // self.box_size) * self.box_size
        block_col = (col // self.box_size) * self.box_size

        for i in range(self.size):
            if board[row][i] == num or board[i][col] == num:
                return False
            
            if board[block_row + i // self.box_size][block_col + i % self.box_size] == num:
                return False
            
        return True

    def find_empty_cell(self, board):
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    return i, j
        return None
    
    def solve(self, ui):
        pq = []
        heappush(pq, (self.heuristic(self.initial_board), list(map(list, self.initial_board))))
        start_time = time.time()
        backtracks = 0
        while pq:
            _, current_board = heappop(pq)
            empty = self.find_empty_cell(current_board)
            if not empty:
                self.print_board(current_board)
                end_time = time.time()
                print("Duration:", round(end_time - start_time, 2), "seconds.")
                print("Number of backtracks:", backtracks)
                return True  # Solved

            row, col = empty
            next_states = []
            res = [i for i in range(1, self.size+1)]
            random.shuffle(res)
            for num in res:
                if self.is_valid(current_board, num, row, col):
                    new_board = list(map(list, current_board))
                    new_board[row][col] = num
                    self.board = new_board
                    ui.updateUI(self)
                    backtracks += 1
                    heappush(next_states, (self.heuristic(new_board), new_board))

            # Prune to maintain beam width
            for _ in range(min(self.beam_width, len(next_states))):
                heappush(pq, heappop(next_states))

        if self.beam_width < 9:
            self.beam_width += 1
            self.board = self.initial_board
            return self.solve(ui)
        else:
            print("No solution found.")
            return False

    def heuristic(self, board):
        # Simple heuristic: Count the number of empty cells
        return -sum(row.count(0) for row in board)

    def print_board(self, board):
        for row in board:
            print(" ".join(str(num) if num != 0 else '.' for num in row))
 
class ForwardCheckingSolver:
    def getDomains(self, board):
        domains = []
        for i in range(self.size):
            rowDoms = []
            for j in range(self.size):
                # Domain of unassigned cell is digits 1-9
                if board[i][j] == 0:
                    rowDoms.append([1,2,3,4,5,6,7,8,9])
                # Domain of pre-assigned cell is only the digit assigned to that cell 
                else:
                    rowDoms.append([board[i][j]])
            domains.append(rowDoms)
        return domains 
    
    # Create a list of domains (possible assignments) for each cell of the board
    def __init__(self, board):
        self.board = board
        self.size = len(board)
        self.box_size = int(self.size ** 0.5)
        self.domains = self.getDomains(board)
        self.numBacktracks = 0

    def isValid(self, row, col, num):
        # Check if the number is already in the row
        if num in self.board[row]:
            return False

        # Check if the number is already in the column
        if num in [self.board[i][col] for i in range(self.size)]:
            return False

        # Check if the number is already in the box
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i, j in itertools.product(range(self.box_size), repeat=2):
            if self.board[start_row + i][start_col + j] == num:
                return False
             
    def findEmptyCell(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        return None
    
    def enforceForwardChecking(self, row, col, assigned_num):
        # Remove assigned_num from the domains of cells in the same row
        for i in range(self.size):
            if i != col and assigned_num in self.domains[row][i]:
                self.domains[row][i].remove(assigned_num)
        
        # Remove assigned_num from the domains of cells in the same column
        for j in range(self.size):
            if j != row and assigned_num in self.domains[j][col]:
                self.domains[j][col].remove(assigned_num)
        
        # Remove assigned_num from the domains of cells in the same box
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        for i in range(self.box_size):
            for j in range(self.box_size):
                r, c = start_row + i, start_col + j
                if (r, c) != (row, col) and assigned_num in self.domains[r][c]:
                    self.domains[r][c].remove(assigned_num)

    #We should be looking at all of the numbers in tht domain                
    # Creates a copy of the current list of domains for backtracking              
    def copyDoms(self):
        domCopy = []
        for row in self.domains:
            rowCopy = []
            for x in row:
                rowCopy.append(x.copy())
            domCopy.append(rowCopy)
        return domCopy
    
    def solve(self, ui):
        #Loop through entire board and enforce forwardchecking
       
        for i in range(self.size):
       
            for j in range(self.size):
                # Domain of unassigned cell is digits 1-9
                if self.board[i][j] != 0:
                    self.enforceForwardChecking(i,j, self.board[i][j])      
        return self.backtrack(ui)
    
    def backtrack(self, ui):
        
        # Find an empty cell
        empty_cell = self.findEmptyCell()
        # If there are no empty cells, the puzzle is solved
        if not empty_cell:
            print("Backtracks: " + str(self.numBacktracks))
            return True  # Puzzle solved

        row, col = empty_cell

        # If a cell with an empty domain was found, this assignment will not work, so backtrack
        if(len(self.domains[row][col]) < 1):
            return False

        # For each number (num) in the domain of the selected cell:
        cellDoms = self.domains[row][col].copy()
        for num in cellDoms:
            # Assign the cell to be num
            self.board[row][col] = num
            domainCopy = self.copyDoms()
            self.domains[row][col] = [num]
            # Update the UI to display the current assignment
            ui.updateUI(self)
            
            self.enforceForwardChecking(row, col, num)

            # Enforce Forward consistency using the new assignment, and attempt to continue solving
            if self.backtrack(ui):
                return True
            # If this assignment did no work, backtrack and revert to previous state
            self.board[row][col] = 0  # Backtrack
            self.numBacktracks = self.numBacktracks + 1
            self.domains = domainCopy
            # Remove num from this cell's domain
            self.domains[row][col].remove(num)
        return False

    def getBoard(self):
        return self.board

    def forwardChecking(solver, row, col, num):
        # Enforce forward checking after assigning num to the cell at (row, col)
        solver.enforceForwardChecking(row, col, num)    


class SudokuUI:
    def __init__(self, master, initial_board, solver_algorithm):
        self.master = master
        self.initial_board = initial_board
        self.size = len(initial_board)
        self.solver_algorithm = solver_algorithm

        self.canvas = tk.Canvas(master, width=self.size * 50, height=self.size * 50)
        self.canvas.pack()

        self.drawGrid()
        self.drawNumbers(initial_board)
        self.solveAndDisplay()

    def drawGrid(self):
        for i in range(self.size + 1):
            line_width = 2 if i % self.size == 0 else 1
            self.canvas.create_line(i * 50, 0, i * 50, self.size * 50, width=line_width)
            self.canvas.create_line(0, i * 50, self.size * 50, i * 50, width=line_width)

    def drawNumbers(self, board):
        self.canvas.delete("numbers")
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] != 0:
                    self.canvas.create_text(j * 50 + 25, i * 50 + 25, text=str(board[i][j]), tags="numbers")

    def solveAndDisplay(self):
        if self.solver_algorithm == "AC":
            solver = ArcConsistencySolver(self.initial_board)
        elif self.solver_algorithm == "FC":
            solver = ForwardCheckingSolver(self.initial_board)
        elif self.solver_algorithm == "BS":
            solver = BeamSudokuSolver(self.initial_board)
        elif self.solver_algorithm == "PC":
            solver = PathConsistencySolver(self.initial_board)
        elif self.solver_algorithm == "BM":
            solver = BreakoutMethodSolver(self.initial_board)
        else:
            print("Usage: python sudoku.py BOARD ALG")
            sys.exit(1)

        self.drawNumbers(self.initial_board)

        start = time.time()

        if not solver.solve(self):
            print("No solution exists")
            sys.exit(1)
        else:
            end = time.time()
            print("Time: " + str(round(end - start)) + " seconds")
            self.drawNumbers(solver.board)

    def updateUI(self, solver):
        self.drawNumbers(solver.board)
        self.master.update()
        time.sleep(0.05)

def parseBoardFile(filename):
    try:
        with open(filename, 'r') as file:
            board = [[int(num) if num != '-' else 0 for num in line.split()] for line in file]
        return board
    except FileNotFoundError:
        print("Please select a file and an algorithm")
        sys.exit(1)

def main():
    
    algs = {
        "Forward Checking": "FC",
        "Arc Consistency": "AC",
        "Path Consistency": "PC",
        "Breakout Method": "BM",
        "Beam Stack Search": "BS"
    }

    boards = {
        "Board 1 (Easy)": "Sudoku Boards/board1.txt",
        "Board 2 (Easy)": "Sudoku Boards/board2.txt",
        "Board 3 (Medium)": "Sudoku Boards/board3.txt",
        "Board 4 (Medium)": "Sudoku Boards/board4.txt",
        "Board 5 (Hard)": "Sudoku Boards/board5.txt",
        "Board 6 (Hard)": "Sudoku Boards/board6.txt"
    }

    root = tk.Tk()
    root.title("Sudoku Solver")
    root.geometry("350x300")

    leftFrame = tk.Frame(root)
    rightFrame = tk.Frame(root)

    solver_algorithm = tk.StringVar(root, "1")
    filename = tk.StringVar(root, "1")

    tk.Label(leftFrame, text = "Algorithm", font = "Verdana 11 underline").pack(side = tk.TOP)
    for (text, val) in algs.items():
        tk.Radiobutton(leftFrame, text = text, variable = solver_algorithm, value = val).pack(side = tk.TOP, ipady = 5)
    tk.Label(leftFrame, text = "").pack(side = tk.TOP, ipady = 5)
    leftFrame.pack(side = tk.LEFT)

    tk.Label(rightFrame, text = "Board", font = "Verdana 11 underline").pack(side = tk.TOP)
    for (text, val) in boards.items():
        tk.Radiobutton(rightFrame, text = text, variable = filename, value = val).pack(ipady = 5)
    rightFrame.pack(side = tk.RIGHT)

    tk.Button(root, text = "Solve the Puzzle", command = root.destroy).pack(side = tk.BOTTOM)

    root.mainloop()

    root = tk.Tk()
    root.title("Sudoku Solver")

    initial_board = parseBoardFile(filename.get())
    sudoku_ui = SudokuUI(root, initial_board, solver_algorithm.get())

    root.mainloop()

if __name__ == "__main__":
    main()
