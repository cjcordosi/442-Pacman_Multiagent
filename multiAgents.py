# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
		GhostPos = successorGameState.getGhostPositions()

		"*** YOUR CODE HERE ***"
		score = 0
		distFood = []
		distGhost = []
		if (len(distFood) == 0):
			score += 1
		else:
			for i in newFood.asList():
				dist = util.manhattanDistance(newPos, i)
				distFood.append(dist)
			ClosestFood = min(distFood)
			if (ClosestFood == 0):
				score += 1
			score += 1/ClosestFood

		for i in GhostPos:
			dist = util.manhattanDistance(newPos, i)
			distGhost.append(dist)
		ClosestGhost = min(distGhost) 
		if (ClosestGhost < 2):
			score -= 10
		return score + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.  Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.  Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.  It's
	only partially specified, and designed to be extended.  Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.

		Here are some method calls that might be useful when implementing minimax.

		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1

		gameState.generateSuccessor(agentIndex, action):
		Returns the successor game state after an agent takes an action

		gameState.getNumAgents():
		Returns the total number of agents in the game

		gameState.isWin():
		Returns whether or not the game state is a winning state

		gameState.isLose():
		Returns whether or not the game state is a losing state
		"""
		"*** YOUR CODE HERE ***"

		def value(self, gameState, depth, agentIndex):

			# If the gameState is a terminating state, evaluate the score
			if ((gameState.isWin()) or (gameState.isLose()) or (depth == self.depth)):
				return self.evaluationFunction(gameState)

			# If the agentIndex == 0, i.e. PacMan NOT ghost, maximize the score
			if (agentIndex==0):
				return max_value(self, gameState, depth, agentIndex)[0]

			# If the agentIndex != 0, i.e. ghost NOT PacMan, minimize the score
			else:
				return min_value(self, gameState, depth, agentIndex)[0]

		def max_value(self, gameState, depth, agentIndex):
			v = float('-inf')
			bestAction = None

			# For each action PacMan can take, we will be calculating a new value for
			# the same depth, but increased agent index (ghosts)
			for action in gameState.getLegalActions(agentIndex):
				successor = gameState.generateSuccessor(agentIndex, action)
				NewV = value(self, successor, depth, agentIndex + 1)

				# If the new value is greater than the previous greatest value, set new
				# best value and action and return
				if(NewV > v):
					v = NewV
					bestAction = action
			return v, bestAction

		def min_value(self, gameState, depth, agentIndex):
			v = float('inf')
			bestAction = None

			# For each action a ghost can take, we will be calculating a new value for
			# the same depth, but increased agent index (ghosts again)
			for action in gameState.getLegalActions(agentIndex):
				successor = gameState.generateSuccessor(agentIndex, action)
				nextAgentIndex = agentIndex + 1
				nextDepth = depth
				# ONLY change agent index to 0 (PacMan) and increase depth if we have 
				# finished making decisions for all ghosts at that depth
				if (gameState.getNumAgents() == agentIndex + 1):
					nextAgentIndex = 0
					nextDepth = depth + 1
				NewV = value(self, successor, nextDepth, nextAgentIndex)

				# If the new value is lesser than the previous least value, set new
				# best value and action and return
				if(NewV < v):
					v = NewV
					bestAction = action
			return v, bestAction

		v, action = (max_value(self, gameState, 0, 0))
		return action

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"

		def value(self, gameState, depth, agentIndex, alpha, beta):

			# If the gameState is a terminating state, evaluate the score
			if ((gameState.isWin()) or (gameState.isLose()) or (depth == self.depth)):
				return self.evaluationFunction(gameState)

			# If the agentIndex == 0, i.e. PacMan NOT ghost, maximize the score
			if (agentIndex==0):
				return max_value(self, gameState, depth, agentIndex, alpha, beta)[0]

			# If the agentIndex != 0, i.e. ghost NOT PacMan, minimize the score
			else:
				return min_value(self, gameState, depth, agentIndex, alpha, beta)[0]

		def max_value(self, gameState, depth, agentIndex, alpha, beta):
			v = float('-inf')
			bestAction = None

			# For each action PacMan can take, we will be calculating a new value for
			# the same depth, but increased agent index (ghosts)
			for action in gameState.getLegalActions(agentIndex):
				successor = gameState.generateSuccessor(agentIndex, action)
				nextAgentIndex = agentIndex + 1
				if (gameState.getNumAgents() == agentIndex + 1):
					nextAgentIndex = 0
				NewV = value(self, successor, depth, nextAgentIndex, alpha, beta)

				# If the new value is greater than the previous greatest value, set new
				# best value and action and return
				if(NewV > v):
					v = NewV
					bestAction = action
				if (v>beta):
					return v, action
				alpha = max(alpha, v)
			return v, bestAction

		def min_value(self, gameState, depth, agentIndex, alpha, beta):
			v = float('inf')
			bestAction = None

			# For each action a ghost can take, we will be calculating a new value for
			# the same depth, but increased agent index (ghosts again)
			for action in gameState.getLegalActions(agentIndex):
				successor = gameState.generateSuccessor(agentIndex, action)
				nextAgentIndex = agentIndex + 1
				nextDepth = depth

				# ONLY change agent index to 0 (PacMan) and increase depth if we have 
				# finished making decisions for all ghosts at that depth
				if (gameState.getNumAgents() == agentIndex + 1):
					nextAgentIndex = 0
					nextDepth = depth + 1
				NewV = value(self, successor, nextDepth, nextAgentIndex, alpha, beta)

				# If the new value is lesser than the previous least value, set new
				# best value and action and return
				if(NewV < v):
					v = NewV
					bestAction = action
				if (v<alpha):
					return v, action
				beta = min(beta, v)
			return v, bestAction

		v, action = (max_value(self, gameState, 0, 0, float('-inf'), float('inf')))
		return action


class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		"*** YOUR CODE HERE ***"

		def value(self, gameState, depth, agentIndex):

			# If the gameState is a terminating state, evaluate the score
			if ((gameState.isWin()) or (gameState.isLose()) or (depth == self.depth)):
				return self.evaluationFunction(gameState)

			# If the agentIndex == 0, i.e. PacMan NOT ghost, maximize the score
			if (agentIndex==0):
				return max_value(self, gameState, depth, agentIndex)[0]

			# If the agentIndex != 0, i.e. ghost NOT PacMan, minimize the score
			else:
				return exp_value(self, gameState, depth, agentIndex)

		def max_value(self, gameState, depth, agentIndex):
			v = float('-inf')
			bestAction = None

			# For each action PacMan can take, we will be calculating a new value for
			# the same depth, but increased agent index (ghosts)
			for action in gameState.getLegalActions(agentIndex):
				successor = gameState.generateSuccessor(agentIndex, action)
				nextAgentIndex = agentIndex + 1
				if (gameState.getNumAgents() == agentIndex + 1):
					nextAgentIndex = 0
				NewV = value(self, successor, depth, nextAgentIndex)

				# If the new value is greater than the previous greatest value, set new
				# best value and action and return
				if(NewV > v):
					v = NewV
					bestAction = action
			return v, bestAction

		def exp_value(self, gameState, depth, agentIndex):
			v = 0

			# For each action a ghost can take, we will be calculating a new value for
			# the same depth, but increased agent index (ghosts again)
			for action in gameState.getLegalActions(agentIndex):
				successor = gameState.generateSuccessor(agentIndex, action)
				nextAgentIndex = agentIndex + 1
				nextDepth = depth

				# ONLY change agent index to 0 (PacMan) and increase depth if we have 
				# finished making decisions for all ghosts at that depth
				if (gameState.getNumAgents() == agentIndex + 1):
					nextAgentIndex = 0
					nextDepth = depth +1
				NewV = value(self, successor, nextDepth, nextAgentIndex)

				# Instead of finding the best value and action, this time we find the expected
				# value by using uniform distribution to find the probability of the successor
				# being chosen. Multiply the probability by the new value, and add it to the value.
				p = 1/len(gameState.getLegalActions(agentIndex))
				v += p * NewV
			return v

		v, action = (max_value(self, gameState, 0, 0))
		return action

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: <write something here so we know what you did>
	"""

	"*** YOUR CODE HERE ***"

	# Basic gameState information based upon currentGameState
	newPos = currentGameState.getPacmanPosition()
	newFood = currentGameState.getFood()
	newGhostStates = currentGameState.getGhostStates()
	newGhostPositions = currentGameState.getGhostPositions()
	newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
	currentScore = currentGameState.getScore()

	# Create a list of food distances, and calculate the manhattanDistance
	# from PacMan current position to each food
	foodDist = []
	for food in newFood.asList():
		foodDist.append(manhattanDistance(newPos, food))

	# Create a list of ghosts, and if they are not scared, append them to a list
	# of normal ghosts. If there exists at least one normal ghost, get each ghosts
	# position and calculate the manhattanDistance to Pacman current position. Get
	# the location of the closest normal ghost.
	ghosts = []
	ghostDist = []
	for ghost in newGhostStates:
		if (ghost.scaredTimer == 0):
			ghosts.append(ghost)
	if(len(ghosts) > 0):
		for ghost in ghosts:
			ghostPos = ghost.getPosition()
			ghostDist.append(manhattanDistance(newPos, ghostPos))
		closestGhost = min(ghostDist)
	
	# Create a list of ghosts, and if they are scared, append them to a list
	# of scared ghosts. If there exists at least one scared ghost, get each scared ghosts
	# position and calculate the manhattanDistance to Pacman current position. Get
	# the location of the closest scared ghost.
	scaredGhosts = []
	scaredGhostDist = []
	for ghost in newGhostStates:
		if (ghost.scaredTimer != 0):
			scaredGhosts.append(ghost)
	if(len(scaredGhosts) > 0):
		for ghost in scaredGhosts:
			ghostPos = ghost.getPosition()
			scaredGhostDist.append(manhattanDistance(newPos, ghostPos))
		closestScaredGhost = min(scaredGhostDist)
	
	# For the closest food, use this calculate to penalize the score
	# based on the manhattanDistance to that food. The further the food, 
	# the higher the score penalty, therefor distance from food is discouraged.
	if len(foodDist) > 0:
		closestFood = min(foodDist)
		currentScore -= 1 * closestFood
	
	# For the closest ghost, use this calculation to penalize the score
	# based on the manhattan distance to that ghost. The closer the ghost,
	# the higher the score penalty, therefore distance from ghosts is encouraged.
	if len(ghostDist) > 0 and closestGhost != 0:
		currentScore -= 2 * (1/closestGhost)

	# For the closest scared ghost, use this calculation to penalize the score
	# based on the manhattan distance to that scared ghost. The further the scared ghost,
	# the higher the score penalty, therefore closeness to scared ghosts is encouraged for
	# bigger point bonuses.
	if len(scaredGhostDist) > 0 and closestScaredGhost != 0:
		currentScore -= 5 * (closestScaredGhost)

	return currentScore


# Abbreviation
better = betterEvaluationFunction
