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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        t= list(zip(legalMoves, [round(i,2) for i in scores]))
        print(t, t[chosenIndex])
        # print("===============")
        # input()

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        # if action==Directions.STOP:
        #     return float('-inf')

        # 푸드리스트를 돌면서 일단 맨해튼거리를 찾고
        # 맨허튼거리가 최소인 푸드위치 반환
        # def getClosestFood() -> tuple[tuple[int, int], tuple[float, float]]:
        #     minDiffX, minDiffY = (float("inf"), float("inf"))
        #     curX, curY = newPos
        #     closestFoodPosition = (0, 0)
        #     for foodX, foodY in newFood.asList():

        #         diffX, diffY = (abs(curX - foodX), abs(curY - foodY))

        #         if diffX + diffY < minDiffX + minDiffY:
        #             minDiffX = diffX
        #             minDiffY = diffY
        #             closestFoodPosition = (foodX, foodY)

        #         print(newPos, (foodX, foodY), (diffX, diffY), (minDiffX, minDiffY))

        #     return (closestFoodPosition, (minDiffX, minDiffY))

        # closestFoodPosition, (diffX, diffY) = getClosestFood()

        # print(
        #     action,
        #     "newpos",
        #     newPos,
        #     "closestfood",
        #     closestFoodPosition,
        #     "diff",
        #     (diffX, diffY),
        #     end=" ",
        # )

        # diffWeight = 0
        # if action in (Directions.EAST, Directions.WEST):
        #     diffWeight = float("inf") if diffX == 0 else 1 / diffX
        # elif action in (Directions.SOUTH, Directions.NORTH):
        #     diffWeight = float("inf") if diffY == 0 else 1 / diffY
        # else:
        #     diffWeight = float("-inf")

        # print(diffWeight)
        # input()

        # return diffWeight
        NUM_FOODS_WEIGHT=9999
        
        if successorGameState.isWin():
            return float('inf')
        elif successorGameState.isLose():
            return float('-inf')
        elif action==Directions.STOP:
            return float('-inf')

        
        # 고스트와 멀어지는 방향으로 가중치 설정
        # print(newGhostStates[0])
        # print(newGhostStates)
        ghostDiff=manhattanDistance(newPos,newGhostStates[0].getPosition())
        # print('ghostDiff', ghostDiff)
        if ghostDiff<=2:
            return float('-inf')

        def getClosestFood() -> tuple[tuple[float, float], float]:
            minDist = float("inf")
            closestFoodPos = (0, 0)
            for foodPos in newFood.asList():
                dist = manhattanDistance(newPos, foodPos)
                if dist < minDist:
                    minDist = dist
                    closestFoodPos = foodPos
                # print(newPos, foodPos, dist, minDist, closestFoodPos)

            return (closestFoodPos, minDist)

        closestFood, minDist = getClosestFood()

        closestFoodDiff = 1/minDist
        numFoods=1/len(newFood.asList())
        
        # if action in (Directions.EAST, Directions.WEST):
        #     pass
        # elif action in (Directions.SOUTH, Directions.NORTH):
        #     pass
        # else:
        #     diffWeight = float("-inf")
        linearF=closestFoodDiff+NUM_FOODS_WEIGHT*numFoods
        # print(action, closestFoodDiff, numFoods, linearF)
        # input()

        return linearF
        ####################################################
        # 해당 푸드위치로 갈 수 있는 액션에 가중치를 크게 둠

        # 간략한 a star알고리즘을 작성하고,- >??
        # 1. 일단 음식을 줄이는 쪽으로 이동(이동거리가 작은쪽으로 우선이동) -> 가까이 가고 다음 선택지에서는 음식이 사라지니 음식갯수가 가중치에서 큰역할
        # 2.

        # 가까이 있는 방향으로 가중치를 높여야 하는데

        # 2. 주변에 음식이 없으면 음식이 있는 방향에 가중치를 둠 -> 음식이 어디있는지 있는지 bfs탐색이 필요함 -> 가까운
        # -> 음식이 어디있는지는 사실 알고 있고 -> 가까운 음식의 위치를 알고있으니 a*로 길이 참조
        # -> 길이의 값을 가중치로 삼음,
        # 팩맨이랑 가까워지는 쪽으로 가도 전혀 상관없으나 팩맨과 같은 위치로 갈 가능성이 있는 방향은 무한대의 가중치

        # 3. 음식이 가까울수록 가중치를 크게
        # 4. 그리고 대각선에서 죽을 수 있는 확률
        #

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
