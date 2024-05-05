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
        t = list(zip(legalMoves, [round(i, 2) for i in scores]))
        print(t)
        print("===============")
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
        NUM_FOODS_WEIGHT = 9999
        NO_SELECT = -999999

        if successorGameState.isWin():
            return float("inf")
        elif successorGameState.isLose():
            return float("-inf")
        elif action == Directions.STOP:
            return float("-inf")

        # 고스트와 멀어지는 방향으로 가중치 설정
        # 전부다 -inf면 어떻게 할껀데,
        # 멀어지는 방향에 가중치를 두어야징..
        # print(newGhostStates[0])
        # print(newGhostStates)
        # ghostPos=newGhostStates[0].getPosition()
        # ghostDist = manhattanDistance(newPos, ghostPos)

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            if manhattanDistance(newPos, ghostPos) <= 2:
                return NO_SELECT

        # if ghostDist <= 2:
        #     curX, curY = newPos
        #     ghostX, ghostY = newGhostStates[0].getPosition()
        #     if (
        #         (action == Directions.WEST and curX > ghostX)
        #         or (action == Directions.EAST and ghostX > curX)
        #         or (action == Directions.NORTH and ghostY > curY)
        #         or (action == Directions.SOUTH and curY > ghostY)
        #     ):
        #         return float("-inf")

        # 2칸 이내에서는 가까워지는 방향은 -inf
        # 멀어지는 방향으로는 가중치 증가

        ghostDiff = 0

        # 벽도 해결해야됨..

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

        closestFoodDiff = 1 / minDist
        numFoods = 1 / len(newFood.asList())

        # if action in (Directions.EAST, Directions.WEST):
        #     pass
        # elif action in (Directions.SOUTH, Directions.NORTH):
        #     pass
        # else:
        #     diffWeight = float("-inf")
        linearF = closestFoodDiff + NUM_FOODS_WEIGHT * numFoods
        # print(action, closestFoodDiff, numFoods, linearF)
        # input()

        # return linearF
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

        # astar를 쓴 경로로 가는 방향에 가중치 up

        return linearF


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
        -> 그럼 유령이 몇번째까지 있는지 어떻게 알지?

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
        numAgents = gameState.getNumAgents()
        print(f"maxDepth: {self.depth}")

        def getNextAgent(agentIndex):
            return (agentIndex + 1) % numAgents

        def minimax(state: GameState, agentIndex: int, depth: int):
            if depth == self.depth or state.isWin() or state.isLose():
                # print(f"call eval function depth: {self.depth}")
                return (self.evaluationFunction(state), "")

            legalActions = state.getLegalActions(agentIndex)
            successors = [
                state.generateSuccessor(agentIndex, action) for action in legalActions
            ]

            # print(state, agentIndex)
            # input()

            nextAgent = getNextAgent(agentIndex)
            nextDepth = depth + 1 if nextAgent == 0 else depth

            evaluation = [
                minimax(successor, nextAgent, nextDepth) for successor in successors
            ]

            values, actions = zip(*evaluation)

            maxValue = max(values) if agentIndex == 0 else min(values)
            maxValueIndex = values.index(maxValue)
            maxAction = legalActions[maxValueIndex]

            return (maxValue, maxAction)

        # 재귀적으로 구성하라...?

        # return Directions.WEST
        maxValue, maxAction = minimax(gameState, 0, 0)
        return maxAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        ALPHA_INIT_VALUE = (float("-inf"), "")
        BETA_INIT_VALUE = (float("inf"), "")
        print(f"maxDepth: {self.depth} numAgents:{numAgents}")

        def getNextAgent(agentIndex):
            return (agentIndex + 1) % numAgents

        def getNextDepth(nextAgent, depth):
            return depth + 1 if nextAgent == 0 else depth

        def callNextAgent(
            successor: GameState, currentAgent, nextAgent, nextDepth, alpha, beta
        ):
            if nextAgent == 0:
                return maxAgent(successor, nextAgent, nextDepth, alpha, beta)
            else:
                return minAgent(successor, nextAgent, nextDepth, alpha, beta)

        # maxAgent구현중
        def maxAgent(state: GameState, agentIndex, depth, alpha, beta):
            if depth == self.depth or state.isLose() or state.isWin():
                ev = self.evaluationFunction(state)
                return (ev, "")

            v = ALPHA_INIT_VALUE
            betaValue, betaAction = beta  # +inf
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                # 생성하고
                successor = state.generateSuccessor(agentIndex, action)

                # 값을 내놔
                nextAgent = getNextAgent(agentIndex)
                nextDepth = getNextDepth(nextAgent, depth)
                value, prevAction = callNextAgent(
                    successor, agentIndex, nextAgent, nextDepth, alpha, beta
                )
                vaTuple = (value, action)

                v = max(v, vaTuple, key=lambda x: x[0])
                # 프루닝 조건
                if (
                    value > betaValue
                ):  # 기본값으로 beta가 +inf 이기 때문에 프루닝 될 수 없음.
                    return vaTuple

                # 프루닝이 안된경우
                # 다음도 봐야하면 알파값 업데이트
                alpha = max(
                    alpha, vaTuple, key=lambda x: x[0]
                )  # 항상 최신 액션을 따르기만 하면 문제는 없음

                # 최신액션을 따르는게 아니라 가장 최소 액션을 따라야함
                # v가 큰경우 v와 와 현재 action,
                # alpha가 큰경우 alpha와 alpha action

            return v

            # 다음 브랜치 탐색 ...??????
            # 다음브랜치를 어떻게 탐색하지?

        def minAgent(state: GameState, agentIndex, depth, alpha, beta):
            if depth == self.depth or state.isLose() or state.isWin():
                ev = self.evaluationFunction(state)

                return (ev, "")

            v = BETA_INIT_VALUE
            alphaValue, _ = alpha
            legalActions = state.getLegalActions(agentIndex)

            for action in legalActions:
                # 생성하고
                successor = state.generateSuccessor(agentIndex, action)

                # 값을 내놔
                nextAgent = getNextAgent(agentIndex)
                nextDepth = getNextDepth(nextAgent, depth)
                value, _ = callNextAgent(
                    successor, agentIndex, nextAgent, nextDepth, alpha, beta
                )
                vaTuple = (value, action)
                v = min(v, vaTuple, key=lambda x: x[0])

                # 프루닝 조건
                if value < alphaValue:
                    return vaTuple

                # 프루닝이 안된경우
                # 다음도 봐야하면 알파값 업데이트
                beta = min(
                    beta, vaTuple, key=lambda x: x[0]
                )  # 항상 최신 액션을 따르기만 하면 문제는 없음
                # 최신액션을 따르는게 아니라 가장 최소 액션을 따라야함
                # v가 큰경우 v와 와 현재 action,
                # alpha가 큰경우 alpha와 alpha action

            return v

        maxValue, maxAction = maxAgent(
            gameState, 0, 0, ALPHA_INIT_VALUE, BETA_INIT_VALUE
        )

        return maxAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        numAgents = gameState.getNumAgents()
        print(f"maxDepth: {self.depth} numAgents:{numAgents}")

        def getNextAgent(agentIndex):
            return (agentIndex + 1) % numAgents

        def exp(values):
            return sum(values) / len(values)

        def minimax(state: GameState, agentIndex: int, depth: int):
            if depth == self.depth or state.isWin() or state.isLose():
                # print(f"call eval function depth: {self.depth}")
                return (self.evaluationFunction(state), "")

            legalActions = state.getLegalActions(agentIndex)
            successors = [
                state.generateSuccessor(agentIndex, action) for action in legalActions
            ]

            nextAgent = getNextAgent(agentIndex)
            nextDepth = depth + 1 if nextAgent == 0 else depth

            evaluation = [
                minimax(successor, nextAgent, nextDepth) for successor in successors
            ]

            values, actions = zip(*evaluation)

            if agentIndex == 0:
                maxValue = max(values)
                maxValueIndex = values.index(maxValue)
                maxAction = legalActions[maxValueIndex]
                return (maxValue, maxAction)
            else:
                expValue = exp(values)
                return (expValue, "")

        # 재귀적으로 구성하라...?

        # return Directions.WEST
        maxValue, maxAction = minimax(gameState, 0, 0)
        return maxAction


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
