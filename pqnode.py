from queue import PriorityQueue


class Node:
    uniqueId = 0

    def __lt__(self, other):
        if self.f == other.f:
            if self.g == other.g:
                if self.h == other.h:
                    return self.id < other.id
                return self.h < other.h
            return self.g < other.g
        return self.f < other.f

    def __init__(self, pos, g, h):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.id = Node.getUniqueId()

    @classmethod
    def getUniqueId(cls):
        Node.uniqueId += 1
        return Node.uniqueId


