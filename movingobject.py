import numpy
import random
import copy

class MovingObject:

    def __init__(self, area):
        self.state = self.__random_state(area)
        self.actions = self.__create_actions(area.shape)
        self.moves = self.__create_moves()


# public method
    def reset_state(self, area):
        self.state = self.__random_state(area)


    def get_act_index(self, epsilon):
        return self.__select_action(epsilon)


    def move_state(self, act_index):
        return self.state + self.moves.get(act_index)


# private method
    def __random_state(self, area):
        h, w = area.shape
        margin = area.margin

        while True:
            y = numpy.random.randint(margin, h - margin)
            x = numpy.random.randint(margin, w - margin)

            if area.state[y, x] == 0:
                break

        return numpy.array([y, x])


    def __create_actions(self, area_shape):
        actions = []
        h, w = area_shape

        for j in range(h):
            actions.append([])
            for i in range(w):
                action_org = [0, 1, 2, 3]
                action = self.__remove_actions(action_org, h, w, j, i)
                actions[j].append(action)

        return numpy.array(actions)


    def __remove_actions(self, action_org, h, w, j, i):
        action = copy.deepcopy(action_org)
        margin = 2
        if j == margin:
            action.remove(3)
        if j == h - margin - 1:
            action.remove(1)
        if i == margin:
            action.remove(0)
        if i == w - margin - 1:
            action.remove(2)

        return action


    def __create_moves(self):
        # 0->left, 1->down, 2-> right, 3->up
        return {0: numpy.array([0, -1]), 1: numpy.array([1, 0]), 2: numpy.array([0, 1]), 3: numpy.array([-1, 0])}


    def __select_action(self, epsilon):
        y, x = self.state
        action = self.actions[y, x]
        if numpy.random.rand() > epsilon:
            return self.__greedy_action(action)
        else:
            return self.__random_action(action)


    def __greedy_action(self, action):
        sy, sx = self.around
        print('aa:',sy, sx)
        qmax = self.q[action, sy, sx].max()
        indexes = list(numpy.argwhere(self.q[action, sy, sx] == qmax))
        random.shuffle(indexes)

        return action[indexes[0]]


    def __random_action(self, action):
        act_indexes = copy.deepcopy(action)
        random.shuffle(act_indexes)

        return act_indexes[0]
