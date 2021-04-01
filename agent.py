from model.movingobject import *
import numpy

class Agent(MovingObject):

    def __init__(self, area):
        MovingObject.__init__(self, area)
        self.value = 255
        self.q = self.__creat_q_table()
        self.around = numpy.zeros(2).astype('int')
        self.act_index = -1
        self.reward = 0


# public method
    def lookout(self, area):
        y, x = self.state
        self.around = self.__get_object_states(area, y, x)


    def act(self, area, epsilon):
        _y, _x = self.state
        area.state[_y, _x] = 0
        if self.__is_alone():
            epsilon = 1.0
        self.act_index = MovingObject.get_act_index(self, epsilon)
        y, x = MovingObject.move_state(self, self.act_index)
        if area.state[y, x] == 0:
            self.state = numpy.array([y, x])
            area.state[y, x] = self.value
        else:
            area.state[_y, _x] = self.value


    def update_q(self, area, alpha, gamma):
        sy, sx = self.around
        self.reward = self.__get_reward(area)
        act_index = self.act_index
        q = self.q[act_index, sy, sx]
        self.q[act_index, sy, sx] = q + alpha * (self.reward + gamma * self.__get_max_q() - q)


    def save_q(self, fname):
        numpy.save(fname, self.q)


# private method
    def __creat_q_table(self):
        q = numpy.zeros((4, 26, 26))

        return q

    def __get_object_states(self, area, y, x):
        margin = area.margin
        around = area.state[y - margin: y + margin + 1, x - margin: x + margin + 1].reshape((1, -1))
        t_state = numpy.argwhere(around == 127)[:, 1]
        _a_state = numpy.argwhere(around == 255)[:, 1]
        a_state = numpy.delete(_a_state, numpy.argwhere(_a_state == 12)[:, 0], axis = 0)
        if numpy.array_equal(t_state, numpy.array([])):
            t_state = numpy.array([25])
        if numpy.array_equal(a_state, numpy.array([])):
            a_state = numpy.array([25])

        return numpy.r_[t_state, a_state]


    def __is_alone(self):
        return numpy.array_equal(self.around, numpy.array([25, 25]))


    def __get_reward(self, area):
        return 3 if self.__is_surrounding(area) else -1


    def __is_surrounding(self, area):
        t_state = numpy.argwhere(area.state == 127)
        a_state = numpy.argwhere(area.state == 255)

        check = numpy.array_equal(numpy.abs((a_state - t_state).sum(axis = 1)), numpy.array([1, 1]))
        check += numpy.array_equal(a_state.mean(axis = 0), t_state)

        return check


    def __get_max_q(self):
        y, x = self.state
        actions = self.actions[y, x]
        sy, sx = self.around
        _sy, _sx = self.__next_around(sy, sx)
        return self.q[actions, _sy, _sx].max()


    def __next_around(self, sy, sx):
        act_index = self.act_index
        _sy = self.__next_state(act_index, sy)
        _sx = self.__next_state(act_index, sx)

        return numpy.array([_sy, _sx])


    def __next_state(self, act_index, z):
        if z != 25:
            if act_index == 0 and (z + 1) % 5 != 0:
                z += 1
            elif act_index == 1 and z / 5 != 0:
                z -= 5
            elif act_index == 2 and z % 5 != 0:
                z -= 1
            elif act_index == 3 and z / 5 != 4:
                z += 5
            else:
                z = 25

        return z
