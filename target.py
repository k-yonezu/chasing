from model.movingobject import *
import numpy

class Target(MovingObject):

    def __init__(self, area):
        MovingObject.__init__(self, area)
        self.value = 127


# public method
    def act(self, area):
        _y, _x = self.state
        if self.__check_catched(area, _y, _x) == False:
            area.state[_y, _x] = 0
            while True:
                act_index = MovingObject.get_act_index(self, 1.0)
                y, x = MovingObject.move_state(self, act_index)
                if area.state[y, x] == 0:
                    self.state = numpy.array([y ,x])
                    area.state[y, x] = self.value
                    break


# private method
    def __check_catched(self, area, _y, _x):
        check = self.__is_surrounded(area)
        check *= self.__is_atcorners(area, _y, _x)

        return check


    def __is_surrounded(self, area):
        t_state = numpy.argwhere(area.state == 127)
        a_state = numpy.argwhere(area.state == 255)

        return numpy.array_equal(numpy.abs((a_state - t_state).sum(axis = 1)), numpy.array([1, 1]))


    def __is_atcorners(self, area, _y, _x):
        h, w = area.shape
        margin = 2
        c = _y == margin or _y == h - margin - 1
        c *= _x == margin or _x == w - margin - 1

        return c
