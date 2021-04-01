import numpy

class Area:

    def __init__(self, shape, margin):
        self.shape = shape
        self.margin = margin
        self.state = self.__init_state()


# public method
    def update_state(self, mvobj):
        y, x = mvobj.state
        self.state[y, x] = mvobj.value


    def reset_state(self):
        self.state = self.__init_state()


# private method
    def __init_state(self):
        return numpy.zeros(self.shape).astype('uint8')
