# distutils: language = c++
# distutils: sources = move_create.cpp
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "move_create.h":
    cdef cppclass Move_Create:
        void getMove(const string &, const string &)
        vector[float] getFeatures(const string & , const string &)
        Move_Create() except +
        void setGame(const string &)
        @staticmethod
        void init()
        string boardAfterMoveMade()
        string board()


cdef class PyMove:
    cdef Move_Create *thisptr
    # cdef void init = Move_Create.init()
    def __cinit__(self):
        self.thisptr = new Move_Create()
    def __dealloc__(self):
        del self.thisptr
    def setGame(self, file):
        self.thisptr.setGame(file)
    def getMove(self, s1, s2):
        return self.thisptr.getMove(s1, s2)
    def getFeatures(self, s1, s2):
        return self.thisptr.getFeatures(s1, s2)
    def board(self):
        return self.thisptr.board()
    def boardAfterMoveMade(self):
        return self.thisptr.boardAfterMoveMade()
    @staticmethod
    def init():
        Move_Create.init()