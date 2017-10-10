TEMPLATE = lib
INCLUDEPATH += .
DEPENDPATH += .
VERSION = 0.99
QT -= gui core
win32:!win32-g++ { # VS solutions don't like having two projects named "quackle"
  TARGET = libquackle
}

debug {
  OBJECTS_DIR = obj/debug
  DESTDIR = lib/debug
}

release {
  OBJECTS_DIR = obj/release
  DESTDIR = lib/release
}

QMAKE_CXXFLAGS += -std=c++11 -Wall -w
QMAKE_LFLAGS += -lstdc++fs

# enable/disable debug symbols
#CONFIG += debug staticlib
CONFIG += release staticlib
CONFIG -= x11

# Input
HEADERS += *.h
HEADERS += inference/infer.h
INCLUDEPATH += inference/include

SOURCES += $$files(*.cpp)
SOURCES += inference/infer.cpp
SOURCES -= \
	loaddawg.cpp \
	loadgaddag.cpp \
	makedawg.cpp \
	quackletest.cpp

macx-g++ {
    QMAKE_CXXFLAGS += -fpermissive
}

macx-xcode {
	CONFIG += x86
}
