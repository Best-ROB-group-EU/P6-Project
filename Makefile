CC = "g++"
PROJECT = P6-PROJECT
SRC = main.cpp

LIBS = `pkg-config opencv4 --cflags --libs -lcnpy -lz --std=c++11`


$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)