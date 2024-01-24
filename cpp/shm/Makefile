CPPFLAGS := -O0 -g
LDFLAGS := -lrt

all: server client

clean:
	rm -rf server client

server: server.cc
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LDFLAGS)

client: client.cc
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LDFLAGS)
