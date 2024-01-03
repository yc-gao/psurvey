#include <iostream>

#include "google/protobuf/text_format.h"

#include "foo.pb.h"

int main(int argc, char *argv[]) {

  Group g;
  {
    Person *p = g.add_persons();
    p->set_name("demo");
  }
  {
    Person *p = g.add_persons();
    p->set_name("assd");
  }

  std::string buf;
  google::protobuf::TextFormat::PrintToString(g, &buf);
  std::cout << buf;

  {
    Group g;
    google::protobuf::TextFormat::ParseFromString(buf, &g);
    for (auto &&p : g.persons()) {
      std::cout << p.name() << std::endl;
    }
  }

  return 0;
}
