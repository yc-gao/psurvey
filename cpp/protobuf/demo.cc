#include <cassert>
#include <iostream>
#include <sstream>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

#include "demo.pb.h"

int main(int argc, char *argv[]) {
  {
    std::stringstream ss;
    Demo1 demo1;
    demo1.set_name("demo");
    demo1.SerializeToOstream(&ss);

    Demo2 demo2;
    demo2.ParseFromIstream(&ss);
    assert(demo2.name() == "demo");
    google::protobuf::io::OstreamOutputStream os(&std::cout);
    google::protobuf::TextFormat::Print(demo2, &os);
  }

  {
    std::stringstream ss;
    {
      Demo2 demo2;
      demo2.set_name("demo");
      demo2.set_age(12);
      demo2.SerializeToOstream(&ss);
    }
    {
      Demo1 demo1;
      demo1.ParseFromIstream(&ss);
      ss.clear();
      assert(demo1.name() == "demo");
      google::protobuf::io::OstreamOutputStream os(&std::cout);
      google::protobuf::TextFormat::Print(demo1, &os);

      demo1.SerializeToOstream(&ss);
    }
    {
      Demo2 demo2;
      demo2.ParseFromIstream(&ss);
      assert(demo2.name() == "demo");
      assert(demo2.age() == 12);
      google::protobuf::io::OstreamOutputStream os(&std::cout);
      google::protobuf::TextFormat::Print(demo2, &os);
    }
  }
  return 0;
}
