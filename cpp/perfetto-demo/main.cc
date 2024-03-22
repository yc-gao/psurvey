#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>

#include "perfetto_trace.pb.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

bool do_load(perfetto::protos::Trace *trace, const std::string &fpath,
             const std::string &format) {
  std::istream *in = &std::cin;
  std::fstream fs;
  if (fpath != "cin" && fpath != "-") {
    fs = std::fstream(fpath, std::ios::in);
    in = &fs;
  }
  if (format == "bin") {
    return trace->ParseFromIstream(in);
  }

  google::protobuf::io::IstreamInputStream ins(in);
  if (format == "text") {
    return google::protobuf::TextFormat::Parse(&ins, trace);
  }
  return false;
}

int do_parse(perfetto::protos::Trace *trace, char **argv, int argc) {
  std::string format = "text";
  for (int i = 0; i < argc;) {
    if (std::strcmp(argv[i], "-f") == 0) {
      format = argv[i + 1];
      i += 2;
    } else if (std::strcmp(argv[i], "-i") == 0) {
      if (do_load(trace, argv[i + 1], format)) {
        i += 2;
        return i;
      }
      return -1;
    } else {
      return -1;
    }
  }
  return -1;
}

bool do_dump(perfetto::protos::Trace *trace, const std::string &fpath,
             const std::string &format) {
  std::ostream *out = &std::cout;
  std::fstream fs;
  if (fpath != "cout" && fpath != "-") {
    fs = std::fstream(fpath, std::ios::out);
    out = &fs;
  }
  if (format == "bin") {
    return trace->SerializeToOstream(out);
  }

  google::protobuf::io::OstreamOutputStream outs(out);
  if (format == "text") {
    return google::protobuf::TextFormat::Print(*trace, &outs);
  }
  return false;
}

int do_print(perfetto::protos::Trace *trace, char **argv, int argc) {
  std::string format = "text";
  for (int i = 0; i < argc;) {
    if (std::strcmp(argv[i], "-f") == 0) {
      format = argv[i + 1];
      i += 2;
    } else if (std::strcmp(argv[i], "-o") == 0) {
      if (do_dump(trace, argv[i + 1], format)) {
        i += 2;
        return i;
      }
      return -1;
    } else {
      return -1;
    }
  }
  return -1;
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  perfetto::protos::Trace trace;
  argc--;
  argv++;
  int idx;
  if ((idx = do_parse(&trace, argv, argc)) != -1) {
    argc -= idx;
    argv += idx;
  } else {
    return 1;
  }

  while (argc) {
    if ((idx = do_print(&trace, argv, argc)) != -1) {
      argc -= idx;
      argv += idx;
    } else {
      return 1;
    }
  }
  return 0;
}
