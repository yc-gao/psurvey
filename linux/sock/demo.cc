#include <arpa/inet.h>
#include <linux/ip.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]) {
  int raw_socket = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
  if (raw_socket == -1) {
    perror("create socket failed");
    return 1;
  }
  char buffer[1024];
  while (1) {
    auto size = recv(raw_socket, buffer, sizeof(buffer), 0);
    if (size == -1) {
      perror("recv socket failed");
      return 1;
    }
    if (!size) {
      break;
    }
    std::cout << "new inet packet, " << size << " bytes" << std::endl;
    struct iphdr *ip_header = (struct iphdr *)buffer;
    std::cout << "version: " << ((int)ip_header->version & 0xff) << std::endl;
    std::cout << "ttl: " << ((int)ip_header->ttl & 0xff) << std::endl;
    std::cout << "protocol: " << ((int)ip_header->protocol & 0xff) << std::endl;
    std::cout << "saddr: 0x" << std::hex << ntohl(ip_header->saddr)
              << std::endl;
    std::cout << "daddr: 0x" << std::hex << ntohl(ip_header->daddr)
              << std::endl;
  }
  return 0;
}
