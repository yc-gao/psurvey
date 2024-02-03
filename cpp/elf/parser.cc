#include "elf.h"
#include "fcntl.h"
#include "sys/mman.h"
#include "sys/stat.h"
#include "unistd.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#define Elf_Ehdr Elf64_Ehdr
#define Elf_Shdr Elf64_Shdr
#define Elf_Sym Elf64_Sym

class FileMapping {
  int fd;
  std::size_t size;
  std::uint8_t *addr;

public:
  ~FileMapping() {
    if (fd != -1) {
      munmap(addr, size);
      close(fd);
    }
  }
  FileMapping(const FileMapping &) = delete;
  FileMapping &operator=(const FileMapping &) = delete;
  FileMapping(FileMapping &&) = delete;
  FileMapping &operator=(FileMapping &&) = delete;

  FileMapping(const char *fname) : fd(open(fname, O_RDONLY)) {
    if (Good()) {
      struct stat buffer;
      assert(!fstat(fd, &buffer));
      addr = (std::uint8_t *)mmap(NULL, buffer.st_size, PROT_READ, MAP_PRIVATE,
                                  fd, 0);
      assert(addr != MAP_FAILED);
    }
  }
  bool Good() const { return fd != -1; }
  std::uint8_t *Addr() { return addr; }
};

Elf_Shdr *elf_get_section(Elf_Ehdr *elf_hdr, int idx) {
  if (idx >= elf_hdr->e_shnum) {
    return nullptr;
  }
  return (Elf_Shdr *)((char *)elf_hdr + elf_hdr->e_shoff +
                      elf_hdr->e_shentsize * idx);
}

Elf_Shdr *elf_find_section(Elf_Ehdr *elf_hdr, const char *name) {
  Elf_Shdr *str = elf_get_section(elf_hdr, elf_hdr->e_shstrndx);
  for (int i = 0; i < elf_hdr->e_shnum; i++) {
    Elf_Shdr *elf_shdr = elf_get_section(elf_hdr, i);
    const char *tmp =
        (const char *)(elf_hdr) + str->sh_offset + elf_shdr->sh_name;
    if (strcmp(name, tmp) == 0) {
      return elf_shdr;
    }
  }
  return nullptr;
}

void DumoSymbol(Elf_Ehdr *elf_hdr) {
  Elf_Shdr *str_section = elf_find_section(elf_hdr, ".strtab");
  Elf_Shdr *tab_section = elf_find_section(elf_hdr, ".symtab");
  assert(str_section && tab_section);
  for (int j = 0; j < tab_section->sh_size / tab_section->sh_entsize; j++) {
    Elf_Sym *sym = (Elf_Sym *)((char *)elf_hdr + tab_section->sh_offset +
                               j * tab_section->sh_entsize);
    std::cout << "idx: " << std::dec << j << ", name: "
              << (const char *)(elf_hdr) + str_section->sh_offset + sym->st_name
              << ", st_shndx: " << sym->st_shndx << ", value: 0x" << std::hex
              << sym->st_value << std::endl;
  }
}

int main(int argc, char *argv[]) {
  assert(argc == 2);
  FileMapping fmap(argv[1]);

  Elf_Ehdr *elf_hdr = (Elf_Ehdr *)fmap.Addr();
  DumoSymbol(elf_hdr);
  return 0;
}
