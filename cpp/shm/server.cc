#include <fcntl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <thread>

#include "common.h"

const char *shm_path = "/sample";

int do_systemv() {
  key_t shm_key = ftok(shm_path, 0);
  int shmid = shmget(shm_key, sizeof(int), 0666 | IPC_CREAT);
  if (shmid == -1) {
    return 1;
  }
  MAKE_DEFER(shmctl(shmid, IPC_RMID, NULL));

  void *shm_addr = shmat(shmid, NULL, 0);
  if (shm_addr == (void *)-1) {
    return 1;
  }
  MAKE_DEFER(shmdt(shm_addr));

  *(int *)(shm_addr) = 100;
  while (*(int *)(shm_addr) != 101) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "recv signal from client" << std::endl;
  return 0;
}

int do_posix() {
  int fd = shm_open(shm_path, O_CREAT | O_RDWR, 0777);
  if (fd < 0) {
    return 1;
  }
  MAKE_DEFER(shm_unlink(shm_path));

  if (ftruncate(fd, 4096)) {
    return 1;
  }

  void *buf = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (buf == MAP_FAILED) {
    return 1;
  }
  MAKE_DEFER(munmap(buf, 4096));

  *(int *)(buf) = 100;
  while (*(int *)(buf) != 101) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return 0;
}

int main(int argc, char *argv[]) { return do_posix(); }
