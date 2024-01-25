#include <fcntl.h>
#include <sys/mman.h>
#include <sys/shm.h>

#include <iostream>
#include <thread>

#include "common.h"

const char *shm_path = "/sample";

int do_systemv() {

  key_t shm_key = ftok(shm_path, 0);
  int shmid = shmget(shm_key, sizeof(int), 0666);
  if (shmid == -1) {
    return 1;
  }
  MAKE_DEFER(shmctl(shmid, IPC_RMID, NULL));

  void *shm_addr = shmat(shmid, NULL, 0);
  if (shm_addr == (void *)-1) {
    return 1;
  }
  MAKE_DEFER(shmdt(shm_addr));

  while (*(int *)(shm_addr) != 100) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "recv signal from server" << std::endl;
  (*(int *)(shm_addr))++;
  return 0;
}

int do_posix() {
  auto buf = ShmArea::Attach(shm_path, 4096);

  while (*(int *)(buf.get()) != 100) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "recv signal from server" << std::endl;
  (*(int *)(buf.get()))++;

  return 0;
}

int main(int argc, char *argv[]) { return do_posix(); }
