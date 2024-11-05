// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/dmabuf_buffer.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace lrt {
namespace internal {

namespace {

class DmaBufLibrary {
 public:
  using Ptr = std::unique_ptr<DmaBufLibrary>;

  ~DmaBufLibrary() {
    if (allocator_) {
      free_allocator_(allocator_);
    }
  }

  static absl::StatusOr<Ptr> Create() {
    DlHandle dlhandle(::dlopen("libdmabufheap.so", RTLD_LAZY | RTLD_LOCAL),
                      ::dlclose);
    if (!dlhandle) {
      return absl::InternalError("libdmabufheap.so not found");
    }

    auto create_allocator = reinterpret_cast<CreateAllocator>(
        ::dlsym(dlhandle.get(), "CreateDmabufHeapBufferAllocator"));
    if (!create_allocator) {
      return absl::InternalError("CreateDmabufHeapBufferAllocator not found");
    }

    auto free_allocator = reinterpret_cast<FreeAllocator>(
        ::dlsym(dlhandle.get(), "FreeDmabufHeapBufferAllocator"));
    if (!free_allocator) {
      return absl::InternalError("FreeDmabufHeapBufferAllocator not found");
    }

    auto alloc_buffer = reinterpret_cast<AllocBuffer>(
        ::dlsym(dlhandle.get(), "DmabufHeapAlloc"));
    if (!alloc_buffer) {
      return absl::InternalError("DmabufHeapAlloc not found");
    }

    void* allocator = create_allocator();
    if (!allocator) {
      return absl::InternalError("CreateDmabufHeapBufferAllocator failed");
    }

    return Ptr(new DmaBufLibrary(std::move(dlhandle), allocator, free_allocator,
                                 alloc_buffer));
  }

  absl::StatusOr<DmaBufBuffer> Alloc(size_t size) {
    int fd = alloc_buffer_(allocator_, kDmaBufHeap, size, /*flags=*/0,
                           /*legacy_align=*/0);
    if (fd < 0) {
      return absl::InternalError("Failed to allocate DMA-BUF buffer");
    }
    void* addr =
        ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      return absl::InternalError("Failed to mem-map DMA-BUF buffer");
    }
    records_[addr] = Record{.fd = fd, .addr = addr, .size = size};
    return DmaBufBuffer{.fd = fd, .addr = addr};
  }

  void Free(void* addr) {
    auto iter = records_.find(addr);
    if (iter == records_.end()) {
      return;
    }
    auto& record = iter->second;
    ::munmap(record.addr, record.size);
    ::close(record.fd);
    records_.erase(iter);
  }

 private:
  static constexpr const char* kDmaBufHeap = "system";

  struct Record {
    int fd;
    void* addr;
    size_t size;
  };

  using DlHandle = std::unique_ptr<void, int (*)(void*)>;
  using CreateAllocator = void* (*)();
  using FreeAllocator = void (*)(void*);
  using AllocBuffer = int (*)(void*, const char*, size_t, unsigned int, size_t);

  DmaBufLibrary(DlHandle&& dlhandle, void* allocator,
                FreeAllocator free_allocator, AllocBuffer alloc_buffer)
      : dlhandle_(std::move(dlhandle)) {
    allocator_ = allocator;
    free_allocator_ = free_allocator;
    alloc_buffer_ = alloc_buffer;
  }

  DlHandle dlhandle_;
  void* allocator_;
  FreeAllocator free_allocator_;
  AllocBuffer alloc_buffer_;
  absl::node_hash_map<void*, Record> records_;
};

DmaBufLibrary* TheDmaBufLibrary;
ABSL_CONST_INIT absl::Mutex TheMutex(absl::kConstInit);

absl::Status InitLibraryIfNeededUnlocked() {
  if (!TheDmaBufLibrary) {
    if (auto library = DmaBufLibrary::Create(); library.ok()) {
      TheDmaBufLibrary = library->release();
    } else {
      return library.status();
    }
  }
  return {};
}

}  // namespace

bool DmaBufBuffer::IsSupported() {
  absl::MutexLock lock(&TheMutex);
  auto status = InitLibraryIfNeededUnlocked();
  return status.ok();
}

absl::StatusOr<DmaBufBuffer> DmaBufBuffer::Alloc(size_t size) {
  absl::MutexLock lock(&TheMutex);
  if (auto status = InitLibraryIfNeededUnlocked(); !status.ok()) {
    return status;
  }
  return TheDmaBufLibrary->Alloc(size);
}

void DmaBufBuffer::Free(void* addr) {
  absl::MutexLock lock(&TheMutex);
  if (TheDmaBufLibrary) {
    TheDmaBufLibrary->Free(addr);
  }
}

}  // namespace internal
}  // namespace lrt