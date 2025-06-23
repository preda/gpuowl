// Copyright (C) Mihai Preda

#pragma once

#include "common.h"
#include "clwrap.h"
#include "Context.h"
#include "Args.h"
#include "Event.h"

#include <deque>
#include <vector>

class Args;
class TimeInfo;

class Events : public std::deque<Event> {
public:
  void clearCompleted();
  void synced();
};

class Queue : public QueueHolder {
  Events events;
  bool hasEvents;

  void writeTE(cl_mem buf, u64 size, const void* data, TimeInfo *tInfo);
  void fillBufTE(cl_mem buf, u32 patSize, const void* pattern, u64 size, TimeInfo* tInfo);
  void flush();
  void print();
  void add(EventHolder &&e, TimeInfo* ti);

public:
  const Context* context;

  Queue(const Context& context, bool profile);

  static int registerThread();
  static int tid();

  template<typename T>
  void write(cl_mem buf, const vector<T>& v, TimeInfo* tInfo) { writeTE(buf, v.size() * sizeof(T), v.data(), tInfo); }

  template<typename T>
  void fillBuf(cl_mem buf, T pattern, u32 size, TimeInfo* tInfo) { fillBufTE(buf, sizeof(T), &pattern, size, tInfo); }

  void run(cl_kernel kernel, size_t groupSize, size_t workSize, TimeInfo* tInfo);
  void readSync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void readAsync(cl_mem buf, u32 size, void* out, TimeInfo* tInfo);
  void copyBuf(cl_mem src, cl_mem dst, u32 size, TimeInfo* tInfo);
  void finish();

  void setSquareTime(int);          // Set the time to do one squaring (in microseconds)

private:                            // This replaces the "call queue->finish every 400 squarings" code in Gpu.cpp.  Solves the busy wait on nVidia GPUs.
  int MAX_QUEUE_COUNT;              // Queue size before a marker will be enqueued.  Typically, 100 to 1000 squarings.
  cl_event markerEvent;             // Event associated with an enqueued marker placed in the queue every MAX_QUEUE_COUNT entries and before r/w operations.
  bool markerQueued;                // TRUE if a marker and event have been queued
  int queueCount;                   // Count of items added to the queue since last marker
  int squareTime;                   // Time to do one squaring (in microseconds)
  void queueMarkerEvent();          // Queue the marker event
  void waitForMarkerEvent();        // Wait for marker event to complete
};
