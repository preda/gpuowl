// Copyright (C) 2017-2024 Mihai Preda.

#include "gpuid.h"
#include "clwrap.h"
#include "File.h"

using namespace std;

static bool startsWith(string_view a, string_view b) {
  return a.substr(0, b.length()) == b;
}

string getBdfFromSysfs(int pos) {
  assert(pos >= 0);
  
  const string PREFIX = "PCI_SLOT_NAME=";
  const string SKIP = "0000:";

  if (File f = File::openRead("/sys/class/drm/card"s + std::to_string(pos) + "/device/uevent")) {
    for (const auto& line : f) {
      if (startsWith(line, PREFIX)) {
        string bdf = line.substr(PREFIX.length());
        if (startsWith(bdf, SKIP)) { bdf = bdf.substr(SKIP.length()); }
        if (!bdf.empty() && bdf.back() == '\n') { bdf.pop_back(); }
        return bdf;
      }
    }
  }
  return "";
}

/* BDF is PCIe Bus:Device.Function e.g. "6a:00.0" */
int getSysfsFromBdf(const string& wantBdf) {  
  for (u32 pos = 0; filesystem::exists("/sys/class/drm/card"s + std::to_string(pos)); ++pos) {
    if (wantBdf == getBdfFromSysfs(pos)) { return pos; }
  }
  return -1; // not found
}

int getSysfsFromUid(const string& wantUid) {
  for (u32 pos = 0; filesystem::exists("/sys/class/drm/card"s + std::to_string(pos)); ++pos) {
    if (File f = File::openRead("/sys/class/drm/card"s + std::to_string(pos) + "/device/unique_id")) {
      string uid = f.readLine();
      if (!uid.empty() && uid.back() == '\n') { uid.pop_back(); }
      if (wantUid == uid) { return pos; }
    }
  }
  log("Sysfs UID '%s' not found\n", wantUid.c_str());
  return -1;
}

string getUidFromSysfs(int pos) {
  File f = File::openRead("/sys/class/drm/card"s + std::to_string(pos) + "/device/unique_id");
  std::string uuid = f ? f.readLine() : "";
  if (!uuid.empty() && uuid.back() == '\n') { uuid.pop_back(); }
  return uuid;
}

/* BDF is PCIe Bus:Device.Function e.g. "6a:00.0" */
string getUidFromBdf(const string& bdf) {
  int pos = getSysfsFromBdf(bdf);
  return pos >= 0 ? getUidFromSysfs(pos) : "";
}

string getBdfFromUid(const string& uid) {
  int pos = getSysfsFromUid(uid);
  return (pos >= 0) ? getBdfFromSysfs(pos) : "";
}

int getPosFromBdf(const string& bdf) {
  auto openclIds = getAllDeviceIDs();
  for (int pos = 0; pos < int(openclIds.size()); ++pos) {
    if (bdf == getBdfFromDevice(openclIds[pos])) { return pos; }
  }
  
  log("OpenCL device with BDF '%s' not found\n", bdf.c_str());
  return -1;
}

string getBdfFromPos(int pos) {
  assert(pos >= 0);
  return getBdfFromDevice(getDevice(pos));
}

int getPosFromUid(const string& uid) {
  string bdf = getBdfFromUid(uid);
  if (bdf.empty()) { return -1; }
  return getPosFromBdf(bdf);
}

string getUidFromPos(int pos) {
  return getUidFromBdf(getBdfFromPos(pos));
}
