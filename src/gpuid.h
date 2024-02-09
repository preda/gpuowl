// Copyright (C) 2017 - 2024 Mihai Preda

#pragma once

#include <string>

/*
We need to identify the GPU where we want to run the program.

OpenCL exposes just a list of devices; the most rudimentary way to identify a device is the position in this list.

If the OpenCL driver supports it, we are able to query the PCIe BDF identifier (looks like "0c:00.0") for a device, and
identify a device by the BDF ("Bus:Device.Function").
  
On Linux with AMDGPU/ROCm there is additional sysfs support which exposes a unique 64-bit identifier for each GPU.

Here we have functions that allow to establish the correspondence between these three distict identifiers: position in
OpenCL list, PCIe BDF, and unique-id (UID).

The sysfs files of interest are:
/sys/class/drm/card<N>/device/unique_id
/sys/class/drm/card<N>/device/uevent
the N above is what we call the "sysfs position".

It addition to these three identifiers (OpenCL position, UID, BDF), we have also the OpenCL cl_device_id which is an
opaque device identifier used at runtime by OpenCL.
*/

// Get PCIe BDF from the sysfs position.
std::string getBdfFromSysfs(int pos);

// Get sysfs position from PCIe BDF (Bus:Device.Function e.g. "6a:00.0")
int getSysfsFromBdf(const std::string& bdf);

// Get sysfs position from UID
int getSysfsFromUid(const std::string& uid);

// Get UID from sysfs position
std::string getUidFromSysfs(int pos);

// Get UID from BDF. (BDF is PCIe Bus:Device.Function e.g. "6a:00.0")
std::string getUidFromBdf(const std::string& bdf);

// Get BDF from UID
std::string getBdfFromUid(const std::string& uid);

// Get the position in the OpenCL enumeration from BDF
int getPosFromBdf(const std::string& bdf);

// Get BDF from position in the OpenCL device enumeration
std::string getBdfFromPos(int pos);

// Get the position in the OpenCL enumeration from UID
int getPosFromUid(const std::string& uid);

std::string getUidFromPos(int pos);
