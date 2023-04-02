#pragma once

#include <string_view>
#include <string>

template <typename T>
const char* typeName(T&& v) {
  const char* ret = typeid(v).name();
  try {
    size_t pos = 0;
    stoi(ret, &pos);
    return ret + pos;
  } catch (...) {
    return ret;
  }
}
