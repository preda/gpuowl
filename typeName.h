#pragma once

#include <string_view>
#include <string>

#if 0
template <typename T>
constexpr auto typeName() {
    std::string_view name, prefix, suffix;
#if defined(__clang__)
    name = __PRETTY_FUNCTION__;
    prefix = "auto type_name() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto type_name() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void)";
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}
#endif

template <typename T>
const char* typeName(T&& v) {
  const char* ret = typeid(v).name();
  size_t pos;
  stoi(ret, &pos);
  return ret + pos;
}
