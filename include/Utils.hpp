#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
  os << "[";
  if(!v.empty())
    {
      auto iter = std::begin(v);
      os << *iter;
      ++iter;
      while(iter!=std::end(v))
	{
	  os << ", " << *iter;
	  ++iter;
	}
    }
  return os << "]";
}

#endif /* UTILS_H */
