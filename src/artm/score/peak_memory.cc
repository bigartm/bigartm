// Copyright 2017, Additive Regularization of Topic Models.

// Author: Alexander Frey (sashafrey@gmail.com)

#include <cmath>

#if defined(WIN32)
#include "windows.h"  // NOLINT
#include "psapi.h"    // NOLINT
#endif
#undef ERROR

#include "artm/core/common.h"
#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/peak_memory.h"

namespace artm {
namespace score {

std::shared_ptr<Score> PeakMemory::CalculateScore(const artm::core::PhiMatrix& p_wt) {
  PeakMemoryScore* peak_memory_score = new PeakMemoryScore();
  std::shared_ptr<Score> retval(peak_memory_score);

#if defined(WIN32)
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
  peak_memory_score->set_value((size_t)info.PeakWorkingSetSize);
#else
  // ToDo(ofrei): find and use some *nix tool for memory usage calculation
  peak_memory_score->set_value(0);
#endif

  return retval;
}

}  // namespace score
}  // namespace artm
