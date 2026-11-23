// Native ComfyUI exports carry explicit per-module alpha and no metadata.
// Reject other files here instead of pulling in the complete model converter.
#include "weights/quant/quant_convert.h"

namespace mi::quant {
namespace {
void normalized(const std::map<std::string, std::string>& metadata) {
  if (!metadata.empty()) throw QuantError("cache helper accepts normalized adapter files without metadata only");
}
}
std::optional<double> lora_metadata_alpha(const std::map<std::string, std::string>& metadata) {
  normalized(metadata);
  return std::nullopt;
}
std::string lora_adapter_config_format(const std::map<std::string, std::string>& metadata) {
  normalized(metadata);
  return {};
}
std::map<std::string, double> lora_adapter_config_alphas(const std::map<std::string, std::string>& metadata) {
  normalized(metadata);
  return {};
}
double lora_alpha_scale(std::optional<double> alpha, std::int64_t rank) {
  return alpha ? *alpha / static_cast<double>(rank) : 1.0;
}
void lora_merge_row(float*, std::int64_t, const float*, const float*, std::int64_t, float, double*) {
  throw QuantError("cache helper accepts LoKr only; low-rank adapters must remain on the runtime side branch");
}
}
