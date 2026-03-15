// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/feature/index.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace colmap {
namespace {

class ExactFeatureDescriptorIndex : public FeatureDescriptorIndex {
 public:
  explicit ExactFeatureDescriptorIndex(int /*num_threads*/) {}

  void Build(const FeatureDescriptorsFloat& index_descriptors) override {
    type_ = index_descriptors.type;
    descriptors_ = index_descriptors.data;
  }

  void Search(int num_neighbors,
              const FeatureDescriptorsFloat& query_descriptors,
              Eigen::RowMajorMatrixXi& indices,
              Eigen::RowMajorMatrixXf& l2_dists) const override {
    THROW_CHECK_EQ(query_descriptors.type, type_);

    if (num_neighbors <= 0 || descriptors_.rows() == 0) {
      indices.resize(0, 0);
      l2_dists.resize(0, 0);
      return;
    }

    THROW_CHECK_EQ(query_descriptors.data.cols(), descriptors_.cols());
    const Eigen::Index num_query_descriptors = query_descriptors.data.rows();
    if (num_query_descriptors == 0) {
      return;
    }

    const Eigen::Index num_eff_neighbors =
        std::min<Eigen::Index>(num_neighbors, descriptors_.rows());
    if (num_eff_neighbors <= 0) {
      indices.resize(num_query_descriptors, 0);
      l2_dists.resize(num_query_descriptors, 0);
      return;
    }

    l2_dists.resize(num_query_descriptors, num_eff_neighbors);
    indices.resize(num_query_descriptors, num_eff_neighbors);

    std::vector<int> candidate_indices(descriptors_.rows());
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
    std::vector<float> candidate_dists(descriptors_.rows());

    for (Eigen::Index query_idx = 0; query_idx < num_query_descriptors;
         ++query_idx) {
      const auto query = query_descriptors.data.row(query_idx);
      for (Eigen::Index desc_idx = 0; desc_idx < descriptors_.rows();
           ++desc_idx) {
        candidate_dists[desc_idx] =
            (query - descriptors_.row(desc_idx)).squaredNorm();
      }

      if (num_eff_neighbors < descriptors_.rows()) {
        std::nth_element(
            candidate_indices.begin(),
            candidate_indices.begin() + num_eff_neighbors,
            candidate_indices.end(),
            [&](const int lhs, const int rhs) {
              if (candidate_dists[lhs] == candidate_dists[rhs]) {
                return lhs < rhs;
              }
              return candidate_dists[lhs] < candidate_dists[rhs];
            });
      }
      std::sort(candidate_indices.begin(),
                candidate_indices.begin() + num_eff_neighbors,
                [&](const int lhs, const int rhs) {
                  if (candidate_dists[lhs] == candidate_dists[rhs]) {
                    return lhs < rhs;
                  }
                  return candidate_dists[lhs] < candidate_dists[rhs];
                });

      for (Eigen::Index neighbor_idx = 0; neighbor_idx < num_eff_neighbors;
           ++neighbor_idx) {
        const int index = candidate_indices[neighbor_idx];
        indices(query_idx, neighbor_idx) = index;
        l2_dists(query_idx, neighbor_idx) = candidate_dists[index];
      }
    }
  }

 private:
  FeatureExtractorType type_ = FeatureExtractorType::UNDEFINED;
  FeatureDescriptorsFloatData descriptors_;
};

}  // namespace

std::unique_ptr<FeatureDescriptorIndex> FeatureDescriptorIndex::Create(
    Type type, int num_threads) {
  switch (type) {
    case Type::FAISS:
      return std::make_unique<ExactFeatureDescriptorIndex>(num_threads);
    default:
      throw std::runtime_error("Feature descriptor index not implemented");
  }
  return nullptr;
}

}  // namespace colmap
