#pragma once

#include "util.hpp"

#include <pcl/common/common.h>
#include <pcl/surface/gp3.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"


void solve(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

class Optimizer {
    private:
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  TestBlockSolver;
    typedef g2o::LinearSolverCSparse<TestBlockSolver::PoseMatrixType> TestLinearSolver;
    g2o::SparseOptimizer test_optimizer;
    pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;

    public:
    Optimizer(InputParams &input_params) {
        g2o::OptimizationAlgorithmLevenberg* test_solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<TestBlockSolver>(g2o::make_unique<TestLinearSolver>()));
        test_optimizer.setAlgorithm(test_solver);

        gp3.setSearchRadius (input_params.patch_min);

        // Set typical values for the parameters
        gp3.setMu (2.5);
        gp3.setMaximumNearestNeighbors (100);
        gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
        gp3.setMinimumAngle(M_PI/18); // 10 degrees
        gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
        gp3.setNormalConsistency(false);
    }

    std::unordered_map<int, pcl::PointXYZ>
    optimize(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
            std::unordered_map<int, int> &prev2next,
            std::unordered_map<int, pcl::PointXYZ> &vectors,
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud);

};