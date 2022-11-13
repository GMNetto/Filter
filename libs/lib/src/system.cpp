#include "system.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <libalglib/stdafx.h>
#include <libalglib/optimization.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"

using namespace std;


class VertexTest : public g2o::VertexSE3
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    pcl::PointXYZRGBNormal p, q;
    bool mode;
    VertexTest()
    {
    }

    VertexTest(pcl::PointXYZRGBNormal p_): p(p_) 
    {
        mode = false;
    }

    VertexTest(pcl::PointXYZRGBNormal p_, pcl::PointXYZRGBNormal q_): p(p_), q(q_) 
    {
        mode = true;
    }


    virtual void oplusImpl(const number_t* update)
    {
        Eigen::Map<const g2o::Vector6> v(update);
        g2o::Isometry3 increment = g2o::internal::fromVectorMQT(v);
        //std::cout << "Vertex Increment: " << _numOplusCalls << "\n" << increment.matrix() << std::endl;
        _estimate = increment * _estimate;
    }
};


class EdgeTest : public g2o::BaseBinaryEdge<6, g2o::Isometry3, VertexTest, VertexTest> {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeTest()
    {
    }
    virtual bool read(std::istream& )
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }
    virtual bool write(std::ostream& ) const
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }

    void computeError()
    {
        const VertexTest* vt = static_cast<const VertexTest*>(vertex(0));
        const g2o::Isometry3 &transform_t = vt->estimate();
        

        //std::cout << "M1\n" << transform_t.matrix() << std::endl;

        const VertexTest* vn = static_cast<const VertexTest*>(vertex(1));
        const g2o::Isometry3 &transform_n = vn->estimate();

        //std::cout << "M2\n" << transform_n.matrix() << std::endl;

        //std::cout << "MT\n" << transform_n.linear().matrix() << std::endl;

        const g2o::Isometry3 result = transform_t.inverse() * transform_n;
        
        g2o::Vector6 compact_result = g2o::internal::toVectorMQT(result);
        float estimated_error = compact_result.squaredNorm();
        //std::cout << "Estimated E: " << estimated_error << std::endl;
        _error = compact_result;
    }
};

class EdgeNext : public g2o::BaseUnaryEdge<3, g2o::Isometry3, VertexTest> {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeNext()
    {
    }
    virtual bool read(std::istream& )
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }
    virtual bool write(std::ostream& ) const
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }

    void computeError()
    {
        const VertexTest* vt = static_cast<const VertexTest*>(vertex(0));
        const g2o::Isometry3 &transform = vt->estimate();

        const pcl::PointXYZRGBNormal &p = vt->p, &q = vt->q;
        Eigen::Vector3d v(p.x, p.y, p.z), e(q.x, q.y, q.z);
        Eigen::Vector3d r = transform * v;

        float err = (e - r).squaredNorm();
        //std::cout << "E: " << err << std::endl;
        _error = (e - r);
    }
};


void solve(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud) {

    //typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  TestBlockSolver;
    typedef g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType> TestLinearSolver;
    g2o::SparseOptimizer test_optimizer;
    //g2o::OptimizationAlgorithmLevenberg* test_solver = new g2o::OptimizationAlgorithmLevenberg(
    //    g2o::make_unique<TestBlockSolver>(g2o::make_unique<TestLinearSolver>()));
    auto linearSolver = g2o::make_unique<TestLinearSolver>();
    linearSolver->setBlockOrdering(false);

    auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* test_solver = new g2o::OptimizationAlgorithmLevenberg(
       std::move(blockSolver)
    );

    g2o::OptimizationAlgorithmGaussNewton* optimizationAlgorithm = new g2o::OptimizationAlgorithmGaussNewton(
        std::move(blockSolver)
    );
    
    test_optimizer.setAlgorithm(test_solver);
    /*
    for (int i = 0; i < 100; i++) {
        pcl::PointXYZRGBNormal p = cloud->points[i];
        VertexTest *test = new VertexTest(p);
        test->setId(i);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(9);
        test->setEstimate(v);
        test_optimizer.addVertex(test);

        EdgeNext *test_edge = new EdgeNext;
        test_edge->setVertex(0, test);
        test_optimizer.addEdge(test_edge);
    }
    */
    pcl::PointXYZRGBNormal p;
    p.x = 0;
    p.y = 0;
    p.z = 0;

    pcl::PointXYZRGBNormal n;
    n.x = 0;
    n.y = 0;
    n.z = 0.5;

    pcl::PointXYZRGBNormal p1, p2, p3, p4;
    p1.x = 0.5;
    p1.y = 0;
    p1.z = 0;

    p2.x = -0.5;
    p2.y = 0;
    p2.z = 0;

    p3.x = 0;
    p3.y = 0.5;
    p3.z = 0;

    p4.x = 0;
    p4.y = -0.5;
    p4.z = 0;

    VertexTest *test = new VertexTest(p, n);
    test->setId(0);
    g2o::Isometry3 v = g2o::Isometry3::Identity();
    auto &m = v.matrix();
    v(2, 3) = 0.5;
    test->setEstimate(v);
    test_optimizer.addVertex(test);

    VertexTest *neighbor1 = new VertexTest(p1), *neighbor2 = new VertexTest(p2), *neighbor3 = new VertexTest(p3), *neighbor4 = new VertexTest(p4);
    neighbor1->setId(1);
    neighbor2->setId(2);
    neighbor3->setId(3);
    neighbor4->setId(4);
    g2o::Isometry3 vn = g2o::Isometry3::Identity();
    neighbor1->setEstimate(vn);
    neighbor2->setEstimate(vn);
    neighbor3->setEstimate(vn);
    neighbor4->setEstimate(vn);
    test_optimizer.addVertex(neighbor1);
    test_optimizer.addVertex(neighbor2);
    test_optimizer.addVertex(neighbor3);
    test_optimizer.addVertex(neighbor4);

    EdgeNext *test_edge = new EdgeNext;
    test_edge->setVertex(0, test);
    test_edge->setInformation(Eigen::Matrix<double, 3, 3>::Identity());
    test_optimizer.addEdge(test_edge);

    EdgeTest *test_edge_neighbor1 = new EdgeTest, *test_edge_neighbor2 = new EdgeTest, *test_edge_neighbor3 = new EdgeTest, *test_edge_neighbor4 = new EdgeTest;
    test_edge_neighbor1->setVertex(0, test);
    test_edge_neighbor1->setVertex(1, neighbor1);
    test_edge_neighbor2->setVertex(0, test);
    test_edge_neighbor2->setVertex(1, neighbor2);
    test_edge_neighbor3->setVertex(0, test);
    test_edge_neighbor3->setVertex(1, neighbor3);
    test_edge_neighbor4->setVertex(0, test);
    test_edge_neighbor4->setVertex(1, neighbor4);

    //test_edge_neighbor1->setInformation(Eigen::Vector<double, 3>::Identity());
    test_edge_neighbor1->information().setIdentity();
    test_edge_neighbor2->information().setIdentity();
    test_edge_neighbor3->information().setIdentity();
    test_edge_neighbor4->information().setIdentity();
    test_optimizer.addEdge(test_edge_neighbor1);
    test_optimizer.addEdge(test_edge_neighbor2);
    test_optimizer.addEdge(test_edge_neighbor3);
    test_optimizer.addEdge(test_edge_neighbor4);

    // EdgeTest *test_edge_neighbor2 = new EdgeTest;
    // test_edge_neighbor2->setVertex(1, test);
    // test_edge_neighbor2->setVertex(0, neighbor);
    // test_edge_neighbor2->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    // test_optimizer.addEdge(test_edge_neighbor2);


    test_optimizer.initializeOptimization();
    test_optimizer.setVerbose(true);
    test_optimizer.optimize(5);

    std::cout << "First Transform:\n" << test->estimate().matrix() << std::endl;
    std::cout << "Second Transform:\n" << neighbor1->estimate().matrix() << std::endl;
}


// Include all points, not only parts of triangle
std::unordered_map<int, pcl::PointXYZ>
Optimizer::optimize(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2,
        std::unordered_map<int, int> &prev2next,
        std::unordered_map<int, pcl::PointXYZ> &vectors,
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr new_cloud) {
        
    if (cloud1->size() == 0) {
        pcl::copyPointCloud(*cloud1, *new_cloud);
        return vectors;
    }

    std::unordered_map<int, VertexTest*> vertices;

    test_optimizer.clear();

    std::vector<int> indices;
    std::vector<float> squared_distances;
    pcl::search::KdTree<pcl::PointXYZRGBNormal> tree;
    tree.setInputCloud (cloud1);

    for(auto iter = prev2next.begin(); iter != prev2next.end(); ++iter)
    {
        int vertex_idx =  iter->first;
        VertexTest *v;
        pcl::PointXYZRGBNormal p;

        pcl::PointXYZ &vector = vectors[vertex_idx];

        p = cloud1->points[vertex_idx];
        pcl::PointXYZRGBNormal n = cloud2->points[iter->second];
        v = new VertexTest(p, n);
        v->setId(vertex_idx);
        g2o::Isometry3 id = g2o::Isometry3::Identity();
        id.translation() = Eigen::Vector3d(vector.x, vector.y, vector.z);
        v->setEstimate(id);
        test_optimizer.addVertex(v);
        vertices[vertex_idx] = v;

        EdgeNext *e = new EdgeNext;
        e->setVertex(0, v);
        //e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        e->information().setIdentity();
        test_optimizer.addEdge(e);
    }

    pcl::PolygonMesh triangles;
    gp3.setInputCloud (cloud1);
    gp3.reconstruct (triangles);

    pcl::io::savePLYFile("/home/gustavo/filter/dewan/mesh.ply", triangles);

    std::cout << "Number Triangles: " << triangles.polygons.size() << std::endl;

    for (int i = 0; i < triangles.polygons.size(); i++) {

        int vertex_idx =  triangles.polygons[i].vertices[0];
        VertexTest *v;
        pcl::PointXYZRGBNormal p;
        bool exists = false;
        if (vertices.find(vertex_idx) == vertices.end()) {
            p = cloud1->points[vertex_idx];
            v = new VertexTest(p);
            v->setId(vertex_idx);
            g2o::Isometry3 id = g2o::Isometry3::Identity();
            v->setEstimate(id);
            test_optimizer.addVertex(v);
            vertices[vertex_idx] = v;
        } else {
            v = vertices[vertex_idx];
            exists = true;
        }
        for (int n = 1; n < triangles.polygons[i].vertices.size(); n++) {
            int neighbor = triangles.polygons[i].vertices[n];
            VertexTest *neighbor_vertex;
            if (vertices.find(neighbor) == vertices.end()) {
                pcl::PointXYZRGBNormal n = cloud1->points[neighbor];
                neighbor_vertex = new VertexTest(n);
                neighbor_vertex->setId(neighbor);
                g2o::Isometry3 id = g2o::Isometry3::Identity();
                if (exists) {
                    id = v->estimate();
                }
                neighbor_vertex->setEstimate(id);
                test_optimizer.addVertex(neighbor_vertex);
                vertices[neighbor] = neighbor_vertex;
            } else {
                neighbor_vertex = vertices[neighbor]; 
            }
            EdgeTest *edge_neighbor = new EdgeTest;
            edge_neighbor->setVertex(0, v);
            edge_neighbor->setVertex(1, neighbor_vertex);
            //edge_neighbor->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            edge_neighbor->information().setIdentity();
            test_optimizer.addEdge(edge_neighbor);          
        }
    }

    std::cout << "Optimizing" << std::endl;
    test_optimizer.initializeOptimization();
    //test_optimizer.setVerbose(true);
    test_optimizer.optimize(number_iter);
    std::cout << "Optimized" << std::endl;
    pcl::copyPointCloud(*cloud1, *new_cloud);

    for (int i = 0; i < new_cloud->size(); i++) {
        pcl::PointXYZINormal &p = new_cloud->points[i];
        p.normal_x = 0;
        p.normal_y = 0;
        p.normal_z = 0;
    }

    std::cout << "Input Cloud " << cloud1->size() << std::endl;
    std::cout << "Processed Cloud" << vertices.size() << std::endl;
    for (auto iter = vertices.begin(); iter != vertices.end(); ++iter) {
        pcl::PointXYZINormal &p = new_cloud->points[iter->first];
        const g2o::Isometry3 &transform = iter->second->estimate();
        Eigen::Vector3d v(p.x, p.y, p.z);
        Eigen::Vector3d r = transform * v;

        p.normal_x = r(0) - p.x;
        p.normal_y = r(1) - p.y;
        p.normal_z = r(2) - p.z;

        //std::cout << iter->first << " Transformed: " << r - v << std::endl;
    }

    return vectors;
}